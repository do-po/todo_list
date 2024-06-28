import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
from ray.air import session
import os
import multiprocessing
from refer.module.func import debug_message, save_model_with_versioning, load_latest_model, trial_dirname_creator
from datetime import datetime
import random

# 시스템 정보 가져오기
num_cpus = multiprocessing.cpu_count()
num_gpus = len(tf.config.list_physical_devices('GPU'))

# Ray 초기화
ray.init(ignore_reinit_error=True)

# 데이터 로드 및 전처리 (DB에서 데이터 가져오기로 수정필요)
file_path = './refer/output/'
debug_message("작업 데이터 로드 중...")
importance_data = pd.read_csv(f'{file_path}importance_data.csv')
urgency_data = pd.read_csv(f'{file_path}urgency_data.csv')
achievement_data = pd.read_csv(f'{file_path}achievement_data.csv')
daily_time_data = pd.read_csv(f'{file_path}daily_time_data.csv')
debug_message("작업 데이터 로드 완료")

# 입력 데이터 준비
importance_scores = importance_data['importance'].values
urgency_scores = urgency_data['urgency'].values
achievement_ratios = achievement_data['achievement_ratio'].values

# 오늘 날짜의 daily_time 설정
today_date = datetime.today().strftime('%Y-%m-%d')
daily_time_row = daily_time_data[daily_time_data['date'] == today_date]

if not daily_time_row.empty:
    daily_time = daily_time_row['daily_time'].values[0]
else:
    raise ValueError(f"No daily time data available for today ({today_date})")

X = np.vstack((importance_scores, urgency_scores, achievement_ratios)).T
y = daily_time_data['time_distribution'].values

# 데이터 정규화
debug_message("데이터 정규화 중...")
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 조건에 따라 y 값 정규화
normalize_y = True
if normalize_y:
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

debug_message("데이터 정규화 완료")

# 시간 분배 모델 생성 함수
def create_time_distribution_model(config):
    debug_message("모델 생성 중...")
    model = Sequential()
    model.add(Dense(config["hidden_layer1_size"], input_dim=X_train.shape[1], activation='relu'))
    if config["batch_norm"]:
        model.add(BatchNormalization())
    model.add(Dropout(config["dropout_rate"]))
    model.add(Dense(config["hidden_layer2_size"], activation='relu'))
    if config["batch_norm"]:
        model.add(BatchNormalization())
    model.add(Dropout(config["dropout_rate"]))
    model.add(Dense(1, activation='linear'))  # 최종 출력층은 선형 활성화 함수를 사용하여 실수 값을 출력

    optimizer_name = config["optimizer"]
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=config["lr"])
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    debug_message("모델 생성 완료")
    return model

# 학습 함수
def train_model(config, X_train, y_train, X_test, y_test):
    print("train_model 함수에서 config는 ", config)
    try:
        debug_message("모델 학습 시작...")
        model = create_time_distribution_model(config)
        log_dir = os.path.join("logs", f"trial_{tune.get_trial_id()[:8]}")
        os.makedirs(log_dir, exist_ok=True)  # 로그 디렉토리가 없으면 생성
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        tune_report_callback = TuneReportCallback({"mean_squared_error": "val_mse"})
        history = model.fit(X_train, y_train, 
                            epochs=config["epochs"], 
                            batch_size=config["batch_size"], 
                            validation_split=0.2, 
                            verbose=1,
                            callbacks=[tensorboard_callback, tune_report_callback])
        debug_message("모델 학습 완료")
        
        # 모델 평가
        debug_message("모델 평가 중...")
        train_loss = model.evaluate(X_train, y_train, verbose=1)
        val_loss = model.evaluate(X_test, y_test, verbose=1)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_test, y_pred_test)
        debug_message("모델 평가 완료")
        
        # 평가 결과 보고
        session.report({"mean_squared_error": val_mse, "train_loss": train_loss, "val_loss": val_loss, "train_mse": train_mse, "val_mse": val_mse})
    except Exception as e:
        debug_message(f"학습 중 오류 발생: {str(e)}")
        session.report({"mean_squared_error": float("inf"), "train_loss": float("inf"), "val_loss": float("inf"), "train_mse": float("inf"), "val_mse": float("inf")})

# Ray Tune을 사용한 하이퍼파라미터 최적화
def tune_model():
    config = {
        'lr': tune.choice([0.0001, 0.001, 0.01]),  # 학습률 결정: 학습 과정에서 가중치가 조정되는 속도
        'batch_size': tune.choice([16, 32, 64]),  # 배치 크기: 한 번의 훈련 반복에서 사용되는 샘플의 수
        'epochs': tune.choice([100, 200, 300]),  # 에포크 수: 전체 데이터셋을 훈련하는 반복 횟수
        'hidden_layer1_size': tune.choice([64, 128, 256]),  # 첫 번째 은닉층의 노드 수 결정: 모델의 복잡성 조절
        'hidden_layer2_size': tune.choice([32, 64, 128]),  # 두 번째 은닉층의 노드 수 결정: 모델의 복잡성 조절
        'dropout_rate': tune.choice([0.1, 0.2, 0.3]),  # 드롭아웃 비율: 과적합을 방지하기 위해 일부 뉴런을 무작위로 제외
        'batch_norm': tune.choice([True, False]),  # 배치 정규화 사용 여부: 학습을 안정화하고 가속화
        'optimizer': tune.choice(['adam', 'sgd', 'rmsprop']),  # 옵티마이저: 학습 과정에서 가중치를 업데이트하는 방법 결정
        'l2_lambda': tune.choice([0.0001, 0.001, 0.01]),  # L2 정규화: 가중치의 크기를 제한하여 과적합 방지
        'l1_lambda': tune.choice([0.0001, 0.001, 0.01]),  # L1 정규화: 가중치의 크기를 제한하여 과적합 방지
        'num_cpus': num_cpus,  # CPU 수 설정
        'num_gpus': num_gpus   # GPU 수 설정
    }
    # config에서 최대 에포크 값 추출
    max_epochs = max(config['epochs'].categories)

    # ASHA 스케줄러 설정: Asynchronous Successive Halving Algorithm
    scheduler = ASHAScheduler(
        # metric="mean_squared_error",  # 최적화할 메트릭
        # mode="min",  # 최소화할 것인지, 최대화할 것인지 설정
        max_t=max_epochs,  # 각 실험에서 실행할 최대 시간 또는 최대 스텝 (여기서는 에포크 수)
        grace_period=1,  # 각 실험을 종료하기 전에 최소한으로 실행할 시간 또는 스텝
        reduction_factor=2  # 리소스를 절감하기 위해 각 실험을 종료할 때마다 감소시킬 비율
    )
    
    # Ray Tune을 사용하여 하이퍼파라미터 최적화 수행
    debug_message("Ray Tune 하이퍼파라미터 최적화 시작...")
    analysis = tune.run(
        tune.with_parameters(train_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test),  # 학습 함수
        resources_per_trial={"cpu": config['num_cpus'], "gpu": config['num_gpus']},  # 각 시도에서 사용할 리소스 설정
        config=config,  # 하이퍼파라미터 설정
        num_samples=10,  # 샘플링 횟수: 각 설정으로 몇 번의 실험을 실행할지
        scheduler=scheduler,  # 스케줄러 설정
        verbose=1,  # 학습 과정 출력 레벨: 0은 출력 없음, 1은 진행 상태 막대 표시, 2는 자세한 로그 출력
        trial_dirname_creator=trial_dirname_creator,  # 디렉토리 이름 생성 함수
        metric= "mean_squared_error",
        mode='min'
    )
    debug_message("Ray Tune 하이퍼파라미터 최적화 완료")
    
    # 최적의 하이퍼파라미터 출력
    print("Best config: ", analysis.get_best_config(metric="mean_squared_error", mode="min"))
    return analysis.get_best_config(metric="mean_squared_error", mode="min"), analysis

if __name__ == "__main__":
    # 최적의 하이퍼파라미터 찾기
    debug_message("최적의 하이퍼파라미터 찾기 시작...")
    best_config, analysis = tune_model()
    debug_message("최적의 하이퍼파라미터 찾기 완료")
    
    debug_message("최적의 하이퍼파라미터로 모델 재학습 중...")
    best_model = create_time_distribution_model(best_config)
    best_model.fit(X_train, y_train, epochs=best_config['epochs'], batch_size=best_config['batch_size'], validation_split=0.2, verbose=1)
    debug_message("모델 재학습 완료")
    
    # 최종 평가 결과
    debug_message("최종 평가 결과 분석 중...")
    best_trial = analysis.get_best_trial("mean_squared_error", mode="min", scope="all")
    best_trained_model = create_time_distribution_model(best_trial.config)
    best_checkpoint_dir = analysis.get_best_checkpoint(best_trial)

    if best_checkpoint_dir:
        model_path = os.path.join(best_checkpoint_dir, "checkpoint")
        best_trained_model.load_weights(model_path)

    debug_message("최종 모델 평가 중...")
    train_loss = best_trained_model.evaluate(X_train, y_train, verbose=1)
    test_loss = best_trained_model.evaluate(X_test, y_test, verbose=1)

    y_pred_train = best_trained_model.predict(X_train)
    y_pred_test = best_trained_model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    log_dir = os.path.join("logs", "final_model")
    os.makedirs(log_dir, exist_ok=True)  # 로그 디렉토리가 없으면 생성
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    print(f'훈련 MSE: {train_mse:.4f}')
    print(f'테스트 MSE: {test_mse:.4f}')
    debug_message("최종 모델 평가 완료")
    
    # 모델 저장
    debug_message("모델 저장 중...")
    saved_filename = save_model_with_versioning(best_trained_model, file_path, "importance_model")
    debug_message(f"모델 저장 완료: {saved_filename}")
    
    print(f"Initial importance model created and saved as {saved_filename}.")

    # Ray 종료
    ray.shutdown()
