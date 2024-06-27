import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import os
import multiprocessing
from refer.module.func import debug_message, save_model_with_versioning




# 데이터 로드 및 전처리 (DB에서 데이터 가져오기로 수정필요)
file_path = './refer/output/'
debug_message("작업 중요도 데이터 로드 중...")
data = pd.read_csv(f'{file_path}data.csv')
debug_message("작업 중요도 데이터 로드 완료")

# 범주형 데이터에 대한 원-핫 인코딩
debug_message("범주형 데이터 원-핫 인코딩 중...")
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data[['gender', 'job', 'mbti']]).toarray()
debug_message("범주형 데이터 원-핫 인코딩 완료")

# 인코딩된 데이터와 수치형 데이터를 결합
numeric_data = data[['age']].values  # 나이 데이터는 별도로 결합
X = np.hstack((encoded_data, numeric_data))

# 데이터 정규화
debug_message("데이터 정규화 중...")
scaler = StandardScaler()  # RobustScaler를 사용할 수도 있음
X = scaler.fit_transform(X)
debug_message("데이터 정규화 완료")

# 목표 변수 정의 (중요도)
y = data[['work', 'edu', 'free_time', 'health', 'chores', 'category_else']].values

# 데이터를 학습용과 테스트용으로 분할
debug_message("데이터 학습용 및 테스트용 분할 중...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
debug_message("데이터 학습용 및 테스트용 분할 완료")

# 모델 생성 함수
def create_model(learning_rate=0.001, hidden_layer1_size=128, hidden_layer2_size=64, dropout_rate=0.2, batch_norm=False, optimizer='adam', kernel_size=(3, 3), l2_lambda=0.01, l1_lambda=0.01):
    model = Sequential()
    # 첫 번째 은닉층: 입력 차원 지정, 활성화 함수로 ReLU 사용, L1과 L2 정규화 추가
    model.add(Dense(hidden_layer1_size, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
    # 배치 정규화 (선택적)
    if batch_norm:
        model.add(BatchNormalization())
    # 드롭아웃 (과적합 방지)
    model.add(Dropout(dropout_rate))
    # 두 번째 은닉층: 활성화 함수로 ReLU 사용, L1과 L2 정규화 추가
    model.add(Dense(hidden_layer2_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
    # 배치 정규화 (선택적)
    if batch_norm:
        model.add(BatchNormalization())
    # 드롭아웃 (과적합 방지)
    model.add(Dropout(dropout_rate))
    # 출력층: 회귀 문제이므로 활성화 함수 없이 노드 수는 목표 변수 수와 동일
    model.add(Dense(y_train.shape[1]))
    # 옵티마이저 설정 및 컴파일
    model.compile(optimizer=tf.keras.optimizers.get(optimizer)(learning_rate=learning_rate), loss='mse')
    return model

# KerasRegressor로 래핑
model = KerasRegressor(build_fn=create_model, verbose=0)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],  # 학습률 결정: 학습 과정에서 가중치가 조정되는 속도
    'batch_size': [16, 32, 64],  # 배치 크기: 한 번의 훈련 반복에서 사용되는 샘플의 수
    'epochs': [50, 100, 200],  # 에포크 수: 전체 데이터셋을 훈련하는 반복 횟수
    'hidden_layer1_size': [64, 128, 256],  # 첫 번째 은닉층의 노드 수 결정: 모델의 복잡성 조절
    'hidden_layer2_size': [32, 64, 128],  # 두 번째 은닉층의 노드 수 결정: 모델의 복잡성 조절
    'dropout_rate': [0.1, 0.2, 0.3],  # 드롭아웃 비율: 과적합을 방지하기 위해 일부 뉴런을 무작위로 제외
    'batch_norm': [True, False],  # 배치 정규화 사용 여부: 학습을 안정화하고 가속화
    'optimizer': ['adam', 'sgd', 'rmsprop'],  # 옵티마이저: 학습 과정에서 가중치를 업데이트하는 방법 결정
    'kernel_size': [(3, 3), (5, 5)],  # 커널 크기: 컨볼루션 층에서 사용할 커널의 크기
    'l2_lambda': [0.01, 0.001, 0.0001],  # L2 정규화: 가중치의 크기를 제한하여 과적합 방지
    'l1_lambda': [0.01, 0.001, 0.0001]  # L1 정규화: 가중치의 크기를 제한하여 과적합 방지
}

# 멀티 프로세싱을 사용한 GridSearchCV
debug_message("GridSearchCV를 사용한 하이퍼파라미터 튜닝 중...")
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=multiprocessing.cpu_count())
grid_result = grid.fit(X_train, y_train)
debug_message("하이퍼파라미터 튜닝 완료")

# 최적의 하이퍼파라미터 출력
best_params = grid_result.best_params_
print(f"최적의 하이퍼파라미터: {best_params}")

# 최적의 하이퍼파라미터로 모델 재학습
debug_message("최적의 하이퍼파라미터로 모델 재학습 중...")
best_model = create_model(learning_rate=best_params['learning_rate'],
                          hidden_layer1_size=best_params['hidden_layer1_size'],
                          hidden_layer2_size=best_params['hidden_layer2_size'],
                          dropout_rate=best_params['dropout_rate'],
                          batch_norm=best_params['batch_norm'],
                          optimizer=best_params['optimizer'],
                          kernel_size=best_params['kernel_size'],
                          l2_lambda=best_params['l2_lambda'],
                          l1_lambda=best_params['l1_lambda'])

best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.2)
debug_message("모델 재학습 완료")

# 모델 평가
debug_message("모델 평가 중...")
train_loss = best_model.evaluate(X_train, y_train)
test_loss = best_model.evaluate(X_test, y_test)
print(f'훈련 손실: {train_loss:.4f}')
print(f'테스트 손실: {test_loss:.4f}')

# 예측 및 평가 (평균 제곱 오차 사용)
debug_message("예측 및 평가 중...")
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f'훈련 MSE: {train_mse:.4f}')
print(f'테스트 MSE: {test_mse:.4f}')
debug_message("예측 및 평가 완료")

# 모델 저장
debug_message("모델 저장 중...")
saved_filename = save_model_with_versioning(best_model, file_path, "importance_model")
debug_message(f"모델 저장 완료: {saved_filename}")

print(f"Initial importance model created and saved as {saved_filename}.")
