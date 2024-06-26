import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
import joblib
from joblib import parallel_backend

# 디버깅 메시지 출력 함수
def debug_message(message):
    print(f"[디버깅] {message}")

# 파일 경로 설정
file_path = './refer/output/'
evaluation_results_path = f'{file_path}model_evaluation_results.csv'

# 1. 초기 모델 생성용 데이터 로드
debug_message("사용자 정보 데이터를 불러오는 중...")
user_info = pd.read_csv(f'{file_path}user_info.csv')
debug_message("사용자 정보 데이터 로드 완료")

debug_message("schedule_2 데이터를 불러오는 중...")
schedule_2 = pd.read_csv(f'{file_path}schedule_2.csv')
debug_message("schedule_2 데이터 로드 완료")

debug_message("user_weight 데이터를 불러오는 중...")
user_weight = pd.read_csv(f'{file_path}user_weight.csv')
debug_message("user_weight 데이터 로드 완료")

# 데이터프레임의 열 이름 출력
debug_message(f"user_weight 열 이름: {user_weight.columns}")
debug_message(f"schedule_2 열 이름: {schedule_2.columns}")

# 2. y 값 설정
debug_message("y 값 설정 중...")
y_df = schedule_2[['user_weight_no']].merge(
    user_weight, on='user_weight_no'
)[['work', 'edu', 'free_time', 'health', 'chores']]
debug_message("y 값 설정 완료")

# 3. 데이터 분할
debug_message("데이터 분할 중...")
X = user_info[['age', 'gender', 'mbti', 'job']]
X_train, X_test, y_train, y_test = train_test_split(X, y_df, test_size=0.2, random_state=42)
debug_message("데이터 분할 완료")

# 4. 원-핫 인코딩 및 전처리 파이프라인 설정
debug_message("원-핫 인코딩 및 전처리 파이프라인 설정 중...")
categorical_features = ['gender', 'mbti', 'job']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
debug_message("원-핫 인코딩 및 전처리 파이프라인 설정 완료")

# 5. 각 타겟에 대한 모델 훈련 및 저장
debug_message("하이퍼파라미터 그리드 설정 중...")

param_grid = {
    'regressor__n_estimators': [300],
    'regressor__learning_rate': [0.0065],
    'regressor__num_leaves': [60],
    'regressor__max_depth': [10],
    'regressor__min_child_samples': [17],
    'regressor__feature_fraction': [0.73],
    'regressor__bagging_fraction': [0.73],
    'regressor__bagging_freq': [9],
    'regressor__lambda_l1': [0.008]
}

debug_message("하이퍼파라미터 그리드 설정 완료")

# 평가 결과를 저장할 데이터프레임을 불러오거나 새로 생성
def get_model_evaluation_results():
    if os.path.exists(evaluation_results_path):
        return pd.read_csv(evaluation_results_path)
    else:
        return pd.DataFrame(columns=['model', 'version', 'target', 'mean_score', 'std_score', 'best_params', 'best_score'])

results_df = get_model_evaluation_results()

best_models = {}
new_results = []
with parallel_backend('threading'):
    for target in y_df.columns:
        debug_message(f"{target}에 대한 모델 훈련 중...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LGBMRegressor(random_state=42))
        ])
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,  
            scoring='neg_mean_squared_error'
        )
        grid_search.fit(X_train, y_train[target])

        # 각 타겟에 대한 최적 모델 저장
        best_models[target] = grid_search.best_estimator_

        # 모든 평가 점수 출력 및 데이터프레임에 추가
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        mean_score = np.mean(means)

        new_results.append({
            'model': 'LightGBM', 
            'version': 0,  # 갱신
            'target': target, 
            'mean_score': mean_score, 
            'std_score': np.mean(stds), 
            'best_params': best_params, 
            'best_score': best_score
        })

        # 최적 모델의 평가 점수 출력
        debug_message(f"{target} 최적 모델 - Mean: {grid_search.best_score_:.3f}, Params: {grid_search.best_params_}")
        debug_message(f"{target}에 대한 모델 훈련 완료")

# 모델 저장 함수
def save_model_with_versioning(models, file_path, base_filename):
    version = 1
    filename = f"{base_filename}_v{version}.pkl"
    while os.path.exists(f"{file_path}{filename}"):
        version += 1
        filename = f"{base_filename}_v{version}.pkl"
    joblib.dump(models, f"{file_path}{filename}")
    return version, filename

# 모든 최적 모델을 하나의 딕셔너리에 저장하고, 그 딕셔너리를 파일로 저장
debug_message("모델 저장 중...")
version, saved_filename = save_model_with_versioning(best_models, file_path, "initial_lightgbm_models")
debug_message(f"모델 저장 완료: {saved_filename}")

# 저장된 모델 버전 업데이트
for result in new_results:
    result['version'] = version

# 새 결과를 데이터프레임으로 변환하여 기존 결과에 추가
new_results_df = pd.DataFrame(new_results)
results_df = pd.concat([results_df, new_results_df], ignore_index=True)

# 평가 결과를 CSV 파일로 저장
results_df.to_csv(evaluation_results_path, index=False)

print(f"Initial LightGBM models created and saved as {saved_filename}.")
