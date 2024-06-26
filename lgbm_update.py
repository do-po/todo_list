import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
import joblib

# 디버깅 메시지 출력 함수
def debug_message(message):
    print(f"[디버깅] {message}")

# 파일 경로 설정
file_path = './refer/output/'

# 기존 모델 및 데이터 로드
debug_message("기존 모델과 데이터를 로드하는 중...")
models = joblib.load(f'{file_path}initial_lightgbm_models.pkl')
user_info = pd.read_csv(f'{file_path}random_user_info.csv')
schedule_1 = pd.read_csv(f'{file_path}random_schedule_1.csv')
base_weight = pd.read_csv(f'{file_path}base_weight.csv')
schedule_2 = pd.read_csv(f'{file_path}random_schedule_2.csv')
user_weight = pd.read_csv(f'{file_path}random_user_weight.csv')
debug_message("기존 모델과 데이터 로드 완료")

# 초기 y 값 설정 및 결합
debug_message("초기 y 값 설정 및 결합 중...")
y_df_1 = schedule_1[['base_weight_no']].merge(base_weight, left_on='base_weight_no', right_on='no')[['work', 'edu', 'free_time', 'health', 'chores']]
y_df_2 = schedule_2[['user_weight_no']].merge(user_weight, on='user_weight_no')[['work', 'edu', 'free_time', 'health', 'chores']]
debug_message("초기 y 값 설정 및 결합 완료")

# X 및 y 결합
debug_message("X 및 y 결합 중...")
X_combined = pd.concat([user_info, user_info])
y_combined = pd.concat([y_df_1, y_df_2])
debug_message("X 및 y 결합 완료")

# 원-핫 인코딩 및 전처리 파이프라인 설정
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

# 파이프라인 설정 및 모델 재학습
debug_message("모델 재학습 중...")
for target in y_combined.columns:
    debug_message(f"{target}에 대한 모델 훈련 중...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(random_state=42))
    ])
    pipeline.fit(X_combined, y_combined[target])
    models[target] = pipeline
    debug_message(f"{target}에 대한 모델 훈련 완료")

# 모델 저장 함수
def save_model_with_versioning(models, file_path, base_filename):
    version = 1
    filename = f"{base_filename}_v{version}.pkl"
    while os.path.exists(f"{file_path}{filename}"):
        version += 1
        filename = f"{base_filename}_v{version}.pkl"
    joblib.dump(models, f"{file_path}{filename}")
    return filename

# 업데이트된 모델 저장
debug_message("업데이트된 모델 저장 중...")
saved_filename = save_model_with_versioning(models, file_path, "updated_lightgbm_models")
debug_message(f"업데이트된 모델 저장 완료: {saved_filename}")

print(f"LightGBM models updated and saved as {saved_filename}.")
