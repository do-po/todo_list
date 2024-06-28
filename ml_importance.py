import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from refer.module.database import MyDB
from refer.module.func import get_data, save_data

# 파일 경로 설정
file_path = './refer/output/'

# 새로운 데이터를 예측하고 결과를 데이터베이스에 저장하는 함수
def lgbm(_user_id, model_filename='initial_lightgbm_models_v7.pkl', table_name='ml'):
    # 모델 로드
    models = joblib.load(f'{file_path}{model_filename}')
    
    # user_info 테이블에서 데이터 로드
    user_info = get_data('user_info')
    
    # user_id에 해당하는 데이터 선택
    new_data = user_info[user_info['user_id'] == _user_id]
    
    if new_data.empty:
        print(f"No data found for user_id {_user_id}")
        return
    
    # 원-핫 인코딩 및 전처리 파이프라인 재사용
    categorical_features = ['gender', 'mbti', 'job']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 새로운 데이터에 전처리 적용
    new_data_preprocessed = preprocessor.fit_transform(new_data)
    
    # 각 타겟에 대해 예측 수행
    predictions = {}
    for target in models:
        model = models[target]
        predictions[target] = model.predict(new_data_preprocessed)
    
    # 예측 결과를 DataFrame으로 변환
    predictions_df = pd.DataFrame(predictions)
    
    # 원본 입력 데이터와 예측 결과를 병합
    results_df = pd.concat([new_data.reset_index(drop=True), predictions_df], axis=1)
    
    # 예측 결과를 데이터베이스에 저장
    save_data(results_df, table_name)
    
    print(f"Predictions for user_id {_user_id} saved to the {table_name} table.")

