import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from datetime import datetime
from refer.module.func import debug_message, load_latest_model




# 데이터 로드
file_path = './refer/output/'
debug_message("사용자 정보 데이터를 불러오는 중...")
user_info = pd.read_csv(f'{file_path}random_user_info.csv')
debug_message("사용자 정보 데이터 로드 완료")

debug_message("schedule_2 데이터를 불러오는 중...")
schedule_2 = pd.read_csv(f'{file_path}random_schedule_2.csv')
debug_message("schedule_2 데이터 로드 완료")

debug_message("user_check 데이터를 불러오는 중...")
user_check = pd.read_csv(f'{file_path}user_check.csv') # 유저의 작업 달성 정도 or 여부를 받아오기
debug_message("user_check 데이터 로드 완료")

debug_message("user_time 데이터를 불러오는 중...")
user_time = pd.read_csv(f'{file_path}user_time.csv') # 유저의 하루 가용 시간을 받아오기
debug_message("user_time 데이터 로드 완료")

# 오늘 날짜의 daily_time 설정
today_date = datetime.today().strftime('%Y-%m-%d')
daily_time_row = user_time[user_time['date'] == today_date]

if not daily_time_row.empty:
    daily_time = daily_time_row['daily_time'].values[0]
else:
    raise ValueError(f"No daily time data available for today ({today_date})")

# 중요도 모델 및 긴급도 모델 로드
debug_message("중요도 모델 로드 중...")
importance_model = load_latest_model(file_path, "importance_model")
debug_message("중요도 모델 로드 완료")

debug_message("긴급도 모델 로드 중...")
urgency_model = load_latest_model(file_path, "urgency_model")
debug_message("긴급도 모델 로드 완료")

# 범주형 데이터에 대한 원-핫 인코딩
debug_message("범주형 데이터에 대한 원-핫 인코딩 중...")
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_data = encoder.fit_transform(user_info[['gender', 'job', 'mbti']]).toarray()
debug_message("범주형 데이터에 대한 원-핫 인코딩 완료")

# 인코딩된 데이터와 수치형 데이터를 결합
numeric_data = user_info[['age']].values
X_user_info = np.hstack((encoded_data, numeric_data)) # 이거 주석으로 설명 달기

# 데이터 정규화
debug_message("데이터 정규화 중...")
scaler = StandardScaler() # rubustscaler 고려해봐
X_user_info = scaler.fit_transform(X_user_info)
debug_message("데이터 정규화 완료")

# 중요도 예측
debug_message("중요도 예측 중...")
importance_scores = importance_model.predict(X_user_info)
debug_message("중요도 예측 완료")

# 긴급도 예측
debug_message("긴급도 예측 중...")
X_schedule = schedule_2[['start_date', 'end_date', 'complexity']].copy()
X_schedule['start_date'] = pd.to_datetime(X_schedule['start_date'])
X_schedule['end_date'] = pd.to_datetime(X_schedule['end_date'])
X_schedule['start_date'] = (X_schedule['start_date'] - X_schedule['start_date'].min()).dt.days
X_schedule['end_date'] = (X_schedule['end_date'] - X_schedule['end_date'].min()).dt.days
X_schedule = scaler.fit_transform(X_schedule)
urgency_scores = urgency_model.predict(X_schedule)
debug_message("긴급도 예측 완료")

# 중요도와 긴급도를 기반으로 작업 분배
debug_message("작업 분배 중...")
total_importance = np.sum(importance_scores)
total_urgency = np.sum(urgency_scores)
weights = (importance_scores / total_importance) + (urgency_scores / total_urgency)
weights /= np.sum(weights)  # 가중치의 합이 1이 되도록 정규화

# 작업 시간 분배
work_distribution = daily_time * weights.flatten()
debug_message(f"작업 시간 분배 완료: {work_distribution}")

# 유저의 작업 달성 정도를 반영한 재분배
debug_message("유저의 작업 달성 정도 반영한 재분배 중...")
achievement_ratios = user_check['achievement_ratio'].values
adjusted_distribution = work_distribution * achievement_ratios
adjusted_distribution /= np.sum(adjusted_distribution)  # 조정된 분배 시간의 합이 daily_time이 되도록 정규화
adjusted_work_distribution = daily_time * adjusted_distribution
debug_message(f"조정된 작업 시간 분배 완료: {adjusted_work_distribution}")

# 결과 출력
print(f"초기 작업 시간 분배: {work_distribution}")
print(f"조정된 작업 시간 분배: {adjusted_work_distribution}")
