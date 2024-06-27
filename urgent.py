import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
from refer.module.func import debug_message, save_model_with_versioning


# 데이터 로드 및 전처리
file_path = './refer/output/'
debug_message("작업 긴급도 데이터 로드 중...")
data = pd.read_csv(f'{file_path}data.csv')
debug_message("작업 긴급도 데이터 로드 완료")

# 초기 긴급도 기준 생성
debug_message("긴급도 관련 데이터 생성 중...")
data['start_date'] = pd.to_datetime(data['start_date'])
data['end_date'] = pd.to_datetime(data['end_date'])
data['days_left'] = (data['end_date'] - data['start_date']).dt.days
data['urgency'] = 1 / (data['days_left'] + 1)  # 마감일이 가까울수록 긴급도가 높아짐
debug_message("긴급도 관련 데이터 생성 완료")

# 긴급도 모델을 위한 입력 데이터 준비
X_urgency = data[['start_date', 'end_date', 'complexity']].copy()
X_urgency['start_date'] = (X_urgency['start_date'] - X_urgency['start_date'].min()).dt.days
X_urgency['end_date'] = (X_urgency['end_date'] - X_urgency['end_date'].min()).dt.days
y_urgency = data['urgency'].values

# 데이터 정규화
debug_message("데이터 정규화 중...")
scaler = StandardScaler() # rubustscaler 고려해봐
X_urgency = scaler.fit_transform(X_urgency)
debug_message("데이터 정규화 완료")

# 데이터를 학습용과 테스트용으로 분할
debug_message("데이터 학습용 및 테스트용 분할 중...")
X_train_urgency, X_test_urgency, y_train_urgency, y_test_urgency = train_test_split(X_urgency, y_urgency, test_size=0.2, random_state=42)
debug_message("데이터 학습용 및 테스트용 분할 완료")

# 하이퍼파라미터 설정
learning_rate = 0.001  # 학습률 결정: 학습 과정에서 가중치가 조정되는 속도
batch_size = 32        # 배치 크기 결정: 한 번의 훈련 반복에서 사용되는 샘플의 수
epochs = 100           # 에포크 수 결정: 전체 데이터셋을 훈련하는 반복 횟수
hidden_layer1_size = 128  # 첫 번째 은닉층의 노드 수 결정: 모델의 복잡성 조절
hidden_layer2_size = 64   # 두 번째 은닉층의 노드 수 결정: 모델의 복잡성 조절

# TensorFlow/Keras를 사용하여 신경망 모델 정의
debug_message("신경망 모델 정의 중...")
urgency_model = Sequential()
urgency_model.add(Dense(hidden_layer1_size, input_dim=X_train_urgency.shape[1], activation='relu'))  # ReLU 활성화 함수 사용: 비선형성을 추가하여 모델의 학습 능력 향상
urgency_model.add(Dense(hidden_layer2_size, activation='relu'))
urgency_model.add(Dense(1))
debug_message("신경망 모델 정의 완료")

# 모델 컴파일
debug_message("모델 컴파일 중...")
urgency_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')  # Adam 옵티마이저 사용: NAdam도 
debug_message("모델 컴파일 완료")

# 모델 학습
debug_message("모델 학습 중...")
urgency_model.fit(X_train_urgency, y_train_urgency, epochs=epochs, batch_size=batch_size, validation_split=0.2)
debug_message("모델 학습 완료")

# 모델 평가
debug_message("모델 평가 중...")
train_loss_urgency = urgency_model.evaluate(X_train_urgency, y_train_urgency)
test_loss_urgency = urgency_model.evaluate(X_test_urgency, y_test_urgency)
print(f'훈련 긴급도 손실: {train_loss_urgency:.4f}')
print(f'테스트 긴급도 손실: {test_loss_urgency:.4f}')

# 예측 및 평가 (평균 제곱 오차 사용)
debug_message("예측 및 평가 중...")
y_pred_train_urgency = urgency_model.predict(X_train_urgency)
y_pred_test_urgency = urgency_model.predict(X_test_urgency)
train_mse_urgency = mean_squared_error(y_train_urgency, y_pred_train_urgency)
test_mse_urgency = mean_squared_error(y_test_urgency, y_pred_test_urgency)
print(f'훈련 긴급도 MSE: {train_mse_urgency:.4f}')
print(f'테스트 긴급도 MSE: {test_mse_urgency:.4f}')
debug_message("예측 및 평가 완료")

# 모델 저장
debug_message("모델 저장 중...")
saved_filename = save_model_with_versioning(urgency_model, file_path, "urgency_model")
debug_message(f"모델 저장 완료: {saved_filename}")

print(f"Initial urgency model created and saved as {saved_filename}.")
