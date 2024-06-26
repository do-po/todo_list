import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import os

# 디버깅 메시지 출력 함수
def debug_message(message):
    print(f"[디버깅] {message}")

# 모델 버져닝 함수
def save_model_with_versioning(model, file_path, base_filename):
    version = 1
    filename = f"{base_filename}_v{version}.h5"
    while os.path.exists(os.path.join(file_path, filename)):
        version += 1
        filename = f"{base_filename}_v{version}.h5"
    model.save(os.path.join(file_path, filename))
    return filename

# 데이터 로드 및 전처리
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
numeric_data = data[['age', 'work', 'edu', 'free_time', 'health', 'chores']].values
X = np.hstack((encoded_data, numeric_data))

# 데이터 정규화
debug_message("데이터 정규화 중...")
scaler = StandardScaler() # rubustscaler 고려해봐
X = scaler.fit_transform(X)
debug_message("데이터 정규화 완료")

# 목표 변수 정의 (중요도)
y = data[['work', 'edu', 'free_time', 'health', 'chores']].values

# 데이터를 학습용과 테스트용으로 분할
debug_message("데이터 학습용 및 테스트용 분할 중...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
debug_message("데이터 학습용 및 테스트용 분할 완료")

# 하이퍼파라미터 설정 # 미니 배치 경사 하강법 사용중이긴 한데 할까 말까 고려 해봐야 해
learning_rate = 0.001  # 학습률 결정: 학습 과정에서 가중치가 조정되는 속도
batch_size = 32        # 배치 크기: 한 번의 훈련 반복에서 사용되는 샘플의 수
epochs = 100           # 에포크 수: 전체 데이터셋을 훈련하는 반복 횟수
hidden_layer1_size = 128  # 첫 번째 은닉층의 노드 수 결정: 모델의 복잡성 조절
hidden_layer2_size = 64   # 두 번째 은닉층의 노드 수 결정: 모델의 복잡성 조절

# TensorFlow/Keras를 사용하여 신경망 모델 정의
debug_message("신경망 모델 정의 중...")
model = Sequential()
model.add(Dense(hidden_layer1_size, input_dim=X_train.shape[1], activation='relu'))  # ReLU 활성화 함수 사용: sigmoid, tanh, leakyrelu 얘기 해야지
model.add(Dense(hidden_layer2_size, activation='relu'))
model.add(Dense(y_train.shape[1])) # 출력 레이어는 1개
debug_message("신경망 모델 정의 완료")

# 모델 컴파일
debug_message("모델 컴파일 중...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')  # Adam 옵티마이저 사용: NAG 사용하는 NAdam, Momentum 사용하는 Adam 중 Adam 사용
debug_message("모델 컴파일 완료")

# 모델 학습
debug_message("모델 학습 중...")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
debug_message("모델 학습 완료")

# 모델 평가
debug_message("모델 평가 중...")
train_loss = model.evaluate(X_train, y_train)
test_loss = model.evaluate(X_test, y_test)
print(f'훈련 손실: {train_loss:.4f}')
print(f'테스트 손실: {test_loss:.4f}')

# 예측 및 평가 (평균 제곱 오차 사용)
debug_message("예측 및 평가 중...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f'훈련 MSE: {train_mse:.4f}')
print(f'테스트 MSE: {test_mse:.4f}')
debug_message("예측 및 평가 완료")

# 모델 저장
debug_message("모델 저장 중...")
saved_filename = save_model_with_versioning(model, file_path, "importance_model")
debug_message(f"모델 저장 완료: {saved_filename}")

print(f"Initial importance model created and saved as {saved_filename}.")
