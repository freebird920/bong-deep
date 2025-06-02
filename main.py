from sklearn.datasets import fetch_openml
import os
import numpy as np


DATA_CACHE_DIR = "./data/mnist_data"  # 원하는 경로로 변경 가능

if not os.path.exists(DATA_CACHE_DIR):
    os.makedirs(DATA_CACHE_DIR)
    print(f"데이터 캐시 디렉토리 '{DATA_CACHE_DIR}'를 생성했습니다.")


# mnist fetch

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    data_home=DATA_CACHE_DIR,  # 이 경로에 데이터가 캐시(저장)됩니다.
)


# 데이터셋 전처리 및 저장

# x(input)데이터 전처리 및 저장
x = mnist.data.astype(np.float32)
# 데이터 정규화(min-max 정규화)
x = x / 255.0

# 학습 및 테스트 데이터 분리 저장
x_train = x[:60000]  # 학습 x 데이터
x_test = x[60000:]  # 테스트 x 데이터


# 정답(label) 데이터 전처리

y = mnist.target.astype(np.int32)  # y에 데이터 저장
classes_num = np.unique(y)  # [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
classes_num = len(classes_num)
# 원 핫 인코딩
y = np.eye(classes_num)[y]
# 테스트/ 트레인 데이터 분리
y_train = y[:60000]
y_test = y[60000:]
# 함수구현


# 가중합 함수 구현
def circulate_z(x, w, b):
    return np.dot(x, w) + b


# sigmoid 함수 구현
def sigmoid(z):
    arg = -1.0 * z
    stable_arg = np.minimum(709.0, arg)
    return 1.0 / (1.0 + np.exp(stable_arg))

print('함수현이 최예진을 좋아할 확률?!' , sigmoid(6))

# sigmoid 함수 미분 구현
def diff_sigmoid(a):
    return a * (1.0 - a)

def softmax(z): 
    stable_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(stable_z)
    return exp_z / sum(exp_z, axis=1, keepdims=True)




# 손실함수(categorical_crossentropy)구현
def categorical_crossentropy(y, y_hat):
    return
