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
x_train = x[:60_000]  # 학습 x 데이터
x_test = x[60_000:]  # 테스트 x 데이터


# 정답(label) 데이터 전처리

y = mnist.target.astype(np.float32)  # y에 데이터 저장
classes_num = np.unique(y)  # [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]

classes_num = len(classes_num)
print(classes_num)

y_train = y[:60_000]  # 학습 레이블
y_test = y[60_000]  # 테스트 레이블
