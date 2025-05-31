# BONG DEEP RESEARCH

- [BONG DEEP RESEARCH](#bong-deep-research)
  - [라이브러리 `import`](#라이브러리-import)
  - [데이터 전처리](#데이터-전처리)
  - [$함수\_{구}현$](#함수_구현)
    - [활성화 함수](#활성화-함수)
      - [sigmoid 함수](#sigmoid-함수)
      - [softmax 함수](#softmax-함수)

## 라이브러리 `import`

- library import
  - numpy

  ```python
  import numpy as np
  ```

- 전체 library `import` 코드

  ```python
  import numpy as np
  import base64, io, PIL.Image
  from google.colab import output, widgets
  from IPython.display import HTML, Javascript, display
  import matplotlib.pyplot as plt
  ```

## 데이터 전처리

- mnist import
  - `mnist`는 손글씨(숫자) 데이터 셋이다.
  - 0~9 까지의 손글씨 숫자로 각각의 데이터를 구성한다.
  - 각 이미지는 `28*28` 픽셀의 흑백 이미지다.
  - 각 이미지에는 각 이미지가 나타내는 정답 데이터가 포함되어 있다.
  - `import` mnist 가져오기
    1. `from sklearn.datasets import fetch_openml`
    1. `mnist = fetch_openml('mnist_784', version=1, as_frame=False)`

    ```python
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    ```

- 입력데이터의 정규화
  - mnist의 각 샘플은 28*28(784) 픽셀 데이터를 담고 있다.
  - 각 픽셀은 0 ~ 255 범위의 값을 가진다.
  - 0 ~ 255 범위의 input은 초기 학습 단계에서 너무 큰 값이 가중합($z$)이 될 수 있다.
  - $sigmoid$ 함수는 입력의 절대값이 5 이상만 되어도 값이 0 또는 1에 거의 수렴하고 기울기도 0에 가까워진다.
  - 따라서 0 ~ 255 범위의 값을 적절한 범위로 조정해야 하는데, 이것을 **정규화**라고 한다.
  - 또, 컴퓨터가 다루는 숫자 자료형(예: `32bit float point`)은 작은 숫자를 다룰 때 오차가 적어진다.

- `min-max`정규화
  $$X_{scaled} = \frac{ X_{original}-X_{min}}{ X_{max} - X_{min}}$$
  - 값의 범위를 0 ~ 1 사이로 조정하는 정규화

- 과적합
  - 학습데이터에 지나치게 맞춘 모델은 학습 데이터를 잘 예측할 수는 있지만, 일반화 성능이 부족하다. 이것을 **과적합(overfitting)**이라고 한다.
  - 과적합 상태에서 학습데이터를 넣어 모델의 성능을 측정하면 모델의 성능이 매우 높게 측정된다. 그러나 실제 성능(일반화 성능)은 그렇지 않을 확률이 매우 높다.
  - 따라서 정확한 성능 측정을 위해서 학습에 사용하지 않는 data를 test에 사용하여야 한다.

- 데이터(`x`)
  1. 데이터 정규화: mnist는 $28*28$ 픽셀 각각에 0~255의 숫자로 검정색을 칠한다.
  1. `x`데이터를 $255.0$으로 나누어 줌으로써 데이터의 범위를 0 ~ 1로 정규화 한다.

- 정답레이블(`y`)
  1. 테스트와 학습 데이터 나누기
  1. 레이블(정답) 데이터는 one-hot-encoding 하기

  ```python
  from sklearn.datasets import fetch_openml
  mnist = fetch_openml('mnist_784', version=1, as_frame=False)

  # x는 input 데이터
  x = mnist.data.astype(np.float32) / 255.0   # 정규화
  x_train   =   x[:60000] # 0~59999번 까지는 학습 데이터로 저장
  x_test    =   x[60000:] # 60000 ~ 데이터는 테스트 데이터로 저장

  # y는 x에 대한 정답 데이터
  y = mnist.target.astype(np.int32)
  y_train   =   y[:60000] # 0~59999번 까지는 학습 데이터로 저장
  y_test    =   y[60000:] # 60000 ~  테스트 데이터로 저장

  # y 데이터의 원핫인코딩
  # 정답 레이블 클래스 수 저장
  num_classes = len(np.unique(y))
  y_train_one_hot = np.eye(num_classes)[y_train]
  y_test_one_hot  = np.eye(num_classes)[y_test]
  ```

## $함수_{구}현$

### 활성화 함수

- **sigmoid**: 은닉층의 활성화 함수
- **softmax**: 출력층의 활성화 함수

#### sigmoid 함수

**sigmoid** 함수 `sigmoid(z)`와 **sigmoid** 함수의 미분인 `diff_sigmoid(a)` 함수를 정의한다.

- `sigmoid(z)`
  - $e^x$은 `np.exp(x)`로 쓴다.
  - **오버플로우 방지**: 컴퓨터가 계산할 수 있는 숫자의 크기에는 한계가 있는데, 64비트 부동소숫점 기준으로 $e^{709}$ 이상의 숫자에서 **overflow**가 발생함. 그래서 아래와 같이 $-z > 709$인 경우 **overflow**를 방지하기 위해서 $e^-z$의 최댓값을 $e^{709}$로 제한한다.
    - `exp_arg = -1.0 * np.asarray(z)` `exp_arg`는 $-z$가 된다.
    - `clipped_exp_arg = np.minimum(exp_arg, 709.0)`는 $-z > 709$일 때 $-z = 709$
  - `def`

  ```python
  def sigmoid(z):
      exp_arg = -1.0 * np.asarray(z)
      clipped_exp_arg = np.minimum(exp_arg, 709.0)
      return 1.0 / (1.0 + np.exp(clipped_exp_arg))
  ```

- `diff_sigmoid(a)`
  - `a = sigmoid(z)`일 때 `a`를 미분한다.
  - `def`

  ```python
  def diff_sigmoid(a):
      return a * (1.0 - a)
  ```

#### softmax 함수

$$
  {Softmax}(Z) = \frac{1}{\sum}e^{z_i} \times
  \begin{pmatrix}
    e^{z_1} \\
    e^{z_2} \\
    \cdots
  \end{pmatrix}
$$

- overflow 방지
  - 이전 층의 벡터를 `Z`라고 할 때
  - `softmax(z)`와 마찬가지로 $Z_i$의 값이 709를 넘어가게 되면 $e^{z_i}$는 overflow가 일어난다.
  - overflow 방지를 위해 $Z_{max}$(Z의 최댓값)를 Z의 모든 원소에 대해 빼준다.
  -
    $$
      Z_{stable} = Z - Z_{max}
    $$
  - $Z_{max}$를 빼주어도 `softmax`함수의 출력값은 바뀌지 않지만, overflow는 방지할 수 있다.

$$
\begin{align}
 {Softmax}(Z - Z_{max})
  &=
  \frac{1}{\sum}e^{z_i - z_{max}} \times
      \begin{pmatrix}
        e^{z_1 - z_{max}} \\
        e^{z_2 - z_{max}} \\
        \cdots
      \end{pmatrix} \\
  &=
  \frac{e^ {-z_{max}}}{e^ {-z_{max}}} \times
  \frac{1}{\sum}e^{z_i} \times
      \begin{pmatrix}
        e^{z_1} \\
        e^{z_2} \\
        \cdots
      \end{pmatrix}\\
  &=
    \frac{1}{\sum}e^{z_i} \times
      \begin{pmatrix}
        e^{z_1} \\
        e^{z_2} \\
        \cdots
      \end{pmatrix}\\
\end{align}
$$

- `softmax(z)`
  - softmax 함수가 한 번에 여러 묶음의 데이터를 처리할 수 있도록 구현한다.
  - `stable_z = z - np.max(z, axis = 1, keepdims = True)`
