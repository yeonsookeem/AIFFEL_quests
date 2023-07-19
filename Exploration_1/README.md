# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 김연수
- 리뷰어 : 이효겸


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 이해되었습니다.
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 다른 입력값을 받지 않기 때문에 에러 유발가능성이 없어 보입니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 단락별 코드를 작성하여 제대로 이해하고 작성한것 같습니다.
- [O] 코드가 간결한가요?
  > 코드가 단락별로 잘 분류되어있고 간결하게 작성되어 있습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 당뇨수치 회귀분석 코드
from sklearn.datasets import load_diabetes
# 당뇨 데이터 로드
diabetes=load_diabetes()
[5]
# 입력데이터와 타겟 데이터 분리
df_X=diabetes.data
df_y=diabetes.target
[31]
import numpy as np
# numpy array로 변환
X = np.array(df_X)
y = np.array(df_y)
[32]
print(X.shape)
print(y.shape)

[33]
from sklearn.model_selection import train_test_split
# 학습데이터와 테스트 데이터를 8:2로 분류
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

[47]
import numpy as np
# 특성 개수 10개와 편차 1개 값 랜덤으로 생성
W = np.random.rand(10)
b = np.random.rand(1)
[48]
W
[49]
# 모델 
def model(X, W, b):
    # 예측 초기값 0
    predictions = 0
    # 특성 개수가 10개 이므로 각 특성 마다 가중치 곱한것을 더하기 위해 포문 사용
    for i in range(10):
        # X에 인덱스 i열을 가져와 해당되는 가중치를 곱하여 더해 나감
        predictions += X[:, i] * W[i]
    # 마지막으로 상수 값 더함
    predictions += b
    # 최종 결과 리턴
    return predictions
[50]
# MSE 계산
def MSE(a, b):
    # a는 예측 결과데이터, b는 비교할 결과데이터
    # 결과 데이터간의 차이를 제곱하여 평균을 냄 = MSE
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse
[51]
def loss(X, W, b, y):
    # 모델을 통하여 X입력데이터에 대한 타겟데이터 리스트 predictions를 리런받음
    predictions = model(X, W, b)
    # 예측한 결과값과 정답 결과값을 MSE함수에 넣어 MSE로스값을 리턴 받음
    L = MSE(predictions, y)
    return L
[52]
def gradient(X, W, b, y):
    # N은 데이터 포인트의 개수 - 입력데이터 개수
    N = len(y)
    
    # y_pred 준비 - X에 대한 예측값
    y_pred = model(X, W, b)
    
    # 공식에 맞게 gradient 계산 - 기울기 계산
    dW = 1/N * 2 * X.T.dot(y_pred - y)
        
    # b의 gradient 계산 - 오차 평균 계산
    db = 2 * (y_pred - y).mean()
    return dW, db
[67]
# 학습률
LEARNING_RATE = 0.1
[68]
# 로스값들을 넣어 확인하기 위하여 losses리스트를 선언
losses = []

# 1000번의 epoch를 돌리기 위하여 range(1, 10001)의 포문 사용
for i in range(1, 1001):
    # 기울기 계산
    dW, db = gradient(X_train, W, b, y_train)
    # 학습률과 가중치를 곱하여 업데이트
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    # 로스값 계산
    L = loss(X_train, W, b, y_train)
    # 로스값을 리스트에 넣고 10번 돌아갈때마다 로스값 확인
    losses.append(L)
    if i % 10 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))
[69]
# 테스트입력값에 대한 결과값 리턴 받음
prediction = model(X_test, W, b)
# mse계산
mse = loss(X_test, W, b, y_test)
mse
# 최종 3000이하 값 출력
2865.029834788191
[70]
import matplotlib.pyplot as plt
[71]
# 입력데이터에 대한 0번째 특성을 가져와 그래프로 비교
plt.scatter(X_test[:, 0], y_test)
plt.scatter(X_test[:, 0], prediction)
plt.show()
```
---
```python
# 환경에 따른 자전거 타는 사람의 수 회귀분석
import pandas as pd

# 데이터 로드
train = pd.read_csv('/aiffel/data/data/bike-sharing-demand/train.csv')

# datetime 컬럼을 datetime 자료형으로 변환
train['datetime'] = pd.to_datetime(train['datetime'])

# 연, 월, 일, 시, 분, 초를 포함하는 6가지 컬럼 생성 - datetime에서 각 년 월 일 시 분 초로 컬럼 분리 
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second

# 결과 출력
print(train)

[32]
# 그래프 그리기 위한 도구 import
import seaborn as sns
import matplotlib.pyplot as plt
[77]
# 2행 3열로 그래프 자리 잡고 각 년 월 일 시 분 초 별로 카운트 그래프 생성
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
sns.countplot(x='year', data=train, ax=axes[0, 0])
sns.countplot(x='month', data=train, ax=axes[0, 1])
sns.countplot(x='day', data=train, ax=axes[0, 2])
sns.countplot(x='hour', data=train, ax=axes[1, 0])
sns.countplot(x='minute', data=train, ax=axes[1, 1])
sns.countplot(x='second', data=train, ax=axes[1, 2])

# 타이틀 설정
plt.tight_layout()
axes[0, 0].set_title('Year')
axes[0, 1].set_title('Month')
axes[0, 2].set_title('Day')
axes[1, 0].set_title('Hour')
axes[1, 1].set_title('Minute')
axes[1, 2].set_title('Second')
# 그래프 보여주기
plt.show()

[105]
# 입력데이터 값으로 년 월 일 시 계절 주말 평일 날씨 온도 습도를 선택
X = train[['year', 'month', 'day', 'hour', 'season', 'workingday', 'holiday', 'weather', 'temp', 'humidity']].values
# 결과데이터 값으로 자전거를 탄 카운트를 선택
y = train['count'].values
[111]
from sklearn.model_selection import train_test_split
# 훈련데이터와 테스트 데이터를 8:2로 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
[112]
from sklearn.linear_model import LinearRegression
# 선형Regression 선언
model = LinearRegression()
[113]
# 모델 훈련
model.fit(X_train, y_train)

# ???
LinearRegression()
[114]
# 학습된 모델 결과 값 확인
predictions = model.predict(X_test)
predictions
array([243.938436  ,  49.06955941, 207.84499101, ..., 305.83755265,
        40.07789828, 242.40528795])
[115]
from sklearn.metrics import mean_squared_error
import numpy as np
# mse 값을 구하고 루트를 씌워 rmse값  141로 150 이하 확인
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
Mean Squared Error (MSE): 19985.171918721586
Root Mean Squared Error (RMSE): 141.3689213325248

[116]
# 훈련데이터의 온도 카운트 값을 그래프로 보여줌
plt.scatter(train['temp'], train['count'])
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()

# 훈련데이터의 습도 카운트 값을 그래프로 보여줌
plt.scatter(train['humidity'], train['count'])
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.show()
```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
전체 적으로 코드가 간결하나 상수 값들에 대한 변수 지정이 필요해 보임
변수 지정으로 인해 해당 값을 수정하거나 가독성이 편해짐


해당 코드에는 잘 작성되어 있으나 함수 밖의 변수명과 함수 매개변수의 변수명을 동일하게 할 시 코드 꼬임의 발생 여지가 생겨
작성 간 오류가 생길 가능성이 높음

자전거 카운트 회귀 분석에 경우

입력데이터가 년도는 크게 상관이 없어 보이고
달과 계절, 일과 평일과 주말 들의 상관관계가 서로 높아 그 중 하나만 택하는 것이 좋아 보임
마지막 그래프가 훈련데이터만 입력값으로 한 그래프를 그렸으므로 
모델의 결과 값에 대한 비교를 할 수 없음
```
