# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 김연수
- 리뷰어 : 박혜원


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [△] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  - 대체로 다 잘 수행해주셨지만, 딥러닝 모델에서 동일한 전처리 조건 (TF-IDF) 에 대한 실험은 빠진 것 같습니다. 

- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 주석은 없지만, 이해하기 어려운 코드가 아니었습니다. 
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 네 명료합니다. 
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네
- [O] 코드가 간결한가요?
  > 네

# 딥러닝 모델 성능 개선 방안 
- 아래와 같은 간단한 Dense 모델 구조에서, 데이터셋에 TF-IDF 를 적용하여 모델을 돌려보니, Accuracy 	0.813446, F1 Score 	0.807571 정도가 나왔었습니다. 아래와 같이 딥러닝 모델을 구현해보시면 좀 더 성능이 나아질 것 같아서 제안드립니다. 

```python
# 모델 구조 

def model_dense(input_dim):
    model_dense = tf.keras.Sequential()
    model_dense.add(tf.keras.layers.Input(shape=(input_dim,)))
    model_dense.add(tf.keras.layers.Dense(256, activation='relu'))
    model_dense.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model_dense


```

