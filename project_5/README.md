# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 김연수
- 리뷰어 : 신유진


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 넵, 하지만 모델을 더 활용해보지 못한 부분이 아쉽습니다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 넵, 잘 이해됩니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 없습니다, 코드 안에 구동원리를 주석으로 잘 설명해 주셔서 에러 유발 가능성은 없어보입니다. 
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 넵 이해하셨습니다. 
- [X] 코드가 간결한가요?
  > 넵 간결합니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 활용해볼 수 있는 다른 모델로는, 우리가 노드에서 배웠던 LSTM과 GRU가 있습니다.
공통적인 특징으로는 각 모델을 넣으신 후 Dropout을 활용해 볼 수 있겠습니다. 

model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim_lstm, input_length=maxlen))   # trainable을 True로 주면 Fine-tuning
model2.add(tf.keras.layers.LSTM(16))  # Embedding Layer에 LSTM을 넣음
model2.add(tf.keras.layers.Dropout(0.2))  
model2.add(tf.keras.layers.Dense(8, activation='sigmoid')) 
model2.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# GRU 레이어로 모델 설계
model3 = tf.keras.Sequential()
model3.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim_GRU, input_length = maxlen))
model3.add(tf.keras.layers.GRU(16))  # GRU state 벡터의 차원수 (변경가능)
model3.add(tf.keras.layers.Dropout(0.2))
model3.add(tf.keras.layers.Dense(8, activation='sigmoid'))
model3.add(tf.keras.layers.Dense(1, activation='sigmoid')) 

```

# 참고 링크 및 코드 개선
```python
# 저도 어제 검색하다가 발견한 사이트인데, 노드의 내용과 많이 비슷하면서 또 부연설명을 하고 있으므로 도움이 되시길 바랍니다. 
https://wikidocs.net/book/2155
```
