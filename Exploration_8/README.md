# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 김연수
- 리뷰어 : 김민식


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네. 각 단계별로 정상동작하는 내역을 확인하였습니다.
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네. 특히 함수 구성 부분에서 개념적으로 추가 정리를 잘 해주셨습니다.
  ```python
  # 해당 함수 부분을 예시로 가저옴
  def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # Self-Attention
    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
        })
    
    # Add & Normalize
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    # Encoder-Decoder Attention
    attention2 = MultiHeadAttention(
    d_model, num_heads, name="attention_2")(inputs={
        'query': attention1,  # Decoder의 다음 단어를 찾아야 하기에 연관성 기준을 Decoder 값으로 지정
        'key': enc_outputs,   # 찾아야하는 정보는 Encoder가 처리한 데이터에 있으니 key와 value 모두 Encoder 값으로 지정
        'value': enc_outputs,
        'mask': padding_mask
    })
    
    # Add & Normalize
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    # Feed Forward
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # Add & Normalize
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)
  ```
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 각 단계별로 정상작동하는 내역을 혹인하였습니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 각 단계에서 뭘 구현하려고 하신건지 잘 작성해주셨고(`주석`) 특히 개념(`수식` 등)을 정리해주신 부분이 인상적이었습니다.
- [O] 코드가 간결한가요?
  > 네. 함수/클래스로 모듈화 잘 되어 있습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 사칙 연산 계산기
class calculator:
    # 예) init의 역할과 각 매서드의 의미를 서술
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    # 예) 덧셈과 연산 작동 방식에 대한 서술
    def add(self):
        result = self.first + self.second
        return result

a = float(input('첫번째 값을 입력하세요.')) 
b = float(input('두번째 값을 입력하세요.')) 
c = calculator(a, b)
print('덧셈', c.add()) 
```

# 참고 링크 및 코드 개선
개인적으로는 말도 안되는 문장을 테스트해보려고 해봤는데, 가능하셨다면 문장 테스트를 조금 더 수행해보셨으면 어땠을까 하는 의견 드립니다.
