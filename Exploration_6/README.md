# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 김연수
- 리뷰어 : [이태훈](https://github.com/git-ThLee)


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네

- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네! 
  ```python
  def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    e_out, e_h, e_c = encoder_model.predict(input_seq)

     # <SOS>에 해당하는 토큰 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tar_word_to_index['sostoken']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition: # stop_condition이 True가 될 때까지 루프 반복

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tar_index_to_word[sampled_token_index]

        if (sampled_token!='eostoken'):
            decoded_sentence += ' '+sampled_token

        #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_token == 'eostoken'  or len(decoded_sentence.split()) >= (headlines_max_len-1)):
            stop_condition = True

        # 길이가 1인 타겟 시퀀스를 업데이트
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # 상태를 업데이트 합니다.
        e_h, e_c = h, c

    return decoded_sentence
  ```

- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네! 없습니다!
  ```python
  # 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
  def seq2text(input_seq):
      temp=''
      for i in input_seq:
          if (i!=0):
              temp = temp + src_index_to_word[i]+' '
      return temp
  
  # 요약문의 정수 시퀀스를 텍스트 시퀀스로 변환
  def seq2headlines(input_seq):
      temp = ''
      for i in input_seq:
          if (i != 0 and i != tar_word_to_index['sostoken'] and i != tar_word_to_index['eostoken']):
              temp = temp + tar_index_to_word[i] + ' '
      return temp
  ```
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네! 아래 코드를 돌리려면 여러 함수를 거처서 확인해야하는데, 공부 열심히 하신거 같습니다!
  ```python
  for i in range(50, 100):
    print("원문 :", seq2text(encoder_input_test[i]))
    print("실제 요약 :", seq2headlines(decoder_input_test[i]))
    print("예측 요약 :", decode_sequence(encoder_input_test[i].reshape(1, text_max_len)))
    print("\n")
  ```
- [X] 코드가 간결한가요?
  > 네! 그래프를 간결하게 그리기가 어려운데, 간결하게 코드를 작성하셧습니다!
  ```python
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show()
  ```

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
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
