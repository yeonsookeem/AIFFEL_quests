# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김연수
- 리뷰어 : 김민식


# PRT(Peer Review Template)
- [O]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
      퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
  > 중간에 `RM`, `PPO` 모델 훈련에서 CUDA 이슈가 있긴 하였지만, 코드 구성 자체는 모두 정상이었으며, `SFT`에 대해 훈련이 진행되고 평가한 내용을 확인하였습니다.
  ```python
  training_args = TrainingArguments(
    output_dir="aiffel/KoChatGPT/test",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=5,
    prediction_loss_only=True,
    fp16 = True
    )
  trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=train_dataset
  )

  trainer.train()
  model.save_pretrained('/aiffel/aiffel/KoChatGPT/output_1_SFT')
  ```
    
- [O]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
  > Markdown으로 구간을 잘 표시해주셨고 (`Data Augmentation- SFT, RM, PPO 각 데이터셋 증강`)
  
  > 각 함수를 통해 증강을 이해할 수 있었습니다.
  ```python
  # Synonym Replacement: 문장 내의 단어를 그 단어의 동의어로 교체
  def synonym_replacement(words, n=5):
      new_words = words.copy()
      random_word_list = list(set([word for word in words if word.isalnum()]))
      random.shuffle(random_word_list)
      num_replaced = 0
      for random_word in random_word_list:
          synonyms = get_synonyms(random_word)
          if len(synonyms) >= 1:
              synonym = random.choice(list(synonyms))
              new_words = [synonym if word == random_word else word for word in new_words]
              num_replaced += 1
          if num_replaced >= n: 
              break
      return new_words
  ```
  ```python
  # Random Deletion: 문장 내의 단어를 무작위로 삭제
  def random_deletion(words, p=0.5): 
      if len(words) == 1: 
          return words
      remaining = list(filter(lambda x: random.uniform(0,1) > p,words)) 
      if len(remaining) == 0: 
          return [random.choice(words)]
      else:
          return remaining
  ```
  ```python
  # Random Swap: 문장 내의 두 단어의 위치를 무작위로 바꿈.
  def random_swap(sentence, n=5): 
      length = range(len(sentence))
      if len(sentence) < 2:  # Check if the sentence has less than 2 words
          return sentence
      for _ in range(n):
          idx1, idx2 = random.sample(length, 2)
          sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]
      return sentence
  ```
  
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
      실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    > 디버깅 부분이 따로 있지 않아 `X`표시를 하였지만, 회고 부분을 통해 OOM 에러 등에 대한 인지를 하고 계심을 확인하였습니다.
- [O]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    > 전체적인 수행과정에서 느낀점을 잘 작성해주셨습니다.
- [O]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
  > 필요한 부분을 함수화하셔서 간결함을 확인할 수 있었습니다.
  ```python
  # Random Swap: 문장 내의 두 단어의 위치를 무작위로 바꿈.
  def random_swap(sentence, n=5): 
      length = range(len(sentence))
      if len(sentence) < 2:  # Check if the sentence has less than 2 words
          return sentence
      for _ in range(n):
          idx1, idx2 = random.sample(length, 2)
          sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]
      return sentence
  ```

# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
OOM 이슈가 많이 보여 다음 링크를 통해 케글 노트북에서 테스트해보시는 것도 한 방법일 것 같습니다.<br>
[How to use Kaggle notebook (feat. 케글필사))](https://bedevelopers.tistory.com/258)