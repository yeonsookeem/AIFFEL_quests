# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김연수
- 리뷰어 : 김성진


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
      퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부

    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

- 각 섹션별로 과정이 나뉘어 있고, annotation 이 작성되어 있습니다.
```python
training_arguments = TrainingArguments(
    output_dir,                                         # output이 저장될 경로
    evaluation_strategy="epoch",           #evaluation하는 빈도
    learning_rate = 2e-5,                         #learning_rate
    per_device_train_batch_size = 512,   # 각 device 당 batch size
    per_device_eval_batch_size = 4,    # evaluation 시에 batch size
    num_train_epochs = 1,                     # train 시킬 총 epochs
    weight_decay = 0.01,                        # weight decay
)
```
  
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
      실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.


- 'RuntimeError: CUDA out of memory' 문제를 해결하기 위해 gc.collect() 등 문제 해결에 대한 내용이 작성되어 있습니다.
```python
import gc
gc.collect()
```

  
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

- 문제 해결에 대한 내용과 함께 회고가 잘 작성되었습니다.
```markdown
회고
- 'RuntimeError: CUDA out of memory' 문제를 해결하기 위해  
: max_length 조절, batch_size 조절, epochs 낮추기, gc.collect() 사용, 데이터 사이즈 줄여보기 등의 여러가지 방법을 시도해보았지만, 결국 훈련을 돌려보지 못했다.
- 그렇지만 오류 덕분에(?) colab과 jupyter notebook을 다 돌려보는 기회가 되었다. 
- 훈련 실행은 실패했지만, huggingface의 모델을 불러와서 task에 적용하는 전체 흐름을 배울 수 있었다.
```
    
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.


- tokenize 를 위한 함수, metrics 함수 등이 작성되어 있습니다.
```python
def transform(data, max_length=64):  # Set your desired max_length here
    return tokenizer(
        data['document'],
        truncation=True,
        padding='max_length',
        max_length=max_length,  # Set the desired maximum sequence length
        return_token_type_ids=False,
    )
```

# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

```python
    per_device_train_batch_size = 512,   # 각 device 당 batch size
    per_device_eval_batch_size = 4,    # evaluation 시에 batch size
```
- RuntimeError: CUDA out of memory. Tried to allocate 96.00 MiB (GPU 0; 14.76 GiB total capacity; 13.38 GiB already allocated; 3.75 MiB free; 13.50 GiB reserved in total by PyTorch)
  - 이 에러... 제가 per_device_train_batch_size 를 8로 했을 때 발생했었는데 4로 줄이고 나서야 진행이 되었습니다.
  - 그런데 subset_nsmc_dataset 로 진행된 코드에서는 batch size 가 맞지않는 문제로 줄일 수가 없네요. 흠...