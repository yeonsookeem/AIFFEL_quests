# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 김연수
- 리뷰어 : 조대호

# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네. 교차검증,data 변수 범주형 변수로 만드는 과정 등 주석을 달아주셔서 이해하는데 도움이 되었습니다.
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > clone 받아서 vscode로 실행해본 결과 에러가 발생하지 않았습니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 주석문을 달아주신것으로 보아 이해하고 작성했다.
- [O] 코드가 간결한가요?
  > baseline을 참고해서 작성하셔서 코드가 간결했다.

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
#전처리 과정에서 features 값이나 y(target)값들을 분석하여 치우침현상이 발생하는 부분들이 있는데 정규분포형식으로 바꿔주는것이학습을 하는데 유리합니다. 이것에 관한 코드를 작성합니다 (exploation3 참고)

#x(features)분포 확인
# id 변수(count==0인 경우)는 제외하고 분포를 확인합니다.
# row*col이 feature의 개수를 의미하고 0번째 인덱스인 id값은 제외
count = 1
columns = train.columns
for row in range(6):
    for col in range(4):
        sns.kdeplot(data=train[columns[count]], ax=ax[row][col])
        ax[row][col].set_title(columns[count], fontsize=15)
        count += 1
        if count == 22 :
            break

#확인한 결과 치우친 분포를 따르는 열들을 모아서 log를 취해줍니다.
log_columns = [ '' ] # 이 리스트에 로그변환을 취할 features를 넣으면 됩니다.
for c in log_columns:
    train[c] = np.log1p(train[c].values)

#y도 마찬가지로 분포를 확인하고 치우쳐 있다면 로그함수로 정규분포로 변환해줍니다

#y의 분포 확인
sns.kdeplot(y)
plt.show()

#y를 로그변환 시켜 정규분포 형태로 만든다.
y = np.log1p(y)
sns.kdeplot(y)
plt.show()
```
