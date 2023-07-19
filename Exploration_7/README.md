# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 김연수
- 리뷰어 : 손정민


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 데이터에 augmentation을 적용하였고, UNetGenerator도 구현하였다. 테스트 데이터를 이용한 20 epochs의 학습 후 생성된 사진을 시각화하는 것까지 수행하였다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 주석으로 각 과정을 상세히 설명하여 코드를 읽기 쉬웠다. 이미지를 첨부하여 구조적인 부분에 대한 이해를 도왔다.
  ```python
  # "C64", "C128" 등으로 쓰여진 것과 같이 
  # "Convolution → BatchNorm → LeakyReLU"의 3개 레이어로 구성된 기본적인 블록을 아래와 같이 하나의 레이어로 만들었습니다.

  # EncodeBlock
  rom tensorflow.keras import layers, Input, Model

  class EncodeBlock(layers.Layer):...
  ```
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 에러 없이 잘 실행되었다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 주석의 내용으로 보아 코드의 동작을 잘 이해하고 작성한 것 같다.
  ```python
  # Discriminator에 사용할 기본적인 블록

  #__init__() 에서 필요한 만큼 많은 설정을 가능하게끔 했습니다. 
  # 필터의 수(n_filters), 필터가 순회하는 간격(stride), 출력 feature map의 크기를 조절할 수 있도록 하는 패딩 설정(custom_pad),
  # BatchNorm의 사용 여부(use_bn), 활성화 함수 사용 여부(act)가 설정 가능


  class DiscBlock(layers.Layer):...
  ```
- [X] 코드가 간결한가요?
  > 직관적인 변수명을 사용했고, 적절한 공백으로 가독성을 높였다.  



# 참고 링크 및 코드 개선

주어진 과제를 잘 수행했기 때문에 코드의 수정사항은 보이지 않지만, 학습 후 loss를 시각화하면 좋을 것 같다.
```python
EPOCHS = 20

generator = UNetGenerator()
discriminator = Discriminator()
history = {'gen_loss':[], 'l1_loss':[], 'disc_loss':[]}

for epoch in range(1, EPOCHS+1):
    for i, (sketch, colored) in enumerate(train_images):
        g_loss, l1_loss, d_loss = train_step(sketch, colored)
        history['gen_loss'].append(g_loss)
        history['l1_loss'].append(l1_loss)
        history['disc_loss'].append(d_loss)  
                
        # 10회 반복마다 손실을 출력합니다.
        if (i+1) % 10 == 0:
            print(f"EPOCH[{epoch}] - STEP[{i+1}] \
                    \nGenerator_loss:{g_loss.numpy():.4f} \
                    \nL1_loss:{l1_loss.numpy():.4f} \
                    \nDiscriminator_loss:{d_loss.numpy():.4f}", end="\n\n")
```
```python
plt.figure(figsize=(16,10))

plt.subplot(311)
plt.plot(history['gen_loss'])
plt.title('Generator Loss')

plt.subplot(312)
plt.plot(history['l1_loss'])
plt.title('L1 Loss')

plt.subplot(313)
plt.plot(history['disc_loss'])
plt.title('Discriminator Loss')

plt.show()
```
