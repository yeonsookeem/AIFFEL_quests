# AIFFEL Campus Online 5th Code Peer Review 
- 코더 : 김연수
- 리뷰어 : 조대호


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 주어진 문제를 해결했고, 추가적인 문제인 스티커 자연스럽게 적용하기 부분이 안되어 있어이 부분을 예시에 추가했습니다. 다양한 각도에서 촬영 후 문제점 해결은 저도 해결하지 못해서 이 부분에 대한 해결방법은 제시하지 못했습니다.
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 필요한 부분에 주석을 적어주셔서 이해하는데 도움이 되었습니다.
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 이미지 및 landmark 경로 설정하는것 외에는 전부 정상작동 하였습니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 주석을 잘 달아놓은것을 보아 코드를 잘 이해하시고 작성한것 같습니다.
- [O] 코드가 간결한가요?
  > 필요한 코드만 작성해서 간결한거 같습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 스티커 자연스럽게 보이기
#cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) , 감마는 bias 값
#alpha , beta를 이용해 투명도 조절 가능
#addweighted를 활용해  스티커 사진의 투명도를 조절할 수 있다.
dst = cv2.addWeighted(sticker_area, 0.7, img_sticker, 0.3, 0)
#sticker_area와 img_sticker 뒤에 숫자를 이용해 투명도를 조절할 수 있다.

img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==255,sticker_area,dst).astype(np.uint8)
#원본 이미지인 img_bgr에 img_sticker 부분에 투명도를 조절한 스티커를 입력해주면 된다.

#참고 사이트: https://dsbook.tistory.com/155
```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

