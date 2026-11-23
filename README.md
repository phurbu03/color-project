# 조명에 강한 색약 보조 색 인식 카메라

컴퓨터 비전 공부용 프로젝트

## 폴더 구조
```
src/color_naming.py      대표 색 추출 + HSV 규칙 색 이름 (11색)
src/illumination.py      조명 보정: Gray World, White Patch
src/evaluate.py          조명별 정확도, 조명 간 일관성, confusion matrix
tools/make_synthetic_data.py   코드 확인용 합성 이미지 생성
data/raw/                직접 촬영한 사진 (파일명: 물체번호_정답색_조명_번호.jpg)
data/synthetic/          합성 이미지 (보고서에 쓰지 않음)
results/                 평가 결과 CSV·그래프
docs/                    기획서, 촬영 가이드
```

## 실행
필요한 것: `opencv-python`, `numpy`, `matplotlib` (Colab에는 이미 설치되어 있음)

```
python tools/make_synthetic_data.py     # 합성 이미지로 코드가 도는지 확인
python src/evaluate.py data/synthetic
python src/evaluate.py data/raw         # 실제 촬영 데이터 평가
```

Colab에서는 폴더를 올린 뒤 `%cd color-project` 하고 위 명령 앞에 `!`를 붙인다.
