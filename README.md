# 조명에 강한 색약 보조 색 인식 카메라

컴퓨터 비전 공부용 프로젝트

주변에 조명 단서가 있으면 자동으로 보정하고, 단서가 없거나 색이 경계 근처면 "불확실"이라고 말한다.

## 폴더 구조
```
src/color_naming.py      대표 색 추출 + HSV 규칙 색 이름 (11색) + 불확실 판단
src/illumination.py      조명 보정: Gray World, White Patch, auto(단서 있을 때만 보정)
src/evaluate.py          조명별 정확도, 조명 간 일관성, confusion matrix, 불확실 표시의 정직성
src/name_photo.py        사진 한 장의 색 이름을 방법별로 출력
tools/make_synthetic_data.py   코드 확인용 합성 이미지 (132장, 쉬움)
tools/make_synthetic_hard.py   코드 확인용 합성 이미지 (990장, 어려움)
data/raw/natural/        직접 촬영: 배경지 없이 실제 장소 (주 조건)
data/raw/paper/          직접 촬영: 배경지 위 (비교 조건)
data/raw/fill/           직접 촬영: 물체가 화면 가득 (실패 사례)
data/synthetic*/         합성 이미지 (보고서에 쓰지 않음)
results/                 평가 결과 CSV·그래프
docs/                    기획서, 촬영 가이드
```
파일명: `물체번호_정답색_조명_번호.jpg` (예: `03_red_warm_2.jpg`). 코드표는 `docs/shooting-guide.md`.

## 실행
필요한 것: `opencv-python`, `numpy`, `matplotlib` (Colab에는 이미 설치되어 있음)

```
python tools/make_synthetic_hard.py     # 합성 이미지로 코드가 도는지 확인
python src/evaluate.py data/synthetic_hard
python src/evaluate.py data/raw/natural # 실제 촬영 데이터 평가
python src/name_photo.py 사진.jpg        # 사진 한 장만 확인
```

Colab에서는 폴더를 올린 뒤 `%cd color-project` 하고 위 명령 앞에 `!`를 붙인다.
