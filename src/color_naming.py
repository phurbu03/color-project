"""색 이름 분류 (2주차: 색공간 변환, 4주차: 블러)

BGR 이미지에서 대표 색을 뽑고, HSV 규칙으로 기본 11색 중 하나의 이름을 붙인다.
색이 경계 근처면 "불확실"로 표시하고 두 번째 후보도 같이 돌려준다.
아래 경계값(threshold)은 전부 "처음 시작값"이다. 실제 촬영 데이터로 조정해야 한다.
"""
import cv2
import numpy as np

# 파일명에 쓰는 영어 코드 -> 화면에 보여줄 한국어 이름
COLOR_KR = {
    "black": "검정", "white": "흰색", "gray": "회색",
    "red": "빨강", "orange": "주황", "yellow": "노랑",
    "green": "초록", "blue": "파랑", "purple": "보라",
    "pink": "분홍", "brown": "갈색",
}
COLOR_NAMES = list(COLOR_KR.keys())

# --- 조정 대상 경계값 (OpenCV 기준: H 0~179, S 0~255, V 0~255) ---
V_BLACK = 50      # V가 이보다 작으면 검정
S_GRAY = 40       # S가 이보다 작으면 무채색(흰색/회색)
V_WHITE = 200     # 무채색 중 V가 이보다 크면 흰색
V_BROWN = 150     # 주황 계열인데 V가 이보다 작으면 갈색 (갈색 = 어두운 주황)
S_PINK = 120      # 빨강 계열인데 S가 이보다 작고
V_PINK = 180      # V가 이보다 크면 분홍 (분홍 = 옅고 밝은 빨강)

# --- 불확실 판단: H, S, V를 이만큼 흔들었을 때 이름이 바뀌면 경계 근처로 본다 ---
DELTA_H = 5
DELTA_S = 20
DELTA_V = 20


def representative_color(img_bgr, center_ratio=0.5, blur_ksize=5):
    """이미지 가운데 영역의 대표 색(B, G, R)을 돌려준다.

    - 블러로 노이즈를 줄이고
    - 가운데 center_ratio 만큼의 영역만 잘라서
    - 채널별 중앙값(median)을 쓴다. 평균보다 반사·그림자 픽셀에 덜 흔들린다.
    """
    blurred = cv2.GaussianBlur(img_bgr, (blur_ksize, blur_ksize), 0)
    h, w = blurred.shape[:2]
    dh = int(h * (1 - center_ratio) / 2)
    dw = int(w * (1 - center_ratio) / 2)
    roi = blurred[dh:h - dh, dw:w - dw]
    return np.median(roi.reshape(-1, 3), axis=0).astype(np.uint8)


def name_hsv(h, s, v):
    """HSV 값 하나를 받아 색 이름(영어 코드)을 돌려준다."""
    # 1) 무채색: 밝기(V)와 채도(S)로 먼저 걸러낸다
    if v < V_BLACK:
        return "black"
    if s < S_GRAY:
        return "white" if v > V_WHITE else "gray"

    # 2) 유채색: 색상(H) 구간으로 나눈다
    if h < 8 or h >= 170:                           # 빨강은 H가 0 근처와 179 근처 양쪽에 있다
        if s < S_PINK and v > V_PINK:
            return "pink"
        return "red"
    if h < 22:
        return "brown" if v < V_BROWN else "orange"
    if h < 35:
        return "brown" if v < V_BROWN - 40 else "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"                                   # 150 <= h < 170


def _to_hsv(bgr):
    pixel = np.uint8([[bgr]])                       # 1x1 이미지로 만들어서 변환
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return int(h), int(s), int(v)


def name_color(bgr):
    """BGR 색 하나 -> 색 이름"""
    return name_hsv(*_to_hsv(bgr))


def name_with_confidence(bgr):
    """BGR 색 하나 -> (색 이름, 두 번째 후보, 확신 여부)

    H, S, V를 하나씩 조금 흔들어 본다. 이름이 바뀌면 경계 근처라는 뜻이므로
    확신 여부를 False로 하고, 바뀐 이름을 두 번째 후보로 돌려준다.
    """
    h, s, v = _to_hsv(bgr)
    name = name_hsv(h, s, v)
    nearby = [
        ((h - DELTA_H) % 180, s, v), ((h + DELTA_H) % 180, s, v),   # H는 0과 179가 이어져 있다
        (h, max(s - DELTA_S, 0), v), (h, min(s + DELTA_S, 255), v),
        (h, s, max(v - DELTA_V, 0)), (h, s, min(v + DELTA_V, 255)),
    ]
    for hsv in nearby:
        other = name_hsv(*hsv)
        if other != name:
            return name, other, False
    return name, None, True


def name_image(img_bgr):
    """이미지 한 장 -> (색 이름, 두 번째 후보, 확신 여부)"""
    return name_with_confidence(representative_color(img_bgr))


def format_result(name, other, confident):
    """화면에 보여줄 문장. 예) '빨강' / '갈색 또는 주황 (불확실)'"""
    if confident:
        return COLOR_KR[name]
    if other is None:
        return f"{COLOR_KR[name]} (불확실)"
    return f"{COLOR_KR[name]} 또는 {COLOR_KR[other]} (불확실)"
