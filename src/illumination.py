"""조명 보정 (3주차: 채널 평균·최댓값, 히스토그램)

찍힌 색 ≈ 물체 색 × 조명 색 이므로, 조명 색을 추정해서 채널별로 나눠주면 된다.
조명 색은 물체만이 아니라 "화면 전체"에서 추정한다. 주변에 찍힌 것들이 단서가 된다.

모든 보정 함수는 (보정된 이미지, 보정을 믿을 수 있는지) 를 돌려준다.
"""
import cv2
import numpy as np

# 화면의 색 다양성이 이보다 작으면 "단서 없음"으로 본다 (조정 대상)
VARIETY_MIN = 0.02


def _apply_gain(img_bgr, gain):
    """채널별로 gain(B, G, R 배율)을 곱하고 0~255로 자른다."""
    out = img_bgr.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def no_correction(img_bgr):
    """보정 없음 (비교 기준)"""
    return img_bgr, True


def gray_world(img_bgr):
    """Gray World: 장면 전체 평균은 회색이라고 가정한다.

    채널별 평균이 서로 같아지도록 배율을 곱한다.
    한 가지 색이 화면을 가득 채우면 가정이 깨져서 실패한다.
    """
    mean_bgr = img_bgr.reshape(-1, 3).mean(axis=0)      # 채널별 평균 = 조명 색 추정
    gain = mean_bgr.mean() / (mean_bgr + 1e-6)
    return _apply_gain(img_bgr, gain), True


def white_patch(img_bgr, top_percent=1.0):
    """White Patch: 장면에서 가장 밝은 부분은 흰색이라고 가정한다.

    채널별 '가장 밝은 값'이 255가 되도록 배율을 곱한다.
    진짜 최댓값은 노이즈 한 픽셀에 흔들리므로, 밝은 쪽 상위 top_percent% 위치의 값을 쓴다.
    장면에 흰색(또는 밝은 무채색)이 없으면 실패한다.
    """
    bright_bgr = np.percentile(img_bgr.reshape(-1, 3), 100 - top_percent, axis=0)
    gain = 255.0 / (bright_bgr + 1e-6)
    return _apply_gain(img_bgr, gain), True


def scene_variety(img_bgr, grid=12):
    """화면에 색이 얼마나 다양하게 들어 있는지 숫자로 돌려준다.

    1) 화면을 grid x grid 칸으로 줄인다 (칸마다 평균 색이 된다)
    2) 칸마다 B, G, R을 합으로 나눠 '색 비율'로 바꾼다 (밝기 차이는 빼고 색만 본다)
    3) 칸들 사이에서 색 비율이 얼마나 퍼져 있는지(표준편차) 본다

    한 가지 색이 화면을 가득 채우면 0에 가깝다. 그러면 조명을 짐작할 단서가 없다.
    """
    small = cv2.resize(img_bgr, (grid, grid), interpolation=cv2.INTER_AREA)
    small = small.reshape(-1, 3).astype(np.float32)
    ratio = small / (small.sum(axis=1, keepdims=True) + 1e-6)
    return float(ratio.std(axis=0).max())


def auto_correct(img_bgr):
    """단서가 있을 때만 보정한다.

    화면의 색 다양성이 충분하면 White Patch로 보정하고,
    부족하면 보정하지 않고 '믿을 수 없음'(False)을 같이 돌려준다.
    """
    if scene_variety(img_bgr) < VARIETY_MIN:
        return img_bgr, False
    return white_patch(img_bgr)


# 평가 코드에서 방법 이름으로 꺼내 쓰기 위한 표
METHODS = {
    "none": no_correction,
    "gray_world": gray_world,
    "white_patch": white_patch,
    "auto": auto_correct,
}
