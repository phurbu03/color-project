"""조명 보정 (3주차: 채널 평균·최댓값, 히스토그램)

찍힌 색 ≈ 물체 색 × 조명 색 이므로, 조명 색을 추정해서 채널별로 나눠주면 된다.
두 방법은 "조명 색을 무엇으로 추정하느냐"만 다르다.
"""
import numpy as np


def _apply_gain(img_bgr, gain):
    """채널별로 gain(B, G, R 배율)을 곱하고 0~255로 자른다."""
    out = img_bgr.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def no_correction(img_bgr):
    """보정 없음 (비교 기준)"""
    return img_bgr


def gray_world(img_bgr):
    """Gray World: 장면 전체 평균은 회색이라고 가정한다.

    채널별 평균이 서로 같아지도록 배율을 곱한다.
    한 가지 색이 화면을 가득 채우면 가정이 깨져서 실패한다.
    """
    mean_bgr = img_bgr.reshape(-1, 3).mean(axis=0)      # 채널별 평균 = 조명 색 추정
    gain = mean_bgr.mean() / (mean_bgr + 1e-6)
    return _apply_gain(img_bgr, gain)


def white_patch(img_bgr, top_percent=1.0):
    """White Patch: 장면에서 가장 밝은 부분은 흰색이라고 가정한다.

    채널별 '가장 밝은 값'이 255가 되도록 배율을 곱한다.
    진짜 최댓값은 노이즈 한 픽셀에 흔들리므로, 밝은 쪽 상위 top_percent% 위치의 값을 쓴다.
    장면에 흰색(또는 밝은 무채색)이 없으면 실패한다.
    """
    bright_bgr = np.percentile(img_bgr.reshape(-1, 3), 100 - top_percent, axis=0)
    gain = 255.0 / (bright_bgr + 1e-6)
    return _apply_gain(img_bgr, gain)


# 평가 코드에서 방법 이름으로 꺼내 쓰기 위한 표
METHODS = {
    "none": no_correction,
    "gray_world": gray_world,
    "white_patch": white_patch,
}
