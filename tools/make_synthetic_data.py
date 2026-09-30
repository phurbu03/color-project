"""가짜(합성) 테스트 이미지 만들기

실제 촬영 전에 평가 코드가 끝까지 도는지 확인하는 용도다.
"찍힌 색 ≈ 물체 색 × 조명 색" 공식 그대로 만든 이미지라서 실제 사진보다 훨씬 쉽다.
여기서 나온 숫자는 보고서에 쓰면 안 된다.

사용법 (프로젝트 폴더에서):
    python tools/make_synthetic_data.py
"""
import os

import cv2
import numpy as np

OUT_DIR = "data/synthetic"

# 물체 색 (R, G, B) - 흰 조명 아래에서의 색
SAMPLES = [
    ("01", "black",  (25, 25, 25)),
    ("02", "white",  (240, 240, 240)),
    ("03", "gray",   (128, 128, 128)),
    ("04", "red",    (200, 30, 30)),
    ("05", "orange", (240, 130, 20)),
    ("06", "yellow", (240, 220, 30)),
    ("07", "green",  (40, 160, 60)),
    ("08", "blue",   (30, 70, 200)),
    ("09", "purple", (120, 50, 190)),
    ("10", "pink",   (250, 170, 190)),
    ("11", "brown",  (120, 70, 30)),
]

# 조명 색 (R, G, B 배율)
LIGHTS = {
    "sun":   (1.00, 1.00, 1.00),
    "shade": (0.70, 0.78, 0.90),   # 어둡고 푸름
    "fluor": (0.90, 1.00, 0.88),   # 약간 초록
    "warm":  (1.00, 0.80, 0.55),   # 전구색: 노랑
    "cool":  (0.88, 0.95, 1.00),   # 주광색: 약간 푸름
    "flash": (1.00, 0.98, 0.95),
}

BACKGROUND = (225, 225, 225)       # 밝은 회색 배경지
SIZE = 300


def make_image(sample_rgb, light_rgb, rng):
    img = np.full((SIZE, SIZE, 3), BACKGROUND, dtype=np.float32)
    img[75:225, 75:225] = sample_rgb                       # 가운데에 색 샘플
    img *= np.array(light_rgb, dtype=np.float32)           # 조명 곱하기
    img += rng.normal(0, 4, img.shape)                     # 카메라 노이즈
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    count = 0
    for obj, color, rgb in SAMPLES:
        for light, light_rgb in LIGHTS.items():
            for i in range(1, 3):
                path = os.path.join(OUT_DIR, f"{obj}_{color}_{light}_{i}.jpg")
                cv2.imwrite(path, make_image(rgb, light_rgb, rng))
                count += 1
    print(f"{count}장 생성: {OUT_DIR}")


if __name__ == "__main__":
    main()
