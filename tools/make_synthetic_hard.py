"""더 어렵고 더 많은 합성 테스트 이미지 만들기 (약 990장)

make_synthetic_data.py 보다 실제 촬영에 가깝게 만든다.
- 물체 33개 (색마다 밝기·진하기가 다른 3개)
- 조명 색과 밝기가 장마다 조금씩 다름
- 한쪽이 어두워지는 밝기 변화, 그림자, 반사(하이라이트), 노이즈
- 장 번호에 따라 배경이 다름
    1, 2: 흰색 배경지   3: 회색 배경지   4: 나무 책상(갈색)   5: 배경 없이 물체만 가득

그래도 가짜 이미지다. 여기서 나온 숫자는 보고서에 쓰면 안 된다.

사용법 (프로젝트 폴더에서):
    python tools/make_synthetic_hard.py
"""
import os

import cv2
import numpy as np

OUT_DIR = "data/synthetic_hard"
SIZE = 300

# 색마다 3개씩 (R, G, B)
COLORS = {
    "black":  [(25, 25, 25), (40, 38, 42), (15, 15, 20)],
    "white":  [(240, 240, 240), (250, 248, 240), (230, 232, 236)],
    "gray":   [(128, 128, 128), (90, 90, 92), (165, 165, 160)],
    "red":    [(200, 30, 30), (170, 20, 40), (225, 50, 40)],
    "orange": [(240, 130, 20), (250, 150, 50), (230, 110, 30)],
    "yellow": [(240, 220, 30), (250, 230, 90), (225, 200, 20)],
    "green":  [(40, 160, 60), (30, 110, 50), (110, 190, 70)],
    "blue":   [(30, 70, 200), (20, 40, 120), (70, 140, 220)],
    "purple": [(120, 50, 190), (90, 40, 130), (150, 100, 200)],
    "pink":   [(250, 170, 190), (240, 130, 170), (245, 190, 205)],
    "brown":  [(120, 70, 30), (95, 60, 40), (140, 90, 50)],
}

# 조명 색 (R, G, B 배율)
LIGHTS = {
    "sun":   (1.00, 1.00, 1.00),
    "shade": (0.70, 0.78, 0.90),
    "fluor": (0.90, 1.00, 0.88),
    "warm":  (1.00, 0.80, 0.55),
    "cool":  (0.88, 0.95, 1.00),
    "flash": (1.00, 0.98, 0.95),
}

# 장 번호 -> 배경 색 (None 이면 물체가 화면을 가득 채움)
BACKGROUNDS = {
    1: (240, 240, 240),
    2: (240, 240, 240),
    3: (190, 190, 190),
    4: (170, 130, 90),
    5: None,
}


def make_image(sample_rgb, light_rgb, background, rng):
    # 1) 배경과 물체
    if background is None:
        img = np.full((SIZE, SIZE, 3), sample_rgb, dtype=np.float32)
        x0, y0, side = 0, 0, SIZE
    else:
        img = np.full((SIZE, SIZE, 3), background, dtype=np.float32)
        side = int(SIZE * rng.uniform(0.45, 0.60))
        x0 = (SIZE - side) // 2 + int(rng.integers(-15, 16))
        y0 = (SIZE - side) // 2 + int(rng.integers(-15, 16))
        img[y0:y0 + side, x0:x0 + side] = sample_rgb

    # 2) 조명: 색을 조금 흔들고 밝기도 바꾼다
    light = np.array(light_rgb) * rng.uniform(0.95, 1.05, 3) * rng.uniform(0.75, 1.05)
    img *= light.astype(np.float32)

    # 3) 한쪽이 어두워지는 밝기 변화
    ramp = np.linspace(rng.uniform(0.80, 0.95), 1.0, SIZE, dtype=np.float32)
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    img *= ramp[None, :, None] if rng.random() < 0.5 else ramp[:, None, None]

    # 4) 그림자 (30%): 세로 띠 하나를 어둡게
    if rng.random() < 0.3:
        sx = int(rng.integers(0, SIZE - 100))
        shadow = np.ones((SIZE, SIZE), dtype=np.float32)
        shadow[:, sx:sx + 100] = 0.6
        shadow = cv2.GaussianBlur(shadow, (41, 41), 0)
        img *= shadow[:, :, None]

    # 5) 반사 하이라이트 (30%): 물체 위에 밝은 점
    if rng.random() < 0.3:
        cx = int(rng.integers(x0 + 20, x0 + side - 20))
        cy = int(rng.integers(y0 + 20, y0 + side - 20))
        spot = np.zeros((SIZE, SIZE), dtype=np.float32)
        cv2.circle(spot, (cx, cy), 18, 1.0, -1)
        spot = cv2.GaussianBlur(spot, (31, 31), 0)
        img += spot[:, :, None] * 150

    # 6) 카메라 노이즈
    img += rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    count = 0
    obj = 0
    for color, rgb_list in COLORS.items():
        for rgb in rgb_list:
            obj += 1
            for light, light_rgb in LIGHTS.items():
                for shot, background in BACKGROUNDS.items():
                    path = os.path.join(OUT_DIR, f"{obj:02d}_{color}_{light}_{shot}.jpg")
                    img = make_image(rgb, light_rgb, background, rng)
                    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    count += 1
    print(f"{count}장 생성: {OUT_DIR}")


if __name__ == "__main__":
    main()
