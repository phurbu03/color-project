"""사진 한 장의 색 이름을 방법별로 출력한다.

사용법 (프로젝트 폴더에서):
    python src/name_photo.py 사진.jpg

물체가 화면 가운데에 오게 찍은 사진을 넣는다.
"""
import sys

from color_naming import format_result, name_image
from evaluate import read_image
from illumination import METHODS, VARIETY_MIN, scene_variety


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    img = read_image(sys.argv[1])
    if img is None:
        print(f"사진을 읽지 못했습니다: {sys.argv[1]}")
        return

    variety = scene_variety(img)
    clue = "있음" if variety >= VARIETY_MIN else "없음"
    print(f"화면의 색 다양성: {variety:.3f} (기준 {VARIETY_MIN}) -> 조명 단서 {clue}\n")

    for method_name, correct_fn in METHODS.items():
        corrected, reliable = correct_fn(img)
        name, other, confident = name_image(corrected)
        print(f"{method_name:<12} {format_result(name, other, confident and reliable)}")


if __name__ == "__main__":
    main()
