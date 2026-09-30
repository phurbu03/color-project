"""평가: 조명별 정확도, 조명 간 일관성, confusion matrix, 불확실 표시의 정직성

사용법 (프로젝트 폴더에서):
    python src/evaluate.py data/raw/natural
    python src/evaluate.py data/synthetic

파일명 규칙: 물체번호_정답색_조명_번호.jpg   예) 03_red_warm_2.jpg
"""
import csv
import os
import sys

import cv2
import numpy as np

from color_naming import COLOR_NAMES, name_image
from illumination import METHODS

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def read_image(path):
    """cv2.imread는 Windows에서 한글 경로를 못 읽으므로 바이트로 읽어서 디코딩한다."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_samples(folder):
    """폴더의 이미지 목록을 [(경로, 물체번호, 정답색, 조명), ...] 으로 돌려준다."""
    samples = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(IMAGE_EXT):
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) != 4 or parts[1] not in COLOR_NAMES:
            print(f"[건너뜀] 파일명 규칙에 안 맞음: {fname}")
            continue
        obj, color, light, _ = parts
        samples.append((os.path.join(folder, fname), obj, color, light))
    return samples


def ratio(rows):
    """rows 중 정답을 맞힌 비율. rows가 비어 있으면 None."""
    if not rows:
        return None
    return sum(r[1] == r[3] for r in rows) / len(rows)


def evaluate(samples, method_name):
    """보정 방법 하나에 대해 예측을 모으고 지표를 계산한다."""
    correct_fn = METHODS[method_name]
    rows = []                                   # (물체, 정답, 조명, 예측, 확신 여부)
    for path, obj, color, light in samples:
        img = read_image(path)
        if img is None:
            print(f"[건너뜀] 읽기 실패: {path}")
            continue
        corrected, reliable = correct_fn(img)
        pred, _, confident = name_image(corrected)
        # 색이 경계 근처이거나, 조명 보정을 믿을 수 없으면 "불확실"
        rows.append((obj, color, light, pred, confident and reliable))

    # 1) 조명별 정확도
    lights = sorted({r[2] for r in rows})
    acc_by_light = {light: ratio([r for r in rows if r[2] == light]) for light in lights}

    # 2) 조명 간 일관성: 한 물체의 모든 사진이 같은 이름으로 나온 물체의 비율
    objects = sorted({r[0] for r in rows})
    consistent = 0
    for obj in objects:
        preds = {r[3] for r in rows if r[0] == obj}
        if len(preds) == 1:
            consistent += 1

    # 3) confusion matrix: 행 = 정답, 열 = 예측
    n = len(COLOR_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for _, color, _, pred, _ in rows:
        cm[COLOR_NAMES.index(color), COLOR_NAMES.index(pred)] += 1

    # 4) 불확실 표시의 정직성: "확신"이라고 한 것은 잘 맞고, "불확실"이라고 한 것은 덜 맞아야 의미가 있다
    sure = [r for r in rows if r[4]]
    unsure = [r for r in rows if not r[4]]

    return {"acc_by_light": acc_by_light, "acc_total": ratio(rows),
            "consistency": consistent / len(objects), "cm": cm, "rows": rows,
            "unsure_ratio": len(unsure) / len(rows),
            "acc_sure": ratio(sure), "acc_unsure": ratio(unsure)}


def fmt(value, width=10):
    """숫자를 표 한 칸으로. 값이 없으면 '-'"""
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.2f}"


def save_results(results, out_dir):
    """요약 표와 confusion matrix를 CSV로 저장한다."""
    os.makedirs(out_dir, exist_ok=True)
    lights = sorted(next(iter(results.values()))["acc_by_light"].keys())
    extra = ["acc_total", "consistency", "unsure_ratio", "acc_sure", "acc_unsure"]

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method"] + lights + extra)
        for name, r in results.items():
            values = [r["acc_by_light"][l] for l in lights] + [r[k] for k in extra]
            w.writerow([name] + ["" if v is None else f"{v:.3f}" for v in values])

    for name, r in results.items():
        with open(os.path.join(out_dir, f"confusion_{name}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + COLOR_NAMES)
            for i, cname in enumerate(COLOR_NAMES):
                w.writerow([cname] + list(r["cm"][i]))


def plot_accuracy(results, out_path):
    """x축 조명, y축 정확도, 선 = 보정 방법 (발표용 핵심 그래프)"""
    import matplotlib
    matplotlib.use("Agg")                       # 창 없이 파일로만 저장
    import matplotlib.pyplot as plt

    lights = sorted(next(iter(results.values()))["acc_by_light"].keys())
    plt.figure(figsize=(8, 4.5))
    for name, r in results.items():
        plt.plot(lights, [r["acc_by_light"][l] for l in lights], marker="o", label=name)
    plt.ylim(0, 1.05)
    plt.xlabel("lighting")
    plt.ylabel("accuracy")
    plt.title("Accuracy by lighting")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw/natural"
    samples = load_samples(folder)
    if not samples:
        print(f"{folder} 에 평가할 이미지가 없습니다.")
        return
    print(f"이미지 {len(samples)}장, 물체 {len({s[1] for s in samples})}개\n")

    results = {name: evaluate(samples, name) for name in METHODS}

    lights = sorted(results["none"]["acc_by_light"].keys())
    print("[조명별 정확도, 조명 간 일관성]")
    print(f"{'method':<12}" + "".join(f"{l:>10}" for l in lights) + f"{'total':>10}{'consist.':>10}")
    for name, r in results.items():
        print(f"{name:<12}" + "".join(fmt(r["acc_by_light"][l]) for l in lights)
              + fmt(r["acc_total"]) + fmt(r["consistency"]))

    print("\n[불확실 표시의 정직성]")
    print(f"{'method':<12}{'불확실 비율':>10}{'확신 정확도':>10}{'불확실 정확도':>10}")
    for name, r in results.items():
        print(f"{name:<12}" + fmt(r["unsure_ratio"], 14) + fmt(r["acc_sure"], 14) + fmt(r["acc_unsure"], 14))

    out_dir = os.path.join("results", os.path.basename(os.path.normpath(folder)))
    save_results(results, out_dir)
    plot_accuracy(results, os.path.join(out_dir, "accuracy_by_light.png"))
    print(f"\n결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
