# -*- coding: utf-8 -*-
"""엔드밀 옆날 360도 회전 이미지에서 레퍼런스 각도 찾기.

방법
----
1. 그레이스케일 이미지를 임계값(thresh, 밝기)으로 이진화 → 어두운 픽셀 = 툴 후보.
   (왼쪽 끝에 어두운 배경 구조물이 함께 찍히므로 단순 최좌단 검색은 불가)
2. 화면 맨 오른쪽 가운데 픽셀을 시드로, 시드와 연결된 덩어리(connected component)만
   툴로 취급한다. 툴과 왼쪽 구조물 사이에는 흰 배경 띠가 있어 자연스럽게 분리된다.
3. 툴 덩어리를 이미지 중심 y(h/2) 기준 위/아래 반쪽으로 나누고, 각 반쪽에서
   x가 가장 작은(맨왼쪽) 픽셀을 먼저 확정한다. x가 y보다 우선순위가 높되,
   모서리가 둥근 경우를 위해 최좌단에서 x-tol(기본 20px) 이내 픽셀까지는
   y 극값 경쟁에 참여시킨다 (세로 직사각형+반원 캡이면 캡 꼭대기까지 포함).
   - 맨왼쪽 위 점: 위 반쪽 최좌단 x + tol 이내에서 가장 위(y 최소)
   - 맨왼쪽 아래 점: 아래 반쪽 최좌단 x + tol 이내에서 가장 아래(y 최대)
   두 점은 절삭날 코너라서 반드시 중심 반대쪽에 하나씩 있어야 한다.
4. 두 점의 x차이 dx는 참고용으로 기록/표시만 한다 (제약 없음).
   중심 반대쪽 조건을 만족 못 하면 그 프레임은 무효(y_diff=0) 처리.
5. y_diff = (아래 y) - (위 y). 인덱스 vs y_diff 그래프를 그리고 최대 프레임을
   레퍼런스로 선정. 360도 회전이므로 약 180도 간격의 피크 2개가 나온다.

사용법
------
python find_ref_angle.py "G:\\내 드라이브\\...\\옆" [--thresh 210] [--x-tol 20]
                         [--min-sep 90] [--out ref_angle_out]
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

INVALID = {"x_min": -1, "x_top": -1, "y_top": -1, "x_bot": -1, "y_bot": -1,
           "y_diff": 0, "dx": -1, "valid": 0}


def natural_key(p: Path):
    """img_0002 < img_0010 처럼 파일명 속 숫자 기준 정렬."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def imread_gray(path: Path):
    """한글/유니코드 경로에서도 동작하는 그레이스케일 로더."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite_uni(path: Path, img):
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


def measure(img: np.ndarray, thresh: int, x_tol: int):
    """한 이미지에서 맨왼쪽 위/아래 점과 y_diff 측정.

    x가 y보다 우선: 반쪽별로 최좌단 x를 먼저 확정한 뒤, 최좌단+x_tol 이내
    픽셀 중에서 y 극값을 고른다 (둥근 모서리의 꼭대기까지 포함).
    반환 dict의 valid=0이면 무효 프레임(reason에 사유).
    """
    h, w = img.shape
    cy = h // 2
    dark = (img < thresh).astype(np.uint8)

    # 시드: 맨 오른쪽 열에서 세로 중앙에 가장 가까운 어두운 픽셀
    right_col = np.flatnonzero(dark[:, w - 1])
    if right_col.size == 0:
        return {**INVALID, "reason": "오른쪽 끝에 툴 없음"}
    seed_y = int(right_col[np.argmin(np.abs(right_col - h // 2))])

    # 시드와 연결된 덩어리만 툴로 취급 (왼쪽 배경 구조물/먼지 제거)
    _, labels = cv2.connectedComponents(dark, connectivity=8)
    tool = labels == labels[seed_y, w - 1]

    ys, xs = np.nonzero(tool)
    x_min = int(xs.min())

    # 중심 위/아래 반쪽 각각에서 최좌단 x 픽셀을 찾고, 같은 x면 y 극값 선택
    up, dn = ys < cy, ys > cy
    if not up.any() or not dn.any():
        return {**INVALID, "x_min": x_min,
                "reason": "위쪽 점 없음" if not up.any() else "아래쪽 점 없음"}

    band_up = up & (xs <= xs[up].min() + x_tol)
    y_top = int(ys[band_up].min())
    x_top = int(xs[band_up][ys[band_up] == y_top].min())
    band_dn = dn & (xs <= xs[dn].min() + x_tol)
    y_bot = int(ys[band_dn].max())
    x_bot = int(xs[band_dn][ys[band_dn] == y_bot].min())
    dx = abs(x_top - x_bot)

    return {"x_min": x_min, "x_top": x_top, "y_top": y_top,
            "x_bot": x_bot, "y_bot": y_bot, "y_diff": y_bot - y_top, "dx": dx,
            "valid": 1, "reason": ""}


def annotate(img, m, out_path: Path):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    cv2.line(vis, (0, h // 2), (w - 1, h // 2), (0, 200, 200), 2)
    if m["x_min"] >= 0:
        cv2.line(vis, (m["x_min"], 0), (m["x_min"], h - 1), (0, 255, 0), 2)
    if m["x_top"] >= 0:
        cv2.circle(vis, (m["x_top"], m["y_top"]), 25, (0, 0, 255), 5)
        cv2.circle(vis, (m["x_bot"], m["y_bot"]), 25, (255, 0, 0), 5)
    label = f"y_diff={m['y_diff']} dx={m['dx']}" if m["valid"] else f"invalid: {m['reason']}"
    cv2.putText(vis, label, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    imwrite_uni(out_path, vis)


def main():
    ap = argparse.ArgumentParser(description="엔드밀 옆날 레퍼런스 각도 찾기")
    ap.add_argument("folder", help="이미지 폴더 경로")
    ap.add_argument("--thresh", type=int, default=210, help="이진화 임계값 (기본 210, 이보다 어두우면 툴)")
    ap.add_argument("--x-tol", type=int, default=20, help="최좌단에서 y 극값 경쟁에 참여시킬 x 허용 오차(px, 기본 20)")
    ap.add_argument("--min-sep", type=int, default=90, help="1·2위 피크 사이 최소 인덱스 간격 (기본 90)")
    ap.add_argument("--out", default="ref_angle_out", help="결과 저장 폴더 (기본 ./ref_angle_out)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    folder = Path(args.folder)
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS], key=natural_key)
    if not files:
        sys.exit(f"이미지 파일이 없습니다: {folder}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(files)}장 처리 시작 (thresh={args.thresh}, x우선, x_tol={args.x_tol})")
    t0 = time.time()
    rows = []
    for i, p in enumerate(files):
        img = imread_gray(p)
        if img is None:
            m = {**INVALID, "reason": "읽기 실패"}
        else:
            m = measure(img, args.thresh, args.x_tol)
        rows.append({"index": i, "file": p.name, **m})
        if (i + 1) % 40 == 0 or i == len(files) - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (len(files) - i - 1)
            print(f"  {i + 1}/{len(files)}  경과 {el:5.1f}s  남은시간 약 {eta:4.0f}s")

    y_diffs = np.array([r["y_diff"] for r in rows])
    valid = np.array([r["valid"] for r in rows], dtype=bool)
    n_invalid = int((~valid).sum())

    # 1위 피크 = 레퍼런스, 2위 피크 = min_sep 이상 떨어진 곳의 최대값
    i1 = int(np.argmax(y_diffs))
    far = np.abs(np.arange(len(rows)) - i1) >= args.min_sep
    i2 = int(np.arange(len(rows))[far][np.argmax(y_diffs[far])]) if far.any() else None

    # CSV 저장
    csv_path = out_dir / "ref_angle_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    # 그래프: x=이미지 인덱스, y=y픽셀수 차이 (무효 프레임은 0 + 회색 x 표시)
    fig, ax = plt.subplots(figsize=(14, 6))
    idx_all = np.arange(len(rows))
    ax.plot(idx_all, y_diffs, "-", lw=1, color="#4477aa")
    if n_invalid:
        ax.plot(idx_all[~valid], np.zeros(n_invalid), "x", ms=5, color="#bbbbbb",
                label=f"invalid ({n_invalid})")
        ax.legend(loc="upper right")
    for rank, ii in enumerate([i1, i2]):
        if ii is None:
            continue
        c = "#cc3311" if rank == 0 else "#ee7733"
        ax.plot(ii, y_diffs[ii], "o", ms=9, color=c)
        ax.annotate(f"#{rank + 1}: {rows[ii]['file']}\nidx={ii}, y_diff={y_diffs[ii]}",
                    (ii, y_diffs[ii]), textcoords="offset points", xytext=(10, -20 - 25 * rank),
                    fontsize=9, color=c)
    ax.set_xlabel("image index")
    ax.set_ylabel("y-pixel difference (leftmost top vs bottom)")
    ax.set_title(f"Reference angle search — {folder.name} "
                 f"({len(rows)} images, x-first, x_tol={args.x_tol})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "ref_angle_plot.png"
    fig.savefig(plot_path, dpi=120)

    # 상위 피크 주석 이미지 저장
    for rank, ii in enumerate([i1, i2]):
        if ii is None:
            continue
        r = rows[ii]
        annotate(imread_gray(folder / r["file"]), r,
                 out_dir / f"ref_top{rank + 1}_{Path(r['file']).stem}_annotated.png")

    print()
    print("=" * 60)
    print(f"유효 {len(rows) - n_invalid}장 / 무효 {n_invalid}장")
    print(f"레퍼런스(1위): idx={i1}  {rows[i1]['file']}  y_diff={y_diffs[i1]}  "
          f"top=({rows[i1]['x_top']},{rows[i1]['y_top']}) bot=({rows[i1]['x_bot']},{rows[i1]['y_bot']})")
    if i2 is not None:
        print(f"2위 피크    : idx={i2}  {rows[i2]['file']}  y_diff={y_diffs[i2]}  "
              f"(1위와 {abs(i2 - i1)}장 차이)")
    print("-" * 60)
    order = np.argsort(y_diffs)[::-1][:10]
    print("y_diff 상위 10:")
    for ii in order:
        print(f"  idx={ii:4d}  {rows[ii]['file']}  y_diff={y_diffs[ii]}  dx={rows[ii]['dx']}")
    print("-" * 60)
    print(f"결과 저장: {csv_path}")
    print(f"          {plot_path}")
    print(f"          {out_dir}\\ref_top*_annotated.png")


if __name__ == "__main__":
    main()
