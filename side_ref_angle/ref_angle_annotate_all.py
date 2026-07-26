# -*- coding: utf-8 -*-
"""find_ref_angle.py가 만든 CSV를 읽어 모든 이미지에 측정 결과를 표시.

- 각 이미지에 최좌단 x선(초록), 중심선(노랑), 맨왼쪽 위(빨강)/아래(파랑) 점,
  y_diff·dx 값을 그려서 <out>\annotated\ 에 축소 저장 (기본 35%)
- 스크롤로 훑어볼 수 있는 index.html 생성 (그래프 + 전체 썸네일 그리드)
- 주석 프레임을 이어붙인 동영상 <out>\ref_angle_annotated.mp4 생성 (기본 60fps)

사용법: python ref_angle_annotate_all.py "이미지폴더" [--out ref_angle_out]
        [--scale 0.35] [--fps 60]   (--fps 0 이면 영상 생략)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def imread_gray(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="원본 이미지 폴더")
    ap.add_argument("--out", default="ref_angle_out", help="find_ref_angle.py 결과 폴더")
    ap.add_argument("--scale", type=float, default=0.35, help="저장 축소 비율 (기본 0.35)")
    ap.add_argument("--fps", type=int, default=60, help="동영상 fps (기본 60, 0이면 영상 생략)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    folder = Path(args.folder)
    out_dir = Path(args.out)
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ref_angle_scores.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("index", "x_min", "x_top", "y_top", "x_bot", "y_bot", "y_diff", "dx"):
            r[k] = int(r[k])
        r["valid"] = int(r.get("valid", 1))
        r["reason"] = r.get("reason", "")

    y = np.array([r["y_diff"] for r in rows])
    i1 = int(np.argmax(y))
    far = np.abs(np.arange(len(rows)) - i1) >= 90
    i2 = int(np.arange(len(rows))[far][np.argmax(y[far])]) if far.any() else -1

    print(f"{len(rows)}장 주석 생성 시작 (scale={args.scale}, fps={args.fps})")
    t0 = time.time()
    entries = []
    writer = None
    video_path = out_dir / "ref_angle_annotated.mp4"
    for n, r in enumerate(rows):
        img = imread_gray(folder / r["file"])
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape
        cv2.line(vis, (0, h // 2), (w - 1, h // 2), (0, 200, 200), 3)
        if r["x_min"] >= 0:
            cv2.line(vis, (r["x_min"], 0), (r["x_min"], h - 1), (0, 255, 0), 3)
        if r["x_top"] >= 0:
            cv2.circle(vis, (r["x_top"], r["y_top"]), 30, (0, 0, 255), 8)
            cv2.circle(vis, (r["x_bot"], r["y_bot"]), 30, (255, 0, 0), 8)
        label = (f"idx={r['index']} y_diff={r['y_diff']} dx={r['dx']}" if r["valid"]
                 else f"idx={r['index']} invalid: {r['reason']}")
        cv2.putText(vis, label, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
        vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        name = f"idx{r['index']:04d}_{Path(r['file']).stem}_yd{r['y_diff']}.png"
        ok, buf = cv2.imencode(".png", vis)
        if ok:
            buf.tofile(str(ann_dir / name))
        entries.append((r, name))
        if args.fps > 0:
            vh, vw = vis.shape[0] // 2 * 2, vis.shape[1] // 2 * 2  # 코덱 호환용 짝수 크기
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (vw, vh))
            writer.write(vis[:vh, :vw])
        if (n + 1) % 40 == 0 or n == len(rows) - 1:
            el = time.time() - t0
            print(f"  {n + 1}/{len(rows)}  경과 {el:5.1f}s")

    # index.html 생성
    cells = []
    for r, name in entries:
        cls = "top1" if r["index"] == i1 else ("top2" if r["index"] == i2 else "")
        stat = (f'y_diff <b>{r["y_diff"]}</b> · dx {r["dx"]}' if r["valid"]
                else f'<span style="color:#e77">invalid: {r["reason"]}</span>')
        cells.append(
            f'<figure class="{cls}"><a href="{name}" target="_blank">'
            f'<img src="{name}" loading="lazy"></a>'
            f'<figcaption>idx {r["index"]} · {r["file"]}<br>{stat}</figcaption></figure>'
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ref angle - all annotated</title>
<style>
body{{font-family:'Malgun Gothic',sans-serif;background:#1b1b1f;color:#ddd;margin:16px}}
img{{width:100%;display:block;border-radius:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}}
figure{{margin:0;padding:4px;background:#26262b;border-radius:6px;border:2px solid transparent}}
figure.top1{{border-color:#e33}} figure.top2{{border-color:#e93}}
figcaption{{font-size:12px;padding:4px 2px;color:#aaa}}
.plot img{{max-width:1400px;margin:0 auto 20px}}
h1{{font-size:18px}}
</style></head><body>
<h1>레퍼런스 각도 탐색 — 전체 {len(rows)}장 주석 (빨간 테두리=1위 idx {i1}, 주황=2위 idx {i2})</h1>
<div class="plot"><a href="../ref_angle_plot.png" target="_blank"><img src="../ref_angle_plot.png"></a></div>
<div class="grid">{''.join(cells)}</div>
</body></html>"""
    (ann_dir / "index.html").write_text(html, encoding="utf-8")

    if writer is not None:
        writer.release()
        print(f"동영상 저장: {video_path} ({len(rows)}프레임, {args.fps}fps)")
    print(f"완료: {ann_dir}\\index.html (브라우저로 여세요)")


if __name__ == "__main__":
    main()
