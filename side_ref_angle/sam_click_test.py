# -*- coding: utf-8 -*-
"""옆날 이미지에서 클릭 점 프롬프트로 SAM 세그멘테이션 테스트.

클릭 위치(기본: 정중앙) 1점을 양성 프롬프트로 주고 multimask 3개(조각/부품/전체)를
오버레이 이미지로 저장한다. 날 영역 인식 가능성 확인용.

사용법
------
python sam_click_test.py "이미지경로" [--point X Y] [--model vit_h|mobile]
                         [--out sam_click_out] [--scale 0.35]

체크포인트는 tool_wear_v2/code/sam_models/ 의 것을 재사용한다
(vit_h = sam_vit_h_4b8939.pth, mobile = mobile_sam.pt).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SAM_DIR = HERE.parent / "tool_wear_v2" / "code" / "sam_models"
MODELS = {  # model_id → (registry_key, 체크포인트, import 패키지)
    "vit_h": ("vit_h", "sam_vit_h_4b8939.pth", "segment_anything"),
    "mobile": ("vit_t", "mobile_sam.pt", "mobile_sam"),
}
COLORS = [(0, 0, 255), (0, 165, 255), (255, 0, 255)]  # BGR


def imread_gray(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def main():
    ap = argparse.ArgumentParser(description="SAM 클릭 점 프롬프트 테스트")
    ap.add_argument("image", help="이미지 경로")
    ap.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"),
                    help="클릭 좌표 (기본: 이미지 정중앙)")
    ap.add_argument("--model", choices=list(MODELS), default="vit_h")
    ap.add_argument("--out", default="sam_click_out", help="결과 저장 폴더")
    ap.add_argument("--scale", type=float, default=0.35, help="저장 축소 비율")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import importlib
    import torch

    img_path = Path(args.image)
    gray = imread_gray(img_path)
    if gray is None:
        sys.exit(f"이미지를 읽을 수 없음: {img_path}")
    h, w = gray.shape
    cx, cy = args.point if args.point else (w // 2, h // 2)
    print(f"{img_path.name} ({w}x{h}), 클릭점 = ({cx}, {cy}), 밝기 = {gray[cy, cx]}")

    key, ckpt_name, pkg = MODELS[args.model]
    ckpt = SAM_DIR / ckpt_name
    if not ckpt.exists():
        sys.exit(f"[SAM] 체크포인트 없음: {ckpt}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = importlib.import_module(pkg)

    t0 = time.time()
    sam = mod.sam_model_registry[key](checkpoint=str(ckpt))
    sam.to(device=dev)
    sam.eval()
    predictor = mod.SamPredictor(sam)
    print(f"모델 로드 {time.time()-t0:.1f}s (model={args.model}, device={dev})")

    t0 = time.time()
    predictor.set_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    print(f"이미지 인코딩 {time.time()-t0:.1f}s")

    masks, scores, _ = predictor.predict(point_coords=np.array([[cx, cy]]),
                                         point_labels=np.array([1]),
                                         multimask_output=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    for i, (m, s) in enumerate(zip(masks, scores)):
        area = int(m.sum())
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        layer = vis.copy()
        layer[m] = COLORS[i]
        vis = cv2.addWeighted(layer, 0.35, vis, 0.65, 0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, COLORS[i], 4)
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 60, 6)
        cv2.putText(vis, f"mask{i} score={s:.3f} area={area} ({100*area/(w*h):.1f}%)",
                    (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLORS[i], 5)
        small = cv2.resize(vis, None, fx=args.scale, fy=args.scale,
                           interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"{stem}_pt{cx}x{cy}_mask{i}.png"
        cv2.imencode(".png", small)[1].tofile(str(out_path))
        print(f"mask{i}: score={s:.4f}  area={area}px ({100*area/(w*h):.1f}%)  → {out_path}")


if __name__ == "__main__":
    main()
