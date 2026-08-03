# -*- coding: utf-8 -*-
"""옆날 이미지에서 SAM으로 날 영역 검출 (뾰족한 날끝까지 복원).

파이프라인
----------
1. 임계값(기본 210) + 오른쪽 가운데 시드 연결요소로 툴 실루엣과 날끝 좌표를 자동 검출
2. SAM 프롬프트 구성: 클릭점(기본 정중앙) + 날끝 안쪽 양성점, 몸통/배경 음성점 3개
3. multimask 예측 → 최고 score 마스크의 로짓을 재입력해 정제(refine)
4. 날끝 패치: SAM은 내부 해상도(1024) 한계로 폭 수 px의 바늘 같은 날끝을 뭉개므로,
   날끝 반경(기본 60px) 안의 실루엣 픽셀 중 날끝에 닿는 조각만 합집합
   → 날끝을 실루엣 정밀도(±1px)로 복원

detect_blade()를 임포트해서 다른 스크립트(wear_side_compare.py 등)에서 재사용 가능.

사용법
------
python sam_blade_detect.py "이미지경로" [--point X Y] [--model vit_h|mobile]
                           [--thresh 210] [--tip-radius 60]
                           [--out sam_blade_out] [--scale 0.35]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

# SAM 체크포인트(2.4GB)는 용량 때문에 저장소에 없다. 아래 위치를 순서대로 찾는다.
#   1) 환경변수 SAM_MODELS_DIR 로 지정한 폴더
#   2) 저장소 최상위의 sam_models/        ← 권장. 옆날·아랫날이 함께 쓴다
#   3) tool_wear_v2/code/sam_models/      ← 옛 위치 (하위 호환)
_SAM_CANDIDATES = [
    Path(os.environ["SAM_MODELS_DIR"]) if os.environ.get("SAM_MODELS_DIR") else None,
    HERE.parent / "sam_models",
    HERE.parent / "tool_wear_v2" / "code" / "sam_models",
]


def sam_dir(ckpt_name=None):
    """체크포인트 폴더를 찾아 반환. ckpt_name 을 주면 그 파일이 있는 폴더를 고른다."""
    for d in _SAM_CANDIDATES:
        if d is None:
            continue
        if (d / ckpt_name).exists() if ckpt_name else d.is_dir():
            return d
    return _SAM_CANDIDATES[1]          # 없으면 권장 위치를 반환(오류 메시지용)


SAM_DIR = _SAM_CANDIDATES[1]
MODELS = {
    "vit_h": ("vit_h", "sam_vit_h_4b8939.pth", "segment_anything"),
    "mobile": ("vit_t", "mobile_sam.pt", "mobile_sam"),
}


def imread_gray(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def tool_silhouette(gray, thresh=210):
    """오른쪽 가운데 시드와 연결된 어두운 픽셀 덩어리(툴)와 날끝 좌표."""
    h, w = gray.shape
    dark = (gray < thresh).astype(np.uint8)
    seed_ys = np.flatnonzero(dark[:, w - 1])
    if seed_ys.size == 0:
        raise RuntimeError("오른쪽 끝에서 툴을 찾지 못함 (thresh 확인)")
    seed_y = int(seed_ys[np.argmin(np.abs(seed_ys - h // 2))])
    _, labels = cv2.connectedComponents(dark, connectivity=8)
    tool = labels == labels[seed_y, w - 1]
    ys, xs = np.nonzero(tool)
    x_min = int(xs.min())
    tip = (x_min, int(np.median(ys[xs <= x_min + 3])))
    return tool, tip


def merge_height(tool, tip):
    """날끝 포함 실루엣 런이 날끝 위로 뻗은 높이 M(px).

    반대날 끝면 덩어리가 날끝과 투영이 겹치면 크고(수십~수백), 날끝이 배경으로
    고립돼 있으면 0. 날끝 패치 필터의 강도 결정과 판별 프레임 선정에 사용.
    """
    x_min, tip_y = tip
    hs = []
    for x in range(x_min + 15, x_min + 121, 5):
        col = np.flatnonzero(tool[:, x])
        if len(col) == 0:
            continue
        runs = np.split(col, np.flatnonzero(np.diff(col) > 1) + 1)
        run = next((r for r in runs if r.min() - 40 <= tip_y <= r.max() + 40), None)
        if run is not None:
            hs.append(max(0, tip_y - int(run.min())))
    return float(np.median(hs)) if hs else -1.0


def get_predictor(model="vit_h"):
    """SAM predictor 생성 (호출측에서 재사용 권장 — 로드에 ~10s)."""
    import importlib
    import torch
    key, ckpt_name, pkg = MODELS[model]
    ckpt = sam_dir(ckpt_name) / ckpt_name
    if not ckpt.exists():
        tried = "\n  ".join(str(d / ckpt_name) for d in _SAM_CANDIDATES if d)
        raise FileNotFoundError(
            f"[SAM] 체크포인트 '{ckpt_name}' 를 찾지 못했습니다. 찾아본 위치:\n  {tried}\n"
            f"  → 권장: 저장소 최상위에 sam_models/ 폴더를 만들고 그 안에 두세요. "
            f"(환경변수 SAM_MODELS_DIR 로 다른 위치 지정 가능)")
    mod = importlib.import_module(pkg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sam = mod.sam_model_registry[key](checkpoint=str(ckpt))
    sam.to(device=dev)
    sam.eval()
    return mod.SamPredictor(sam)


def detect_blade(gray, point=None, predictor=None, thresh=210, tip_radius=60,
                 negs=None):
    """날 영역 검출. 반환: dict(final, base, tool, tip, points, labels).

    negs: 음성 프롬프트 좌표 리스트. None 이면 전체 프레임용 기본 3점.
    (크롭 이미지에서는 기본 위치가 날 위에 떨어질 수 있어 호출측이 지정)
    """
    if predictor is None:
        predictor = get_predictor()
    h, w = gray.shape
    cx, cy = point if point else (w // 2, h // 2)

    tool, tip = tool_silhouette(gray, thresh)
    v = np.array([cx - tip[0], cy - tip[1]], dtype=float)
    n = np.linalg.norm(v)
    if n < 1:
        raise ValueError("클릭점이 날끝과 같음 — 다른 위치를 지정하세요")
    v /= n
    tip_in = (int(tip[0] + 60 * v[0]), int(tip[1] + 60 * v[1]))
    if negs is None:
        negs = [(w - 150, cy), (int(w * 0.75), int(h * 0.2)),
                (max(50, tip[0] - 300), cy)]

    predictor.set_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    pts = np.array([(cx, cy), tip_in] + list(negs))
    lbl = np.array([1, 1] + [0] * len(negs))
    masks, scores, logits = predictor.predict(point_coords=pts, point_labels=lbl,
                                              multimask_output=True)
    best = int(np.argmax(scores))
    refined, _, _ = predictor.predict(point_coords=pts, point_labels=lbl,
                                      mask_input=logits[best][None],
                                      multimask_output=False)
    base = refined[0].astype(bool)

    # 날끝 패치 후보: 날끝 반경 안의 실루엣 픽셀
    R = tip_radius
    Y, X = np.ogrid[:h, :w]
    patch = (tool & ((X - tip[0]) ** 2 + (Y - tip[1]) ** 2 <= R * R) & ~base).astype(np.uint8)

    # 필터 강도는 적응형: 날끝이 배경으로 고립(M=0)돼 있으면 반경+닿음 조건만으로
    # 충분하므로 부채꼴/볼록껍질 필터를 건너뛰어 바늘 가장자리를 깎지 않는다.
    # 반대날 끝면 덩어리가 붙어 있는(M>0) 각도에서만 엄격 필터로 돌출을 방지.
    if merge_height(tool, tip) > 0:
        # 날끝에서 본 base(날 웨지)의 각도 부채꼴 — 패치는 이 방향 범위 안만 허용
        # (날끝 = 실루엣 최좌단이라 모든 픽셀이 오른쪽에 있어 각도 wrap 없음)
        byy, bxx = np.nonzero(base)
        bd2 = (bxx - tip[0]) ** 2 + (byy - tip[1]) ** 2
        nb = (bd2 <= 80 * 80) & (bd2 >= 5 * 5)
        if nb.sum() >= 50:
            ang_b = np.degrees(np.arctan2(byy[nb] - tip[1], bxx[nb] - tip[0]))
            # min/max 는 경계의 소수 픽셀에 부채꼴이 확 열리므로 백분위로 강건하게
            ang_lo, ang_hi = np.percentile(ang_b, 2) - 5, np.percentile(ang_b, 98) + 5
            pyy, pxx = np.nonzero(patch)
            ang_p = np.degrees(np.arctan2(pyy - tip[1], pxx - tip[0]))
            out_of_sector = (ang_p < ang_lo) | (ang_p > ang_hi)
            patch[pyy[out_of_sector], pxx[out_of_sector]] = 0

        # 볼록껍질 제약: 날끝 복원은 볼록 모서리를 메우는 것 — 날끝과 base를 잇는
        # 국소 볼록껍질 밖(능선 위 돌출 등)의 패치 픽셀은 제거
        x0, x1 = max(tip[0] - 20, 0), min(tip[0] + 260, w)
        y0, y1 = max(tip[1] - 260, 0), min(tip[1] + 260, h)
        sub = base[y0:y1, x0:x1]
        sys_, sxs = np.nonzero(sub)
        if len(sxs) >= 3:
            pts = np.vstack([np.column_stack([sxs, sys_]),
                             [[tip[0] - x0, tip[1] - y0]]]).astype(np.int32)
            hullm = np.zeros(sub.shape, np.uint8)
            cv2.fillConvexPoly(hullm, cv2.convexHull(pts), 1)
            hullm = cv2.dilate(hullm, np.ones((5, 5), np.uint8))
            ok = np.zeros((h, w), bool)
            ok[y0:y1, x0:x1] = hullm.astype(bool)
            patch[~ok] = 0

    _, plab = cv2.connectedComponents(patch, connectivity=8)
    tip_patch = np.zeros_like(patch, dtype=bool)
    for pl in range(1, plab.max() + 1):
        yy, xx = np.nonzero(plab == pl)
        if np.min(np.hypot(xx - tip[0], yy - tip[1])) <= 30:
            tip_patch |= plab == pl
    merged = (base | tip_patch).astype(np.uint8)
    _, lab = cv2.connectedComponents(merged, connectivity=8)
    final = lab == np.bincount(lab[base]).argmax()

    return {"final": final, "base": base, "tool": tool, "tip": tip,
            "points": pts, "labels": lbl}


def main():
    ap = argparse.ArgumentParser(description="SAM 날 영역 검출 (날끝 복원 포함)")
    ap.add_argument("image", help="이미지 경로")
    ap.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"),
                    help="클릭 좌표 (기본: 이미지 정중앙)")
    ap.add_argument("--model", choices=list(MODELS), default="vit_h")
    ap.add_argument("--thresh", type=int, default=210, help="실루엣 이진화 임계값 (기본 210)")
    ap.add_argument("--tip-radius", type=int, default=60,
                    help="날끝 복원 패치 반경 px (기본 60)")
    ap.add_argument("--out", default="sam_blade_out", help="결과 저장 폴더")
    ap.add_argument("--scale", type=float, default=0.35, help="오버레이 축소 비율")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    img_path = Path(args.image)
    gray = imread_gray(img_path)
    if gray is None:
        sys.exit(f"이미지를 읽을 수 없음: {img_path}")
    h, w = gray.shape
    cx, cy = args.point if args.point else (w // 2, h // 2)
    print(f"{img_path.name} ({w}x{h})  클릭=({cx},{cy})")

    t0 = time.time()
    predictor = get_predictor(args.model)
    print(f"모델 로드 {time.time()-t0:.1f}s")

    t0 = time.time()
    r = detect_blade(gray, (cx, cy), predictor, args.thresh, args.tip_radius)
    final, tip = r["final"], r["tip"]
    yy, xx = np.nonzero(final)
    tip_dist = float(np.min(np.hypot(xx - tip[0], yy - tip[1])))
    print(f"검출 {time.time()-t0:.1f}s  area={100*final.sum()/(w*h):.2f}%  "
          f"날끝={tip}  날끝까지 {tip_dist:.1f}px")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{img_path.stem}_pt{cx}x{cy}"

    cv2.imencode(".png", (final * 255).astype(np.uint8))[1].tofile(
        str(out_dir / f"{stem}_mask.png"))

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    layer = vis.copy()
    layer[final] = (0, 165, 255)
    vis = cv2.addWeighted(layer, 0.4, vis, 0.6, 0)
    cnts, _ = cv2.findContours(final.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 165, 255), 3)
    for (px, py), l in zip(r["points"].tolist(), r["labels"].tolist()):
        cv2.drawMarker(vis, (px, py), (0, 255, 0) if l else (0, 0, 255),
                       cv2.MARKER_CROSS, 50, 5)
    cv2.circle(vis, tip, 20, (255, 255, 0), 3)
    cv2.putText(vis, f"blade area={100*final.sum()/(w*h):.1f}% tipdist={tip_dist:.0f}px",
                (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 5)
    small = cv2.resize(vis, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", small)[1].tofile(str(out_dir / f"{stem}_overlay.png"))

    zx0, zy0 = max(tip[0] - 150, 0), max(tip[1] - 250, 0)
    crop = vis[zy0:zy0 + 500, zx0:zx0 + 500]
    cv2.imencode(".png", crop)[1].tofile(str(out_dir / f"{stem}_tipzoom.png"))
    print(f"저장: {out_dir}\\{stem}_mask|overlay|tipzoom.png")


if __name__ == "__main__":
    main()
