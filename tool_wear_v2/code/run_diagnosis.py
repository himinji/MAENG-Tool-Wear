# -*- coding: utf-8 -*-
"""SAM 기반 밑면 마모 진단 (검출부터 새 파이프라인).

흐름: 정렬 → 수직화 → SAM(새)+SAM(마모) → 날별 외곽선 후퇴 = 마모(µm)
출력 폴더에 순서대로:
  1_align.png        정렬된 새|마모
  2_sam_new.png      SAM 새 공구 검출
  3_sam_worn.png     SAM 마모 공구 검출
  4_blade1_long.png  날1(긴 날) 외곽선 비교 + 후퇴점
  5_blade2_short.png 날2(짧은 날) 외곽선 비교 + 후퇴점
  summary.png        종합(위 5장 + VB 표)

사용:
  venv\Scripts\python tool_wear_v2\code\run_diagnosis.py ^
    --new  "G:\...\260709_SM45C\Initial" --worn "G:\...\260709_SM45C\Test44"
"""
import os
import sys
import glob
import argparse
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(HERE, "_wear_out")
sys.path.insert(0, HERE)
import botface_pipeline_v2 as bp
import blade_region          # overlay 시각화용
import blade_sam
import blade_wear
import blade_align


def face(folder):
    for sub in ("", "아래", os.path.join("Toolwear", "아래")):
        d = os.path.join(folder, sub) if sub else folder
        if glob.glob(os.path.join(d, "img_*.png")):
            return d
    raise SystemExit(f"[오류] img_*.png 없음: {folder}")


def canonical_pair(new_folder, worn_folder, diam, log=print):
    """정렬·수직화된 (새 nv, 마모 wv, 축, r, um/px, reg_n, reg_w) 반환.
    수직화·정합을 전부 SAM 마스크 기반으로 함 (Hough verticalize/워터셰드 보정 제거).
    거친 회전정렬(align_rotation) → blade_align.refine(SAM 수직+몸통정합 반복)."""
    ref = bp.find_axis_scale(new_folder, diam, log=log)
    axis_n, r, um = ref['axis'], ref['od_r'], ref['um_per_px']
    wa = bp.find_axis_scale(worn_folder, diam, n_avg=90, log=log)
    S = int(3.0 * r)   # 날 끝 잘림 방지(형상 ~1.19r + 회전 여유, ±1.5r)
    fnew = bp.pick_frame(new_folder, axis_n, r, log=log)
    fworn = bp.pick_frame(worn_folder, wa['axis'], wa['od_r'], log=log)
    ncrop, axc = bp.axis_crop(fnew, axis_n, S)
    wcrop, _ = bp.axis_crop(fworn, wa['axis'], S)
    nfill, _ = bp.ring_fill(ncrop, axc, r)
    wfill, _ = bp.ring_fill(wcrop, axc, r)
    # 거친 회전정렬(마모를 새에 대략 겹침) — 위상차 흡수. 미세 자세는 SAM 이 확정.
    _, ang, _ = bp.align_rotation(nfill, wfill, axc, r, log=log)
    wcrop_al = cv2.warpAffine(wcrop, cv2.getRotationMatrix2D((axc[0], axc[1]), ang, 1.0),
                              (S, S), flags=cv2.INTER_LINEAR)
    # 거친 수직화(Hough) — SAM 프롬프트가 날에 앉도록 대략 세움
    nv, wv = bp.verticalize(ncrop, wcrop_al, axc, r, log=log)
    # SAM 마스크 중심선으로 미세 수직 보정 (±8° 캡, 진짜 마모에 강건한 정밀화)
    nv, wv, _ = blade_align.verticalize(nv, wv, axc, r, blade_sam.detect, log=log)
    # 몸통(비마모) 평행이동 정합 — 그라디언트 ECC. 남는 잔차는 측정단계 2D 정렬이 처리.
    wv = bp.register_body(nv, wv, axc, r, log=log)
    reg_n = blade_sam.detect(nv, axc, r)
    reg_w = blade_sam.detect(wv, axc, r)
    return nv, wv, axc, r, um, reg_n, reg_w


def _outline(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def blade_crop(new, worn, mn, mw, res, S):
    ys, xs = np.nonzero(mn | mw)
    pad = 30
    x0, x1 = max(0, xs.min() - pad), min(S, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(S, ys.max() + pad)
    comp = cv2.addWeighted(new, 0.5, worn, 0.5, 0)[y0:y1, x0:x1]
    vis = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
    for m, col in ((mn, (255, 90, 0)), (mw, (0, 0, 255))):
        c = _outline(m)
        if c is not None:
            cv2.polylines(vis, [c - [x0, y0]], True, col, 2)
    vis = cv2.resize(vis, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST)
    if res:
        vmax = max(20.0, res['vb_um'])
        for (px, py, um) in res['retreat_pts']:
            t = min(1.0, um / vmax)
            cv2.circle(vis, (int((px - x0) * 1.6), int((py - y0) * 1.6)), 3,
                       (0, int(120 * (1 - t)), int(120 + 135 * t)), -1)
    return vis


def run(new, worn, out=_DEFAULT_OUT, diam=8.0, name=None):
    """SAM 기반 밑면 마모 진단 1회. 결과 폴더 경로 반환."""
    name = name or os.path.basename(os.path.normpath(worn))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(out, f"{stamp}_{name}")
    os.makedirs(outdir, exist_ok=True)
    nf, wf = face(new), face(worn)
    print(f"[기준] {nf}\n[진단] {wf}\n[결과] {outdir}\\\n" + "-" * 60)

    nv, wv, axc, r, um, reg_n, reg_w = canonical_pair(nf, wf, diam)
    print("-" * 60)
    S = nv.shape[0]

    # 저장물
    side = np.hstack([cv2.cvtColor(nv, cv2.COLOR_GRAY2BGR),
                      np.full((S, 8, 3), 255, np.uint8),
                      cv2.cvtColor(wv, cv2.COLOR_GRAY2BGR)])
    bp.imwrite_unicode(os.path.join(outdir, "1_align.png"), side)
    bp.imwrite_unicode(os.path.join(outdir, "2_sam_new.png"),
                       blade_region.overlay(nv, reg_n, axc, r))
    bp.imwrite_unicode(os.path.join(outdir, "3_sam_worn.png"),
                       blade_region.overlay(wv, reg_w, axc, r))

    crops = {}
    results = {}
    for mkey, fn, bname in (("mask1", "4_blade1_long.png", "날1(긴 날)"),
                            ("mask2", "5_blade2_short.png", "날2(짧은 날)")):
        res = blade_wear.measure(reg_n[mkey], reg_w[mkey], axc, r, um)
        results[bname] = res
        crop = blade_crop(nv, wv, reg_n[mkey], reg_w[mkey], res, S)
        crops[bname] = crop
        bp.imwrite_unicode(os.path.join(outdir, fn), crop)
        if res:
            print(f"  {bname}: VB={res['vb_um']:.1f}µm  코너={res['tip_um']:.1f}µm")
        else:
            print(f"  {bname}: 측정불가")

    # 종합 도표
    for fp in ('Malgun Gothic', 'AppleGothic', 'NanumGothic'):
        try:
            plt.rcParams['font.family'] = fp; break
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3)
    for ax_, img, ttl in (
        (fig.add_subplot(gs[0, 0]), cv2.imread(os.path.join(outdir, "2_sam_new.png")), "SAM 새 공구"),
        (fig.add_subplot(gs[0, 1]), cv2.imread(os.path.join(outdir, "3_sam_worn.png")), "SAM 마모 공구"),
    ):
        ax_.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax_.set_title(ttl); ax_.axis('off')
    for i, (bname, crop) in enumerate(crops.items()):
        ax_ = fig.add_subplot(gs[1, i])
        ax_.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = results[bname]
        t = f"VB={res['vb_um']:.1f}µm  코너={res['tip_um']:.1f}µm" if res else "측정불가"
        ax_.set_title(f"{bname}  ·  {t}\n(파랑=새 / 빨강=마모 / 점=후퇴)", fontsize=11)
        ax_.axis('off')
    axt = fig.add_subplot(gs[:, 2]); axt.axis('off')
    lines = [f"{name}  밑면 마모 진단", f"스케일 {um:.2f} µm/px", ""]
    for bname, res in results.items():
        if res:
            lines += [f"{bname}", f"  VB(플랭크)  {res['vb_um']:.1f} µm",
                      f"  코너 마모   {res['tip_um']:.1f} µm", ""]
    axt.text(0.02, 0.98, "\n".join(lines), va='top', ha='left', fontsize=14,
             linespacing=1.6)
    fig.suptitle(f"{name}  ·  SAM 외곽선 마모 진단", fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary.png"), dpi=130, facecolor='white')
    plt.close(fig)
    print("-" * 60 + f"\n완료 → {outdir}")
    return outdir


def main():
    ap = argparse.ArgumentParser(description="SAM 기반 밑면 마모 진단")
    ap.add_argument("--new", required=True)
    ap.add_argument("--worn", required=True)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--diam", type=float, default=8.0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    run(args.new, args.worn, args.out, args.diam, args.name)


if __name__ == "__main__":
    main()
