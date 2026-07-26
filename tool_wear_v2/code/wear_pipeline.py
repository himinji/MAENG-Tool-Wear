# -*- coding: utf-8 -*-
"""SAM 마모 진단 파이프라인 코어 (설정으로 변형 전환).

세 변형(run_v1/v2/v3)이 이 diagnose() 를 서로 다른 cfg 로 호출한다.
cfg 키:
  vertical      : 'watershed' | 'sam'   수직화 방식
  align         : 'tx' | 'iou'          측정 정렬(x평행이동 / 2D IoU)
  glint_suppress: bool                  SAM 포화픽셀 제거(152554 방식)
  frames        : 'single' | 'multi'    대표프레임 1장 / 여러장 중앙값
  mask_guard    : bool                  치즐/플루트 오검출 혹 자동 절단
  n_frames      : int                   frames='multi' 일 때 장수(기본 5)
  sam           : 'mobile'|'vit_b/l/h'  검출 모델(mobile=경량CPU / vit_*=정식 SAM, GPU)
"""
import os
import re
import sys
import glob
from functools import partial
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import botface_pipeline_v2 as bp
import blade_region
import blade_sam
import blade_wear
import blade_align


def face(folder):
    for sub in ("", "아래", os.path.join("Toolwear", "아래")):
        d = os.path.join(folder, sub) if sub else folder
        if glob.glob(os.path.join(d, "img_*.png")):
            return d
    raise SystemExit(f"[오류] img_*.png 없음: {folder}")


# 날끝(절삭 코너) 궤도 = TIP_RATIO · OD_r. 0709 SAM 날끝-날끝(um=5.20→r=769)/OD(650)=1.18 실측.
# 스케일 = 이 '날끝 궤도' 직경 기준(직경 = 2·TIP_RATIO·OD_r = 공구 직경). SAM 검출 불필요라 강건.
TIP_RATIO = 1.18


def _tip_to_tip_scale(reg, r, diam_mm):
    """스케일 = 직경 / (날끝~날끝 px).
    사용자 정의: 공구 직경 = SAM 검출 마스크의 최상단 픽셀 ~ 최하단 픽셀 높이차.
    (크롭이 날끝을 자르면 값이 줄어들지만, 크롭 배율은 사용자가 별도 조정.)"""
    m1, m2 = reg.get('mask1'), reg.get('mask2')
    if m1 is None or m2 is None:
        return None
    ys = np.nonzero(m1 | m2)[0]
    if ys.size < 50:
        return None
    tip_px = float(ys.max() - ys.min())          # 최상단 → 최하단
    if tip_px < 0.5 * r:                          # 검출 완전 실패 안전장치
        return None
    return diam_mm * 1000.0 / tip_px


def pick_frames(folder, axis, r, n, glint_pen=True):
    """글린트 페널티 준 상위 n 프레임 경로."""
    frames = bp.list_frames(folder)
    cx, cy = axis
    scored = []
    for f in frames[::8]:
        g = bp.imread_unicode(f)
        if g is None:
            continue
        x0, y0 = max(0, int(cx - r * 0.7)), max(0, int(cy - r * 0.7))
        patch = g[y0:int(cy + r * 0.7), x0:int(cx + r * 0.7)]
        if patch.size == 0:
            continue
        std = float(patch.std())
        sat = float((patch >= 250).mean())
        score = std * max(0.0, 1.0 - 3.0 * sat) if glint_pen else std
        scored.append((score, f))
    scored.sort(reverse=True)
    return [f for _, f in scored[:n]]


def _verticalize_watershed(nv, wv, axc, r, log):
    """152554 방식: blade_region.recognize + band_tilt 로 수직 보정."""
    reg = blade_region.recognize(nv, axc, r)
    tilt = blade_region.band_tilt(reg, axc, r) if reg else None
    S = nv.shape[0]
    for _ in range(3):
        if tilt is None or abs(tilt) <= 0.5:
            break
        cands = []
        for A in (-tilt, +tilt):
            Mr = cv2.getRotationMatrix2D((axc[0], axc[1]), A, 1.0)
            g2 = cv2.warpAffine(nv, Mr, (S, S), flags=cv2.INTER_LINEAR)
            r2 = blade_region.recognize(g2, axc, r)
            t2 = blade_region.band_tilt(r2, axc, r) if r2 else None
            if t2 is not None:
                cands.append((abs(t2), A, g2, t2))
        if not cands or min(cands, key=lambda c: c[0])[0] >= abs(tilt):
            break
        cands.sort(key=lambda c: c[0])
        _, A, nv, tilt = cands[0]
        Mr = cv2.getRotationMatrix2D((axc[0], axc[1]), A, 1.0)
        wv = cv2.warpAffine(wv, Mr, (S, S), flags=cv2.INTER_LINEAR)
    return nv, wv


def _canon_one(new_folder, worn_file, axis_n, r, cfg, det, log):
    """한 (새폴더대표, 특정 마모프레임) 쌍을 캐노니컬 포즈로. 반환 (nv, wv, axc)."""
    wa_axis = _canon_one._wa_axis            # 마모 축(캐시)
    S = int(2.4 * r)                         # 검증된 유일 안정값. >2.4r 은 수직화가 크롭의존적으로
                                             # 깨져 SAM이 릴리프 오검출(날끝 ~1.2r 이라 여기서 잘림)
    fnew = _canon_one._fnew
    ncrop, axc = bp.axis_crop(fnew, axis_n, S)
    wcrop, _ = bp.axis_crop(worn_file, wa_axis, S)
    nfill, _ = bp.ring_fill(ncrop, axc, r)
    wfill, _ = bp.ring_fill(wcrop, axc, r)
    _, ang, _ = bp.align_rotation(nfill, wfill, axc, r, log=lambda *a: None)
    wcrop_al = cv2.warpAffine(wcrop, cv2.getRotationMatrix2D((axc[0], axc[1]), ang, 1.0),
                              (S, S), flags=cv2.INTER_LINEAR)
    nv, wv = bp.verticalize(ncrop, wcrop_al, axc, r, log=lambda *a: None)
    if cfg['vertical'] == 'sam':
        nv, wv, _ = blade_align.verticalize(nv, wv, axc, r, det, log=lambda *a: None)
    else:
        nv, wv = _verticalize_watershed(nv, wv, axc, r, log)
    wv = bp.register_body(nv, wv, axc, r, log=lambda *a: None)
    return nv, wv, axc


def diagnose(new_folder, worn_folder, out, diam, name, cfg, log=print):
    name = name or os.path.basename(os.path.normpath(worn_folder))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(out, f"{stamp}_{name}")
    os.makedirs(outdir, exist_ok=True)
    nf, wf = face(new_folder), face(worn_folder)
    log(f"[기준] {nf}\n[진단] {wf}\n[변형] {cfg}\n[결과] {outdir}\\\n" + "-" * 60)

    det = partial(blade_sam.detect, glint_suppress=cfg['glint_suppress'],
                  mask_guard=cfg['mask_guard'], edge_snap=cfg.get('edge_snap', False),
                  model=cfg.get('sam', 'vit_h'))
    ref = bp.find_axis_scale(nf, diam, log=lambda *a: None)
    axis_n, r, um_od = ref['axis'], ref['od_r'], ref['um_per_px']  # um_od=OD 폴백
    wa = bp.find_axis_scale(wf, diam, n_avg=90, log=lambda *a: None)
    _canon_one._wa_axis = wa['axis']
    _canon_one._fnew = bp.pick_frame(nf, axis_n, r)   # 새 공구는 1장 고정

    # 마모 프레임: single=1장, multi=여러장
    if cfg['frames'] == 'multi':
        worn_frames = pick_frames(wf, wa['axis'], wa['od_r'], cfg.get('n_frames', 5))
    else:
        worn_frames = [bp.pick_frame(wf, wa['axis'], wa['od_r'])]

    # 각 마모 프레임마다 측정, 날별 VB/코너 수집 → 중앙값
    per = {'mask1': [], 'mask2': []}
    canon = None                                        # 대표(첫) 프레임 시각화용
    # 스케일 = 날끝 궤도 직경 기준: 직경 = 2·TIP_RATIO·OD_r 이 공구 직경(mm). OD 만으로
    # 결정돼 SAM 검출에 무관(강건). (기존 SAM 날끝-날끝은 _tip_to_tip_scale 로 남겨둠)
    um = diam * 1000.0 / (2.0 * TIP_RATIO * r)
    scale_note = f"날끝궤도({TIP_RATIO:.2f}·OD) {diam:.0f}mm 기준"
    log(f"[스케일] {um:.3f} µm/px  ({scale_note})")
    for wfr in worn_frames:
        nv, wv, axc = _canon_one(nf, wfr, axis_n, r, cfg, det, log)
        reg_n = det(nv, axc, r)
        reg_w = det(wv, axc, r)
        if canon is None:
            canon = (nv, wv, axc, reg_n, reg_w)
        for mk in ('mask1', 'mask2'):
            res = blade_wear.measure(reg_n[mk], reg_w[mk], axc, r, um, align=cfg['align'])
            if res:
                per[mk].append(res)

    nv, wv, axc, reg_n, reg_w = canon
    S = nv.shape[0]
    results = {}
    for mk, bname in (('mask1', '날1(긴 날)'), ('mask2', '날2(짧은 날)')):
        vals = per[mk]
        if not vals:
            results[bname] = None
            continue
        vb = float(np.median([v['vb_um'] for v in vals]))
        tip = float(np.median([v['tip_um'] for v in vals]))
        # 대표 프레임의 후퇴점(시각화)
        rep = blade_wear.measure(reg_n[mk], reg_w[mk], axc, r, um, align=cfg['align'])
        results[bname] = {'vb_um': vb, 'tip_um': tip, 'n': len(vals),
                          'retreat_pts': rep['retreat_pts'] if rep else []}
        log(f"  {bname}: VB={vb:.1f}µm  코너={tip:.1f}µm  (프레임 {len(vals)}장 중앙값)")

    _save(outdir, name, nv, wv, axc, r, um, reg_n, reg_w, results, cfg, scale_note)
    log("-" * 60 + f"\n완료 → {outdir}")
    return outdir


def _outline(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def _blade_crop(new, worn, mn, mw, res, S):
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
        for (px, py, umv) in res['retreat_pts']:
            t = min(1.0, umv / vmax)
            cv2.circle(vis, (int((px - x0) * 1.6), int((py - y0) * 1.6)), 3,
                       (0, int(120 * (1 - t)), int(120 + 135 * t)), -1)
    return vis


def _retreat_graph(ax, mask_new, mask_worn, axis, r, um, align, bname, side, res):
    """절삭날(곡선 쪽) 한 변의 마모 후퇴 — 파랑=새 에지(기준0) / 빨강=마모 에지.
    후퇴 = 새 외곽선 각 점에서 마모 마스크까지 수직 거리(measure() 와 동일) → 코너 포함.
    side='left'(날1) / 'right'(날2). 세로=축기준 세로위치(µm), 가로=후퇴(µm, +=마모)."""
    if align == 'tx':
        dx, dy = blade_wear._align_tx(mask_new, mask_worn, axis, r)
    else:
        dx, dy = blade_wear._align_2d(mask_new, mask_worn, axis, r)
    H, W = mask_new.shape
    worn = cv2.warpAffine(mask_worn, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H),
                          flags=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform((1 - worn).astype(np.uint8), cv2.DIST_L2, 3)
    cnts, _ = cv2.findContours(mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        ax.text(0.5, 0.5, f"{bname}\n측정불가", ha='center', va='center'); ax.axis('off'); return
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    cx, cy = float(axis[0]), float(axis[1])
    prof = {}                                          # 세로위치 y → 최대 후퇴(px)
    for x, y in c:
        if side == 'left' and x > cx + 0.02 * r:       # 절삭날 쪽 반만
            continue
        if side == 'right' and x < cx - 0.02 * r:
            continue
        if abs(y - cy) < 0.30 * r:                     # 축쪽 웹 제외
            continue
        d = float(dt[y, x]) if worn[y, x] == 0 else 0.0
        prof[y] = max(prof.get(y, 0.0), d)
    if len(prof) < 10:
        ax.text(0.5, 0.5, f"{bname}\n측정불가", ha='center', va='center'); ax.axis('off'); return
    ys = np.array(sorted(prof))
    ret = np.array([prof[y] for y in ys]) * um
    sv = (ys - cy) * um                                # 세로 위치(µm, 축 기준, 부호)
    if len(ret) >= 7:                                  # 가벼운 미디언 평활
        k = 3
        ret = np.array([np.median(ret[max(0, i - k):i + k + 1]) for i in range(len(ret))])
    BLUE, RED = '#1f5fd0', '#d62728'
    ax.plot(np.zeros_like(sv), sv, '-', color=BLUE, lw=2.0, label='새 공구')
    ax.plot(ret, sv, '-', color=RED, lw=1.6, label='마모 공구')
    ax.fill_betweenx(sv, 0, ret, color=RED, alpha=0.15)
    ax.invert_yaxis()                                  # 날끝이 바깥(날1=위/날2=아래)을 향하게
    tipv = res['tip_um'] if res else 0.0
    ax.set_title(f"{bname} 절삭날 · 후퇴 {tipv:.1f} µm", fontsize=11)
    ax.set_xlabel("마모 후퇴 (µm, 파랑=새 에지 기준 0)", fontsize=9)
    ax.set_ylabel("세로 위치 (µm, 축 기준)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower center', ncol=2)


def _save(outdir, name, nv, wv, axc, r, um, reg_n, reg_w, results, cfg, scale_note=''):
    S = nv.shape[0]
    disp = re.sub(r'_v\d+$', '', name)                 # 제목용: 'Test44_v1' → 'Test44'
    side = np.hstack([cv2.cvtColor(nv, cv2.COLOR_GRAY2BGR),
                      np.full((S, 8, 3), 255, np.uint8),
                      cv2.cvtColor(wv, cv2.COLOR_GRAY2BGR)])
    bp.imwrite_unicode(os.path.join(outdir, "1_align.png"), side)
    bp.imwrite_unicode(os.path.join(outdir, "2_sam_new.png"),
                       blade_region.overlay(nv, reg_n, axc, r))
    bp.imwrite_unicode(os.path.join(outdir, "3_sam_worn.png"),
                       blade_region.overlay(wv, reg_w, axc, r))
    crops = {}
    for mk, fn, bname in (('mask1', "4_blade1_long.png", '날1(긴 날)'),
                          ('mask2', "5_blade2_short.png", '날2(짧은 날)')):
        crop = _blade_crop(nv, wv, reg_n[mk], reg_w[mk], results.get(bname), S)
        crops[bname] = crop
        bp.imwrite_unicode(os.path.join(outdir, fn), crop)

    for fp in ('Malgun Gothic', 'AppleGothic', 'NanumGothic'):
        try:
            plt.rcParams['font.family'] = fp; break
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.15, 1.0])
    # 1행: SAM 검출
    for ax_, fn, ttl in ((fig.add_subplot(gs[0, 0]), "2_sam_new.png", "SAM 날 검출 · 새 공구"),
                         (fig.add_subplot(gs[0, 1]), "3_sam_worn.png", "SAM 날 검출 · 마모 공구")):
        ax_.imshow(cv2.cvtColor(cv2.imread(os.path.join(outdir, fn)), cv2.COLOR_BGR2RGB))
        ax_.set_title(ttl, fontsize=11); ax_.axis('off')
    # 2행: 날별 외곽선 오버레이 크롭
    for i, (bname, crop) in enumerate(crops.items()):
        ax_ = fig.add_subplot(gs[1, i])
        ax_.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = results.get(bname)
        t = f"VB={res['vb_um']:.1f}µm  코너={res['tip_um']:.1f}µm" if res else "측정불가"
        ax_.set_title(f"{bname}  ·  {t}\n(파랑=새 / 빨강=마모 / 점=후퇴)", fontsize=10)
        ax_.axis('off')
    # 3행: 날별 절삭날 후퇴 그래프 (날1=왼쪽변, 날2=오른쪽변 — 곡선 있는 절삭날)
    for i, (mk, bname, side) in enumerate((('mask1', '날1(긴 날)', 'left'),
                                           ('mask2', '날2(짧은 날)', 'right'))):
        _retreat_graph(fig.add_subplot(gs[2, i]), reg_n[mk], reg_w[mk], axc, r, um,
                       cfg['align'], bname, side, results.get(bname))
    # 우측: 요약 텍스트
    axt = fig.add_subplot(gs[:, 2]); axt.axis('off')
    lines = [f"{disp}  밑면 날 마모 진단", "",
             f"스케일 {um:.2f} µm/px", f"  ({scale_note})", ""]
    for bname, res in results.items():
        if res:
            lines += [f"{bname} (n={res.get('n', 1)})",
                      f"  VB(플랭크)  {res['vb_um']:.1f} µm",
                      f"  코너 마모   {res['tip_um']:.1f} µm", ""]
    axt.text(0.02, 0.98, "\n".join(lines), va='top', ha='left', fontsize=13, linespacing=1.6)
    fig.suptitle(f"{disp} · 밑면 날 마모 진단 (SAM 영역 검출)",
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(outdir, "summary.png"), dpi=130, facecolor='white')
    plt.close(fig)
