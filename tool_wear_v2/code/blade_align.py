# -*- coding: utf-8 -*-
"""SAM 마스크 기반 정렬·수직화 — Hough 수직화 + 워터셰드 보정 대체.

기존 문제: 자세 맞추기가 Hough(날 아닌 직선 잡음)+워터셰드(노출 취약)에 의존.
이미 SAM 이 날을 잘 잡으므로, SAM 마스크로 직접:
  - 수직화: 두 날 마스크 중심선이 세로가 되게 회전
  - 몸통 정합: 곧은 몸통변(비마모)이 겹치게 평행이동
을 실측한다.

닭-달걀(수직 포즈가 있어야 SAM 프롬프트가 맞음)은 반복으로 푼다:
  거친 정렬(align_rotation) → SAM → 수직/정합 실측·적용 → SAM 재검출 → 수렴.
"""
import cv2
import numpy as np


def _centerline_tilt(reg, axis, r):
    """SAM 두 날 마스크의 중심선 기울기[deg] (날별 RANSAC, 오프셋 상쇄 방지).
    수직이면 0. blade_region.band_tilt 와 같은 원리지만 SAM 마스크(깨끗)에 적용."""
    cy = float(axis[1])
    rng = np.random.default_rng(0)
    slopes, wts = [], []
    for name in ('mask1', 'mask2'):
        m = reg.get(name)
        if m is None:
            continue
        ys, xs = np.nonzero(m)
        if len(xs) < 200:
            continue
        d = np.abs(ys - cy)
        sel = (d >= 0.30 * r) & (d <= 0.78 * r)
        if int(sel.sum()) < 100:
            continue
        ys_s, xs_s = ys[sel], xs[sel]
        order = np.argsort(ys_s)
        ys_o, xs_o = ys_s[order], xs_s[order]
        uy, idx = np.unique(ys_o, return_index=True)
        py, px = [], []
        for i, y in enumerate(uy):
            seg = xs_o[idx[i]:(idx[i + 1] if i + 1 < len(uy) else len(xs_o))]
            py.append(float(y)); px.append(float(np.median(seg)))
        if len(py) < 60:
            continue
        py, px = np.array(py), np.array(px)
        best_in, best = 0, None
        for _ in range(200):
            i1, i2 = rng.choice(len(py), 2, replace=False)
            if abs(py[i1] - py[i2]) < 30:
                continue
            a = (px[i2] - px[i1]) / (py[i2] - py[i1])
            inl = np.abs(px - (a * py + (px[i1] - a * py[i1]))) < 3.0
            if int(inl.sum()) > best_in:
                best_in, best = int(inl.sum()), inl
        if best is None or best_in < 0.5 * len(py):
            continue
        slopes.append(float(np.polyfit(py[best], px[best], 1)[0]))
        wts.append(best_in)
    if not slopes:
        return None
    return float(np.degrees(np.arctan(np.average(slopes, weights=wts))))


def verticalize(nv, wv, axis, r, detect, log=print, iters=3):
    """SAM 마스크 중심선으로 nv·wv 를 수직화. 반환 (nv2, wv2, reg_n).
    detect: blade_sam.detect (주입, 순환 import 방지). ±부호 실측, 잔여≤0.4°까지 반복.
    (평행이동 정합은 이 뒤에 register_body 가, 남는 잔차는 측정단계가 처리 → 여기선 수직만.)"""
    S = nv.shape[0]
    cen = (float(axis[0]), float(axis[1]))
    reg_n = detect(nv, axis, r)
    total = 0.0
    for it in range(iters):
        tilt = _centerline_tilt(reg_n, axis, r)
        if tilt is None or abs(tilt) <= 0.4:
            break
        if abs(tilt) > 8.0:            # 거친 수직화 후 잔차는 작아야 정상. 큰 값=SAM 마스크
            break                      # 오류(날 못 잡음) → 신뢰 않고 중단(폭주 방지)
        best = None
        for A in (-tilt, +tilt):
            Mr = cv2.getRotationMatrix2D(cen, A, 1.0)
            probe = cv2.warpAffine(nv, Mr, (S, S), flags=cv2.INTER_LINEAR)
            t2 = _centerline_tilt(detect(probe, axis, r), axis, r)
            if t2 is not None and (best is None or abs(t2) < best[0]):
                best = (abs(t2), A, probe)
        if best is None or best[0] >= abs(tilt):
            break
        _, A, nv = best
        wv = cv2.warpAffine(wv, cv2.getRotationMatrix2D(cen, A, 1.0), (S, S),
                            flags=cv2.INTER_LINEAR)
        reg_n = detect(nv, axis, r)
        total += A
    if log and abs(total) > 1e-6:
        log(f"    [SAM 수직화] {total:+.2f}° 적용 (잔여 기울기 실측 기반)")
    return nv, wv, reg_n
