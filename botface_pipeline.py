# -*- coding: utf-8 -*-
"""
밑면(끝면) 공구 마모 측정 파이프라인 (ver3 코어)

우리가 개발/검증한 방식:
  1. 시퀀스 평균 → 바깥 원(OD) 피팅 → 회전축 + µm 스케일(공구 직경 기준)
  2. 새/마모 대표 프레임을 축 중심으로 회전 정렬 (ZNCC)
  3. 날마다 몸통 국소 정렬 (전역 프린지 제거)  [옵션]
  4. 날 몸통을 수직으로 세움
  5. 두 날(위/아래) 끝의 후퇴량 측정 → µm

한글 경로 안전(np.fromfile). GUI에서 함수로 호출한다.
"""
import os
import re
import glob
import math

import cv2
import numpy as np

FILE_PREFIX = 'img_'
FILE_EXT = '.png'


# ---------------- 한글 경로 안전 IO ----------------
def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size:
            img = cv2.imdecode(data, flags)
            if img is not None:
                return img
    except Exception:
        pass
    return cv2.imread(path, flags)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def _natkey(p):
    nums = re.findall(r'\d+', os.path.basename(p))
    return (int(nums[0]) if nums else 0, os.path.basename(p))


def list_frames(folder):
    fs = glob.glob(os.path.join(folder, f'{FILE_PREFIX}*{FILE_EXT}'))
    fs.sort(key=_natkey)
    return fs


# ---------------- 1) 축 + OD 스케일 ----------------
def average_sequence(folder, n=90, log=print):
    """시퀀스 프레임을 최대 n장 평균 → 회전하는 날은 뭉개지고 OD만 또렷."""
    frames = list_frames(folder)
    if not frames:
        return None
    step = max(1, len(frames) // n)
    sel = frames[::step][:n]
    acc = None
    cnt = 0
    for i, f in enumerate(sel):
        g = imread_unicode(f)
        if g is None:
            continue
        g = g.astype(np.float32)
        acc = g if acc is None else acc + g
        cnt += 1
        if log and (i % 20 == 0):
            log(f"    평균화 {i+1}/{len(sel)}...")
    if acc is None:
        return None
    return (acc / cnt).astype(np.uint8)


def fit_od(avg, log=print):
    """평균영상에서 바깥 원(OD): 방사형 최대경사 + 대수(Kasa) 원피팅(이상치 제거).
    반환 (cx, cy, r) [원본좌표 px]."""
    h, w = avg.shape
    b = cv2.GaussianBlur(avg, (7, 7), 0).astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    # 반경 탐색 범위: 프레임의 15~48%
    r_lo, r_hi = 0.15 * min(h, w), 0.49 * min(h, w)
    r = 0.35 * min(h, w)
    for _ in range(4):
        pts = []
        for a in np.linspace(0, 2 * np.pi, 360, endpoint=False):
            dx, dy = np.cos(a), np.sin(a)
            rs = np.arange(r_lo, r_hi, 1.0)
            xs = cx + rs * dx
            ys = cy + rs * dy
            ok = (xs >= 1) & (xs < w - 1) & (ys >= 1) & (ys < h - 1)
            rs, xs, ys = rs[ok], xs[ok], ys[ok]
            if len(rs) < 10:
                continue
            vals = b[ys.astype(int), xs.astype(int)]
            grad = np.abs(np.gradient(vals))
            k = int(np.argmax(grad))
            pts.append((xs[k], ys[k]))
        P = np.array(pts)
        if len(P) < 20:
            break
        A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
        bb = P[:, 0] ** 2 + P[:, 1] ** 2
        sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
        cx, cy = sol[0], sol[1]
        r = math.sqrt(max(1.0, sol[2] + cx * cx + cy * cy))
        # 이상치 제거 후 재적합
        d = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)
        keep = d < max(8.0, np.percentile(d, 80))
        P = P[keep]
        A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
        bb = P[:, 0] ** 2 + P[:, 1] ** 2
        sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
        cx, cy = sol[0], sol[1]
        r = math.sqrt(max(1.0, sol[2] + cx * cx + cy * cy))
        r_lo, r_hi = r - 60, r + 60
    return float(cx), float(cy), float(r)


def find_axis_scale(folder, diameter_mm, n_avg=90, log=print):
    """반환 dict: axis(cx,cy) [원본], od_r [px], um_per_px, avg 이미지."""
    avg = average_sequence(folder, n_avg, log=log)
    if avg is None:
        raise RuntimeError(f"프레임 없음: {folder}")
    cx, cy, r = fit_od(avg, log=log)
    um_per_px = diameter_mm * 1000.0 / (2.0 * r)
    return {'axis': (cx, cy), 'od_r': r, 'um_per_px': um_per_px, 'avg': avg}


# ---------------- 2) 대표 프레임 + 축 중심 크롭 ----------------
def pick_frame(folder, axis, r, log=print):
    """축 중심 원판 영역에서 대비(표준편차)가 가장 큰(선명·노출 양호) 프레임 선택."""
    frames = list_frames(folder)
    if not frames:
        return None
    cx, cy = axis
    best = (-1, frames[0])
    for f in frames[::8]:
        g = imread_unicode(f)
        if g is None:
            continue
        x0, y0 = int(cx - r * 0.7), int(cy - r * 0.7)
        x1, y1 = int(cx + r * 0.7), int(cy + r * 0.7)
        x0, y0 = max(0, x0), max(0, y0)
        patch = g[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        s = float(patch.std())
        if s > best[0]:
            best = (s, f)
    return best[1]


def axis_crop(path, axis, S):
    """축이 크롭 중앙에 오도록 S×S 크롭. 반환 (crop_gray, axis_in_crop)."""
    g = imread_unicode(path)
    cx, cy = axis
    x0 = int(round(cx - S / 2))
    y0 = int(round(cy - S / 2))
    h, w = g.shape
    canvas = np.full((S, S), int(np.median(g)), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + S), min(h, y0 + S)
    dx0, dy0 = sx0 - x0, sy0 - y0
    canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = g[sy0:sy1, sx0:sx1]
    return canvas, (cx - x0, cy - y0)


def ring_fill(crop, axis, r):
    """OD 바깥(보케 링)을 내부 평균으로 채움 (회전 대칭 → 정렬 무편향)."""
    S = crop.shape[0]
    yy, xx = np.ogrid[:S, :S]
    inside = (xx - axis[0]) ** 2 + (yy - axis[1]) ** 2 <= (r * 0.97) ** 2
    out = crop.copy()
    if inside.any():
        out[~inside] = int(crop[inside].mean())
    return out, inside


# ---------------- 3) 회전 정렬 (ZNCC) ----------------
def _zn(vec):
    v = vec.astype(np.float32)
    return (v - v.mean()) / (v.std() + 1e-6)


def _gradmag(img):
    """그라디언트 크기(에지 세기). 절대 밝기·저주파 조명얼룩에 불변."""
    f = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def align_rotation(new_crop, worn_crop, axis, r, log=print):
    """worn 을 축 중심으로 회전시켜 new 에 정렬. 반환 (aligned_worn, angle, ncc).
    옆카메라 링조명은 지향성(한쪽만 밝음)이라 밝기 그라디언트가 카메라에 고정→
    밝기로 상관하면 조명얼룩이 겹치는 각(≈0°)에 갇힘. 그래서 밝기 대신
    그라디언트 크기(에지=플루트/날 경계)로 상관 → 조명 불변 회전 정렬.
    (동축 조명이던 예전 이미지에도 그대로 유효.)"""
    S = new_crop.shape[0]
    yy, xx = np.ogrid[:S, :S]
    ins = (xx - axis[0]) ** 2 + (yy - axis[1]) ** 2 <= (r * 0.94) ** 2
    nref = _zn(_gradmag(new_crop)[ins])
    Npx = int(ins.sum())
    cen = (float(axis[0]), float(axis[1]))
    best = (-2.0, 0.0)
    for ang in np.arange(0, 360, 1.0):
        M = cv2.getRotationMatrix2D(cen, ang, 1.0)
        rot = cv2.warpAffine(worn_crop, M, (S, S), flags=cv2.INTER_LINEAR)
        c = float(np.dot(nref, _zn(_gradmag(rot)[ins])) / Npx)
        if c > best[0]:
            best = (c, ang)
    # 0.1도 미세
    base = best[1]
    for dd in np.arange(-1.0, 1.01, 0.1):
        M = cv2.getRotationMatrix2D(cen, base + dd, 1.0)
        rot = cv2.warpAffine(worn_crop, M, (S, S), flags=cv2.INTER_LINEAR)
        c = float(np.dot(nref, _zn(_gradmag(rot)[ins])) / Npx)
        if c > best[0]:
            best = (c, base + dd)
    ncc, ang = best
    aligned = cv2.warpAffine(worn_crop, cv2.getRotationMatrix2D(cen, ang, 1.0),
                             (S, S), flags=cv2.INTER_LINEAR)
    if log:
        log(f"    회전 정렬: {ang:.2f}deg  edge-NCC={ncc:.3f}")
    return aligned, ang, ncc


def register_body(new_v, worn_v, axis, r, log=print):
    """몸통(마모 안 되는 안쪽 원판·플루트·웹)을 기준으로 worn 을 new 에 정밀 정합.
    - 몸통은 아무리 절삭해도 마모되지 않는 불변 기준 → 여기서 두 사진을 정확히 겹친 뒤라야
      바깥 날끝 코너의 차이가 '순수 마모'가 된다.
    - 바깥 날끝 코너(마모부)는 마스크(안쪽 0.72r 원판)에서 제외.
    - 평행이동+미세회전(EUCLIDEAN), 조명 불변 위해 그라디언트 크기로 ECC.
    - 실패/비정상(신뢰도↓·과도한 이동)일 땐 원본 유지(안전).
    반환: 정합된 worn (실패 시 입력 그대로)."""
    S = new_v.shape[0]
    cx, cy = axis
    yy, xx = np.ogrid[:S, :S]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= (0.72 * r) ** 2).astype(np.uint8) * 255
    ng = cv2.normalize(_gradmag(new_v), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
    wg = cv2.normalize(_gradmag(worn_v), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
        cc, warp = cv2.findTransformECC(ng, wg, warp, cv2.MOTION_EUCLIDEAN, crit, mask, 5)
    except cv2.error:
        if log:
            log("    몸통 정합 실패(ECC) → 원본 유지")
        return worn_v
    dx, dy = float(warp[0, 2]), float(warp[1, 2])
    deg = math.degrees(math.atan2(warp[1, 0], warp[0, 0]))
    if cc < 0.35 or abs(dx) > 0.2 * r or abs(dy) > 0.2 * r or abs(deg) > 8:
        if log:
            log(f"    몸통 정합 신뢰도 낮음(cc={cc:.2f} dx={dx:.0f} dy={dy:.0f} {deg:+.1f}°) → 원본 유지")
        return worn_v
    reg = cv2.warpAffine(worn_v, warp, (S, S),
                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    if log:
        log(f"    몸통 정합: dx={dx:+.1f} dy={dy:+.1f} 회전{deg:+.2f}° (cc={cc:.2f})")
    return reg


def refine_blade_local(new_v, worn_v, axis, r, side, rng=8, drot=1.5, log=print):
    """전역 정합 후 남는 '그 날' 몸통의 미세 잔차(dθ,dx,dy)를 ZNCC 브루트포스로 제거.
    문서의 '날마다 몸통 국소 정렬' 단계 — 각 날을 따로, 그 날 쪽 안쪽 몸통(마모부 제외)만으로
    맞춰 전역에서 남던 프린지를 없앤다. 그라디언트 ZNCC(조명 불변). 반환: 그 날용 정렬된 worn."""
    S = new_v.shape[0]
    cx, cy = axis
    yy, xx = np.ogrid[:S, :S]
    rad2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m = (rad2 >= (0.15 * r) ** 2) & (rad2 <= (0.88 * r) ** 2)
    m &= (yy <= cy - 0.03 * r) if side == 'left' else (yy >= cy + 0.03 * r)
    if int(m.sum()) < 500:
        return worn_v
    ng = cv2.normalize(_gradmag(new_v), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
    nref = ng[m]
    nref = (nref - nref.mean()) / (nref.std() + 1e-6)
    cen = (float(cx), float(cy))
    best = (-2.0, 0, 0, 0.0)
    for th in np.arange(-drot, drot + 0.01, 0.5):
        rot = cv2.warpAffine(worn_v, cv2.getRotationMatrix2D(cen, th, 1.0), (S, S),
                             flags=cv2.INTER_LINEAR)
        rg = cv2.normalize(_gradmag(rot), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
        for dy in range(-rng, rng + 1, 2):
            for dx in range(-rng, rng + 1, 2):
                Mt = np.float32([[1, 0, dx], [0, 1, dy]])
                sh = cv2.warpAffine(rg, Mt, (S, S), flags=cv2.INTER_LINEAR)[m]
                sh = (sh - sh.mean()) / (sh.std() + 1e-6)
                z = float((nref * sh).mean())
                if z > best[0]:
                    best = (z, dx, dy, th)
    z, dx, dy, th = best
    out = cv2.warpAffine(worn_v, cv2.getRotationMatrix2D(cen, th, 1.0), (S, S),
                         flags=cv2.INTER_LINEAR)
    out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (S, S),
                         flags=cv2.INTER_LINEAR)
    if log:
        nm = '날1' if side == 'left' else '날2'
        log(f"    {nm} 국소정렬: dx={dx} dy={dy} dθ={th:+.1f}° zncc={z:.3f}")
    return out


# ---------------- 4) 날 수직화 ----------------
def blade_deviation(gray, axis, r):
    """날 몸통(중심 세로 밴드) near-vertical 직선의 수직 대비 잔차[deg]. 없으면 None."""
    S = gray.shape[0]
    b = cv2.GaussianBlur(gray, (5, 5), 0)
    e = cv2.Canny(b, 40, 120)
    Y, X = np.mgrid[0:S, 0:S]
    rad = np.hypot(X - axis[0], Y - axis[1])
    m = (np.abs(X - axis[0]) < 0.22 * r) & (rad > 0.18 * r) & (rad < 0.92 * r)
    e = (e * m).astype(np.uint8)
    lines = cv2.HoughLines(e, 1, np.pi / 1800, threshold=90)
    devs = []
    if lines is not None:
        for rho, theta in lines[:40, 0]:
            dev = math.degrees(theta)
            dev = dev - 180 if dev > 90 else dev
            if abs(dev) < 14:
                devs.append(dev)
    return (float(np.mean(devs)) if devs else None)


def verticalize(new_crop, worn_aligned, axis, r, log=print):
    """날 몸통을 수직으로. new 기준으로 회전각을 구해 둘 다 같은 각으로 회전."""
    S = new_crop.shape[0]
    cen = (float(axis[0]), float(axis[1]))
    # 우세 직선(날) 방향을 90°로: Hough로 대략각 → 보정
    b = cv2.GaussianBlur(new_crop, (5, 5), 0)
    e = cv2.Canny(b, 40, 120)
    Y, X = np.mgrid[0:S, 0:S]
    rad = np.hypot(X - axis[0], Y - axis[1])
    e = (e * ((rad < 0.9 * r) & (rad > 0.1 * r))).astype(np.uint8)
    lines = cv2.HoughLinesP(e, 1, np.pi / 720, threshold=90,
                            minLineLength=int(0.45 * r), maxLineGap=30)

    def d2l(p, a, bb):
        a = np.array(a, float); bb = np.array(bb, float); p = np.array(p, float)
        d = bb - a
        t = np.dot(p - a, d) / (np.dot(d, d) + 1e-9)
        return np.linalg.norm(p - (a + t * d))

    cand = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if d2l(cen, (x1, y1), (x2, y2)) < 0.1 * r:
                ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
                L = math.hypot(x2 - x1, y2 - y1)
                cand.append((ang, L))
    if cand:
        # 길이가중 우세각 → 수직(90)으로
        # 90°에 가까운 그룹 우선
        blade = max(cand, key=lambda c: c[1])[0]
    else:
        blade = 90.0
    delta = 90.0 - blade
    R = cv2.getRotationMatrix2D(cen, -delta, 1.0)
    nv = cv2.warpAffine(new_crop, R, (S, S), flags=cv2.INTER_LINEAR)
    wv = cv2.warpAffine(worn_aligned, R, (S, S), flags=cv2.INTER_LINEAR)
    # 잔차 정밀 보정
    dev = blade_deviation(nv, axis, r)
    if dev is not None and abs(dev) < 14:
        R2 = cv2.getRotationMatrix2D(cen, dev, 1.0)
        nv = cv2.warpAffine(nv, R2, (S, S), flags=cv2.INTER_LINEAR)
        wv = cv2.warpAffine(wv, R2, (S, S), flags=cv2.INTER_LINEAR)
    if log:
        d2 = blade_deviation(nv, axis, r)
        log(f"    수직화: 잔차 {0.0 if d2 is None else d2:+.2f}deg")
    return nv, wv


# ---------------- 5) 날 끝 측정 + 결과 이미지 ----------------
def _outer_edge_radius(gray, axis, deg, r):
    """축→deg 광선에서 절삭 코너(밝은 날 → 어둠) 최외곽 반지름[px].
    OD 근처 밴드만 탐색해 안쪽 홈/에지에 안 휘둘리게. 없으면 None."""
    th = np.radians(deg)
    dx, dy = np.cos(th), np.sin(th)
    rs = np.arange(0.80 * r, 1.08 * r, 0.5)      # OD 근처만
    xs = (axis[0] + rs * dx).astype(np.float32).reshape(-1, 1)
    ys = (axis[1] + rs * dy).astype(np.float32).reshape(-1, 1)
    v = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR).ravel()
    if v.max() < 90:                              # 이 각도엔 밝은 날 없음(여유면)
        return None
    thr = 0.5 * (float(v.min()) + float(v.max()))
    ab = np.where(v > thr)[0]
    return float(rs[ab.max()]) if len(ab) else None


def _medfilt_nan(a, k=7):
    n = len(a); out = a.copy()
    for i in range(n):
        w = a[max(0, i - k):i + k + 1]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def measure_blades(new_v, worn_v, axis, r, um_per_px, log=print):
    """수직화 후 두 날 끝(위=270°/아래=90°, y-down)의 코너 반경 프로파일 + 후퇴량.
    반환 blades=[{name, rel_ang, r_new, r_worn, recession_um, xy_new, xy_worn}]"""
    nb = cv2.GaussianBlur(new_v, (3, 3), 0)
    wb = cv2.GaussianBlur(worn_v, (3, 3), 0)
    out = []
    for nm, center_deg in [('날1', 270.0), ('날2', 90.0)]:
        angs = np.arange(center_deg - 38, center_deg + 38, 0.5)
        rn = np.array([_outer_edge_radius(nb, axis, d, r) or np.nan for d in angs], float)
        rw = np.array([_outer_edge_radius(wb, axis, d, r) or np.nan for d in angs], float)
        rn = _medfilt_nan(rn); rw = _medfilt_nan(rw)
        valid = ~np.isnan(rn) & ~np.isnan(rw)
        rec = rn - rw
        rec_um = float(np.nanmedian(rec[valid])) * um_per_px if valid.any() else 0.0
        # 그리기용 좌표 (유효 구간만)
        def xy(rr):
            th = np.radians(angs)
            return (axis[0] + rr * np.cos(th)), (axis[1] + rr * np.sin(th))
        xn = xy(rn); xw = xy(rw)
        out.append({'name': nm, 'center_deg': center_deg, 'rel_ang': angs - center_deg,
                    'r_new': rn, 'r_worn': rw, 'recession_um': rec_um,
                    'xy_new': xn, 'xy_worn': xw})
    return out


def _overlay(new_v, worn_v):
    z = np.zeros_like(new_v)
    return cv2.merge([z, worn_v, new_v])  # R=new, G=worn, 노랑=일치 (BGR: B=0,G=worn,R=new)


def _crop_tip(img_color, tip, W):
    tx, ty = tip
    h, w = img_color.shape[:2]
    x0, y0 = max(0, tx - W), max(0, ty - W)
    x1, y1 = min(w, tx + W), min(h, ty + W)
    return img_color[y0:y1, x0:x1], (x0, y0)


# ---- 날 몸통 세로 에지 추적 (시각화용 스네이크) ----
# 핵심: (1) 축 근처 좁은 밴드만 봐서 바깥 스펙큘러 반사를 배제(rough locate)
#       (2) 밝기 대신 텍스처(가로 가공줄무늬=몸통 / 매끈=플루트)로 경계 판단
#       (3) per-row 독립이 아니라 이전 행 근처만 좇는 edge-following → 매끈
def _band_of(gray, axis, r, side, hw=0.17, vspan=0.30):
    """날 끝(날1=위/날2=아래) 축 근처 밴드 좌표 (x0,y0,x1,y1)."""
    cxg, cyg = axis
    yc = cyg - r if side == 'left' else cyg + r
    x0 = max(0, int(cxg - hw * r)); x1 = min(gray.shape[1], int(cxg + hw * r))
    y0 = max(0, int(yc - vspan * r)); y1 = min(gray.shape[0], int(yc + vspan * r))
    return x0, y0, x1, y1


def _tex_band(band):
    """가로 줄무늬 에너지(세로 Sobel) → 밴드 안에서 정규화(스펙큘러 없어 오염 X)."""
    f = cv2.GaussianBlur(band, (5, 5), 0).astype(np.float32)
    sy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    T = cv2.boxFilter(np.abs(sy), -1, (7, 25))
    return cv2.normalize(T, None, 0, 255, cv2.NORM_MINMAX)


def _step_score(v, x, side, h=9):
    """x 좌/우 텍스처 차(플루트=저, 몸통=고). side='left'면 오른쪽이 몸통."""
    L = float(v[max(0, x - h):x].mean()) if x > 0 else 0.0
    R = float(v[x:x + h].mean()) if x < len(v) else 0.0
    return (R - L) if side == 'left' else (L - R)


def snake_edge(gray, axis, r, side, win=10, lam=1.3, opp=False):
    """몸통 세로 에지를 edge-following 으로 추적.
    side='left'(날1=위 밴드) / 'right'(날2=아래 밴드). win=탐색폭, lam=직선성 패널티.
    opp=False → 몸통 플루트측 에지(날1=왼쪽/날2=오른쪽, 2행용).
    opp=True  → 몸통 반대편 에지(날1=오른쪽/날2=왼쪽, 3행용). 밴드 위치는 그대로,
                스텝 방향만 뒤집어 반대 세로 에지를 잡는다.
    반환 (xs, ys) 전역좌표."""
    x0, y0, x1, y1 = _band_of(gray, axis, r, side)
    step_side = side if not opp else ('right' if side == 'left' else 'left')
    T = _tex_band(gray[y0:y1, x0:x1])
    H, Wd = T.shape
    if H < 4 or Wd < 14:
        return np.array([]), np.array([])
    mid = T[H // 3:2 * H // 3].mean(0)                 # 시드: 중앙 밴드 최고 계단
    seed = max(range(6, Wd - 6), key=lambda x: _step_score(mid, x, step_side))

    def pick(v, prev):
        lo, hi = max(6, prev - win), min(Wd - 6, prev + win)
        if hi <= lo:
            return prev
        # 텍스처 계단 - 직선성 패널티(이전 행 위치서 멀어지면 벌점 → 끝단 드리프트 억제)
        return max(range(lo, hi), key=lambda z: _step_score(v, z, step_side) - lam * abs(z - prev))

    xe = np.zeros(H, int)
    prev = seed
    for row in range(H // 2, H):                       # 시드에서 한쪽 끝으로
        prev = pick(T[row], prev); xe[row] = prev
    prev = seed
    for row in range(H // 2 - 1, -1, -1):              # 반대쪽 끝으로
        prev = pick(T[row], prev); xe[row] = prev
    xs = (x0 + xe).astype(float); ys = (y0 + np.arange(H)).astype(float)
    k = 7
    xs = np.array([np.median(xs[max(0, j - k):j + k + 1]) for j in range(H)])
    return xs, ys


def save_results(name, new_v, worn_v, axis, r, um_per_px, zncc, blades, out_dir,
                 stamp=None, log=print):
    """결과 이미지 저장 + 종합 도표. 반환 경로 dict.
    stamp: 진단 시각 문자열(YYYYMMDD_HHMMSS). 있으면 파일명 앞에 붙임."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fp in ('Malgun Gothic', 'AppleGothic', 'NanumGothic'):
        try:
            plt.rcParams['font.family'] = fp
            break
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus'] = False

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    cx, cy = int(axis[0]), int(axis[1])
    prefix = f'{stamp}_{name}' if stamp else name       # 진단 시각 포함 파일명

    # (1) 회전 정렬한 두 실제 사진 나란히 (별도 저장)
    side = np.hstack([cv2.cvtColor(new_v, cv2.COLOR_GRAY2BGR),
                      np.full((new_v.shape[0], 8, 3), 255, np.uint8),
                      cv2.cvtColor(worn_v, cv2.COLOR_GRAY2BGR)])
    p_side = os.path.join(out_dir, f'{prefix}_align.png')
    imwrite_unicode(p_side, side)
    paths['side'] = p_side

    # 몸통 플루트측 세로 에지 스네이크: 각 날마다 국소 정렬(그 날 몸통 잔차 제거) 후 측정.
    #  (날1=몸통 왼쪽 / 날2=몸통 오른쪽. 마모는 이 에지가 몸통 안쪽으로 파고든 양.)
    #  물리 보정: 마모는 재료를 제거만 하므로 마모 에지가 새 에지보다 바깥(뒤)일 수 없음.
    #  역전(음의 후퇴)은 노이즈로 보고 그 구간을 새 공구와 동일(후퇴 0)로 클램프.
    edge_side = ['left', 'right']
    snakes = []
    for i in range(len(blades)):
        eside = edge_side[i] if i < len(edge_side) else 'left'
        wref = refine_blade_local(new_v, worn_v, axis, r, eside, log=log)  # 날별 국소정렬
        nx, ny = snake_edge(new_v, axis, r, eside)
        wx, wy = snake_edge(wref, axis, r, eside)
        s_in = 1.0 if eside == 'left' else -1.0        # 몸통 안쪽(+x=오른쪽/-x=왼쪽)=마모 방향
        rec = 0.0
        if len(nx) >= 2 and len(wx) >= 2 and len(nx) == len(wx):
            wear = np.maximum(0.0, s_in * (wx - nx))    # 안쪽 방향 후퇴[px], 역전은 0 보정
            wx = nx + s_in * wear                        # 마모 에지를 새 기준 안쪽으로만 클램프
            rec = float(np.median(wear)) * um_per_px
        snakes.append({'eside': eside, 'nx': nx, 'ny': ny, 'wx': wx, 'wy': wy,
                       's_in': s_in, 'rec': rec, 'wref': wref})
        blades[i]['recession_um'] = rec                # 후퇴 숫자를 스네이크 기준으로 통일

    W = int(0.30 * r)
    tips = {0: (cx, int(cy - r)), 1: (cx, int(cy + r))}

    # === 종합 도표 ===
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.05, 1.7], hspace=0.30, wspace=0.18)

    # 1행: 정렬 사진 나란히
    axtop = fig.add_subplot(gs[0, :])
    axtop.imshow(cv2.cvtColor(side, cv2.COLOR_BGR2RGB))
    axtop.set_title("회전 정렬한 두 실제 사진  (왼쪽=새 공구 / 오른쪽=마모 공구)", fontsize=12, fontweight='bold')
    axtop.axis('off')

    # 2행: 날 끝 크롭 + 외곽선 (파랑=새, 빨강=마모). 배경=새+그 날 국소정렬 worn 합성.
    for i, bl in enumerate(blades):
        ax = fig.add_subplot(gs[1, i])
        sn = snakes[i]
        bc = cv2.cvtColor(cv2.addWeighted(new_v, 0.5, sn['wref'], 0.5, 0),
                          cv2.COLOR_GRAY2BGR)
        for (xs, ys), col in [((sn['nx'], sn['ny']), (255, 80, 0)),
                              ((sn['wx'], sn['wy']), (0, 0, 255))]:
            if len(xs) >= 2:
                cv2.polylines(bc, [np.c_[xs, ys].astype(np.int32)], False, col, 3)
        crop, _ = _crop_tip(bc, tips[i], W)
        if crop.size:
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{bl['name']} 끝  ·  후퇴 {bl['recession_um']:.1f} µm\n"
                     f"(배경=새+마모 합성 · 파랑=새 / 빨강=마모 외곽선)",
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    # 3행: 2행과 같은 몸통 에지(날1=왼쪽/날2=오른쪽)를 µm 축으로 플롯
    #  파랑(새)을 x=0 직선 기준으로 두고, 빨강(마모)은 그로부터의 가로 편차(µm).
    #  → 2행 외곽선과 동일한 선이며, 파랑↔빨강 벌어진 폭이 곧 마모(후퇴)량.
    #  x축은 두 날 공통 스케일(마모 큰 쪽 기준)로 고정 → 날1·날2 마모 크기 직접 비교.
    dw_all = []
    for sn in snakes:
        nx, wx = sn['nx'], sn['wx']
        if len(nx) >= 2 and len(wx) == len(nx):
            dw_all.append(sn['s_in'] * (wx - nx) * um_per_px)
    xmax = max((float(np.max(d)) for d in dw_all if len(d)), default=100.0)
    xmax = max(xmax, 20.0) * 1.08                    # 여유
    for i, bl in enumerate(blades):
        ax = fig.add_subplot(gs[2, i])
        sn = snakes[i]
        nx, ny, wx, wy = sn['nx'], sn['ny'], sn['wx'], sn['wy']
        if len(nx) >= 2 and len(wx) >= 2 and len(nx) == len(wx):
            y_um = (ny - axis[1]) * um_per_px            # 세로 위치(축 기준)
            dw = sn['s_in'] * (wx - nx) * um_per_px      # 새 에지 기준 마모 후퇴(+ = 몸통 안쪽)
            ax.plot(np.zeros_like(y_um), y_um, color='#1565C0', lw=2, label='새 공구')
            ax.plot(dw, y_um, color='#C62828', lw=2, label='마모 공구')
            ax.fill_betweenx(y_um, 0, dw, color='#C62828', alpha=0.18)
        ax.axvline(0, color='#1565C0', lw=0.8, alpha=0.4)
        ax.set_xlim(-0.05 * xmax, xmax)              # 두 날 공통 x 스케일
        ax.set_title(f"{bl['name']} 몸통 에지  ·  후퇴 {sn['rec']:.1f} µm", fontsize=11)
        ax.set_xlabel('마모 후퇴 (µm, 파랑=새 에지 기준 0)')
        ax.set_ylabel('세로 위치 (µm, 축 기준)')
        ax.invert_yaxis()                            # 이미지와 같은 방향(아래=+y)
        ax.set_box_aspect(2.4)                        # 좁고 세로로 긴 그래프
        ax.grid(alpha=0.3)
        ax.legend(loc='lower center', fontsize=9, ncol=2)

    fig.suptitle(f"{name}  ·  밑면 마모 진단     스케일 {um_per_px:.2f} µm/px  ·  정렬 edge-NCC {zncc:.3f}",
                 fontsize=14, fontweight='bold')
    p_graph = os.path.join(out_dir, f'{prefix}_result.png')
    fig.savefig(p_graph, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    paths['graph'] = p_graph
    if log:
        log(f"    결과 저장: {os.path.basename(p_side)}, {os.path.basename(p_graph)}")
    return paths


# ---------------- 오케스트레이션 ----------------
def process_tool(initial_folder, test_folder, diameter_mm, out_dir, name,
                 ref_axis_scale=None, stamp=None, log=print):
    """Initial(새공구) 기준으로 test_folder(마모) 밑면 진단.
    ref_axis_scale: Initial 의 find_axis_scale 결과(캐시). 없으면 계산.
    stamp: 진단 시각(YYYYMMDD_HHMMSS). 없으면 현재 시각으로 생성 → 결과 파일명에 포함.
    반환 dict(요약 + 결과 경로)."""
    if stamp is None:
        from datetime import datetime
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if ref_axis_scale is None:
        log("  [기준] 축·스케일 계산(Initial 평균)...")
        ref_axis_scale = find_axis_scale(initial_folder, diameter_mm, log=log)
    axis_n = ref_axis_scale['axis']
    r = ref_axis_scale['od_r']
    um_per_px = ref_axis_scale['um_per_px']
    log(f"  기준 축=({axis_n[0]:.0f},{axis_n[1]:.0f}) OD_r={r:.1f}px  스케일={um_per_px:.2f}µm/px")

    # 마모 공구 축(자체 시퀀스에서)
    log("  [마모] 축 계산...")
    wa = find_axis_scale(test_folder, diameter_mm, n_avg=90, log=log)
    axis_w = wa['axis']

    S = int(2.4 * r)
    # 대표 프레임
    fnew = pick_frame(initial_folder, axis_n, r, log=log)
    fworn = pick_frame(test_folder, axis_w, wa['od_r'], log=log)
    log(f"  대표 프레임: new={os.path.basename(fnew)}  worn={os.path.basename(fworn)}")

    ncrop, axc = axis_crop(fnew, axis_n, S)
    wcrop, _ = axis_crop(fworn, axis_w, S)   # 각자 축 중심으로 크롭 → 축이 동일 픽셀(S/2)
    nfill, ins = ring_fill(ncrop, axc, r)
    wfill, _ = ring_fill(wcrop, axc, r)

    aligned, ang, zncc = align_rotation(nfill, wfill, axc, r, log=log)
    # 원본(비채움) worn 도 같은 각으로 회전해 표시/측정에 사용
    wcrop_al = cv2.warpAffine(wcrop, cv2.getRotationMatrix2D((axc[0], axc[1]), ang, 1.0),
                              (S, S), flags=cv2.INTER_LINEAR)

    nv, wv = verticalize(ncrop, wcrop_al, axc, r, log=log)
    wv = register_body(nv, wv, axc, r, log=log)   # 몸통(비마모 기준) 정밀 정합 → 잔차=순수 마모
    blades = measure_blades(nv, wv, axc, r, um_per_px, log=log)

    # 옛 dist 방식: 진단마다 {시각}_{이름} 하위폴더로 묶음
    # (save_results 안에서 후퇴 숫자를 스네이크 외곽선 기준으로 통일해 덮어씀)
    out_sub = os.path.join(out_dir, f'{stamp}_{name}')
    paths = save_results(name, nv, wv, axc, r, um_per_px, zncc, blades, out_sub,
                         log=log)
    for bl in blades:
        log(f"    {bl['name']} 후퇴 ≈ {bl['recession_um']:.1f} µm")

    return {
        'name': name,
        'um_per_px': um_per_px,
        'zncc': zncc,
        'rotation': ang,
        'blades': [{'name': b['name'], 'recession_um': b['recession_um']} for b in blades],
        'paths': paths,
        'ref_axis_scale': ref_axis_scale,
    }
