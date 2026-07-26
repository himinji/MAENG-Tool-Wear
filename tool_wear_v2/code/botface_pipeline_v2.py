# -*- coding: utf-8 -*-
"""
밑면(끝면) 공구 마모 측정 파이프라인 v2
= botface_pipeline.py(ver3 코어, 원본 무수정 유지) + bottom_align.py 좋은 기능 역이식.

원본 대비 추가/변경 (2026-07-20):
  - measure_blades/process_tool: n_blades 파라미터 (N날 일반화, 기본 2 = 기존 동작)
    · N≠2 는 스네이크 대신 방사형 측정·도표. 후퇴는 물리 클램프(≥0) 통일.
  - verticalize: blade_hint(날 방향 힌트, 원형 길이가중 평균) +
    조/미세 회전 부호를 가정 없이 실측(blade_deviation)으로 결정
  - align_rotation: 미세 탐색 2단계 ±1.0°@0.2° → ±0.2°@0.05°
  - save_results: 날 끝 크롭에 50µm 스케일바, N날 도표 일반화

파이프라인(원본과 동일):
  1. 시퀀스 평균 → 바깥 원(OD) 피팅 → 회전축 + µm 스케일(공구 직경 기준)
  2. 새/마모 대표 프레임을 축 중심으로 회전 정렬 (ZNCC)
  3. 날마다 몸통 국소 정렬 (전역 프린지 제거)  [옵션]
  4. 날 몸통을 수직으로 세움
  5. 각 날 끝의 후퇴량 측정 → µm

한글 경로 안전(np.fromfile). GUI에서 함수로 호출한다.
(GUI 전환은 monitor_gui_v5.py 의 `import botface_pipeline as bp` 를
 `import botface_pipeline_v2 as bp` 로 바꾸면 됨 — 기본값이 같아 동작 동일.)
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


def _accum_od(avg):
    """방사 에지 누적기(원형 일치도)로 OD 교차검증용 (cx,cy,r).
    OD 는 원 → 모든 각도에서 같은 반지름에 에지가 모여 강하게 누적된다.
    비원형 내부무늬(저대비 세션에서 fit_od 를 오도하는 것)는 흩어져 약하다.
    중심을 이미지 중앙 ±0.08 격자로, 반지름을 0.15~0.49로 훑어 평균 |grad| 최대점."""
    h, w = avg.shape
    b = cv2.GaussianBlur(avg, (7, 7), 0).astype(np.float32)
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    gm = cv2.magnitude(gx, gy)
    mn = min(h, w)
    rs = np.arange(int(0.15 * mn), int(0.49 * mn), 2.0)
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    step = max(1, int(0.02 * mn))
    best = (-1.0, w / 2.0, h / 2.0, float(rs[0]))
    for cy in range(int(h / 2 - 0.08 * mn), int(h / 2 + 0.08 * mn) + 1, step):
        for cx in range(int(w / 2 - 0.08 * mn), int(w / 2 + 0.08 * mn) + 1, step):
            for rr in rs:
                xs = (cx + rr * ca).astype(np.int32)
                ys = (cy + rr * sa).astype(np.int32)
                ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                if int(ok.sum()) < 300:
                    continue
                sc = float(gm[ys[ok], xs[ok]].mean())
                if sc > best[0]:
                    best = (sc, float(cx), float(cy), float(rr))
    return best[1], best[2], best[3]


def find_axis_scale(folder, diameter_mm, n_avg=90, log=print):
    """반환 dict: axis(cx,cy) [원본], od_r [px], um_per_px, avg 이미지."""
    avg = average_sequence(folder, n_avg, log=log)
    if avg is None:
        raise RuntimeError(f"프레임 없음: {folder}")
    # 저대비 세션(무코팅 등)에서도 OD 경계가 서도록 대비 향상(CLAHE) 후 검출.
    # 고대비 세션은 결과 거의 불변(0709 649→650). OD 검출 전용 — 반환 avg 는 원본.
    avg_od = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(avg)
    cx, cy, r = fit_od(avg_od, log=log)
    # 강건성 교차검증: fit_od 가 내부무늬에 갇혀 어긋나면 원형 일치도 누적기로 대체.
    # 25% 이내 일치하면 fit_od 유지(대비향상 후 거의 항상 일치).
    acx, acy, ar = _accum_od(avg_od)
    if ar > 0 and abs(r - ar) / ar > 0.25:
        if log:
            log(f"    [OD] fit_od r={r:.0f} 이상치 → 누적기 r={ar:.0f} 채택")
        cx, cy, r = acx, acy, ar
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
    Npx = int(ins.sum())
    cen = (float(axis[0]), float(axis[1]))
    # 점수 2종 결합 (Test44 실측으로 확정):
    #  - gradmag(질감): 새↔저마모 쌍에서 정밀. 단 심마모는 질감이 바뀌어 피크 뭉개짐(0.39)
    #  - 강블러 밝기(거친 명암 구조): 마모돼도 유지되는 플루트/면 패턴 → 심마모에서도
    #    깨끗한 피크(0.95). 둘을 코스 최대값으로 정규화해 합산.
    kb = int(0.10 * r) | 1
    nref = _zn(_gradmag(new_crop)[ins])
    bref = _zn(cv2.GaussianBlur(new_crop.astype(np.float32), (kb, kb), 0)[ins])

    def scores(rot):
        cg = float(np.dot(nref, _zn(_gradmag(rot)[ins])) / Npx)
        rb = cv2.GaussianBlur(rot.astype(np.float32), (kb, kb), 0)
        cb = float(np.dot(bref, _zn(rb[ins])) / Npx)
        return cg, cb

    coarse = np.arange(0, 360, 1.0)
    sg = np.empty(len(coarse))
    sb = np.empty(len(coarse))
    for i, ang in enumerate(coarse):
        M = cv2.getRotationMatrix2D(cen, float(ang), 1.0)
        rot = cv2.warpAffine(worn_crop, M, (S, S), flags=cv2.INTER_LINEAR)
        sg[i], sb[i] = scores(rot)
    ng, nb2 = max(1e-6, float(sg.max())), max(1e-6, float(sb.max()))

    def comb(cg, cb):
        return 0.5 * cg / ng + 0.5 * cb / nb2

    ci = int(np.argmax(0.5 * sg / ng + 0.5 * sb / nb2))
    best = (comb(sg[ci], sb[ci]), float(coarse[ci]), float(sg[ci]), float(sb[ci]))
    # 미세 탐색 2단계: ±1.0° @0.2° → ±0.2° @0.05° (bottom_align 역이식)
    for span, step in ((1.0, 0.2), (0.2, 0.05)):
        base = best[1]
        for dd in np.arange(-span, span + 1e-9, step):
            M = cv2.getRotationMatrix2D(cen, base + dd, 1.0)
            rot = cv2.warpAffine(worn_crop, M, (S, S), flags=cv2.INTER_LINEAR)
            cg, cb = scores(rot)
            c = comb(cg, cb)
            if c > best[0]:
                best = (c, base + dd, cg, cb)
    _, ang, ncc_g, ncc_b = best
    aligned = cv2.warpAffine(worn_crop, cv2.getRotationMatrix2D(cen, ang, 1.0),
                             (S, S), flags=cv2.INTER_LINEAR)
    if log:
        log(f"    회전 정렬: {ang:.2f}deg  edge-NCC={ncc_g:.3f}  구조-NCC={ncc_b:.3f}")
    # 반환 ncc = 정렬 신뢰도 지표. 심마모에서 edge-NCC 는 정렬이 맞아도 낮게 나오므로
    # 둘 중 큰 값을 신뢰도로 보고한다.
    return aligned, ang, max(ncc_g, ncc_b)


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
    if side == 'left':
        m &= (yy <= cy - 0.03 * r)
    elif side == 'right':
        m &= (yy >= cy + 0.03 * r)
    else:                                          # 숫자 = 날 방향[deg] → ±55° 섹터 (N날 일반화)
        ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0
        da = (ang - float(side) + 180.0) % 360.0 - 180.0
        m &= np.abs(da) <= 55.0
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
        nm = {'left': '날1', 'right': '날2'}.get(side)
        if nm is None:                             # 숫자 side (N날 일반화)
            nm = f'날@{float(side):.0f}°'
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


def verticalize(new_crop, worn_aligned, axis, r, blade_hint=None, log=print):
    """날 몸통을 수직으로. new 기준으로 회전각을 구해 둘 다 같은 각으로 회전.
    blade_hint: 날 방향 힌트[deg, mod 180]. Hough 가 엉뚱한 직선을 잡을 때 지정
    (bottom_align --blade-hint 역이식)."""
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
    if cand and blade_hint is not None:
        # 힌트 ±18° 내 직선만 길이가중 평균.
        # 산술평균은 0/180 경계에서 파괴됨(2°와 178°의 평균=90°) → 2θ 벡터 평균(원형).
        tgt = blade_hint % 180.0
        sel = [(a, L) for a, L in cand
               if min(abs(a - tgt), 180 - abs(a - tgt)) < 18]
        if sel:
            sv = sum(L * math.sin(math.radians(2 * a)) for a, L in sel)
            cv = sum(L * math.cos(math.radians(2 * a)) for a, L in sel)
            blade = (math.degrees(math.atan2(sv, cv)) / 2.0) % 180.0
        else:
            blade = tgt
    elif cand:
        # 길이가중 우세각 → 수직(90)으로
        blade = max(cand, key=lambda c: c[1])[0]
    else:
        blade = 90.0 if blade_hint is None else blade_hint % 180.0
    delta = 90.0 - blade
    # 조(coarse) 회전 부호도 실측으로 결정 (bottom_align make_vertical 역이식):
    # Hough mod-180 모호성으로 -delta 가 틀리면 아래 잔차 게이트(<14°)로는 못 살림.
    probes = {}
    best = None                                    # (|잔차|, 부호)
    for sgn in (-1.0, +1.0):
        Rc = cv2.getRotationMatrix2D(cen, sgn * delta, 1.0)
        probes[sgn] = cv2.warpAffine(new_crop, Rc, (S, S), flags=cv2.INTER_LINEAR)
        d = blade_deviation(probes[sgn], axis, r)
        if d is not None and (best is None or abs(d) < best[0]):
            best = (abs(d), sgn)
    sgn = best[1] if best is not None else -1.0    # 실측 불가 시 기존 관례(-delta)
    nv = probes[sgn]
    wv = cv2.warpAffine(worn_aligned, cv2.getRotationMatrix2D(cen, sgn * delta, 1.0),
                        (S, S), flags=cv2.INTER_LINEAR)
    # 잔차 정밀 보정 — 보정 부호를 가정하지 않고 양방향 실측해 좋은 쪽 채택
    # (bottom_align 역이식: 잔차가 실제로 줄어들 때만 적용)
    dev = blade_deviation(nv, axis, r)
    if dev is not None and abs(dev) < 14:
        pick = (abs(dev), None)                    # (잔차, 채택 보정각)
        for sgn in (+1.0, -1.0):
            R2 = cv2.getRotationMatrix2D(cen, sgn * dev, 1.0)
            probe = cv2.warpAffine(nv, R2, (S, S), flags=cv2.INTER_LINEAR)
            d2 = blade_deviation(probe, axis, r)
            if d2 is not None and abs(d2) < pick[0]:
                pick = (abs(d2), sgn * dev)
        if pick[1] is not None:
            R2 = cv2.getRotationMatrix2D(cen, pick[1], 1.0)
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


def measure_blades(new_v, worn_v, axis, r, um_per_px, n_blades=2, sectors=None,
                   log=print):
    """수직화 후 각 날 끝(날1=12시=270°, 이후 360/N° 간격, y-down)의
    코너 반경 프로파일 + 후퇴량 (bottom_align --blades N 역이식, 기본 2날).
    sectors: blade_region.bounds() 결과 — 날별 실제 날 영역의 각도 한계.
      지정되면 고정 ±hw 대신 그 범위만 측정(날 밖 배경 오검출 방지).
    반환 blades=[{name, center_deg, rel_ang, r_new, r_worn, recession_um, xy_new, xy_worn}]"""
    nb = cv2.GaussianBlur(new_v, (3, 3), 0)
    wb = cv2.GaussianBlur(worn_v, (3, 3), 0)
    out = []
    blade_dirs = [(f'날{k + 1}', (270.0 + k * 360.0 / n_blades) % 360.0)
                  for k in range(n_blades)]
    hw = min(38.0, 0.8 * 180.0 / n_blades)         # 섹터 반각(이웃 날과 겹침 방지)
    for k, (nm, center_deg) in enumerate(blade_dirs):
        if sectors is not None and k < len(sectors) and sectors[k]:
            lo, hi = sectors[k]['sector']
            angs = np.arange(lo, hi, 0.5)
        else:
            angs = np.arange(center_deg - hw, center_deg + hw, 0.5)
        rn = np.array([_outer_edge_radius(nb, axis, d, r) or np.nan for d in angs], float)
        rw = np.array([_outer_edge_radius(wb, axis, d, r) or np.nan for d in angs], float)
        rn = _medfilt_nan(rn); rw = _medfilt_nan(rw)
        valid = ~np.isnan(rn) & ~np.isnan(rw)
        rec = rn - rw
        # 물리 보정: 마모는 재료 제거만 → 후퇴 < 0 은 노이즈 (스네이크 경로와 동일 불변식)
        rec_um = max(0.0, float(np.nanmedian(rec[valid])) * um_per_px) if valid.any() else 0.0
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
                 stamp=None, row_limits=None, log=print):
    """결과 이미지 저장 + 종합 도표. 반환 경로 dict.
    stamp: 진단 시각 문자열(YYYYMMDD_HHMMSS). 있으면 파일명 앞에 붙임.
    row_limits: 날별 (ymin,ymax) — 인식된 날 영역의 세로 범위. 지정되면 스네이크
      외곽선·후퇴 계산을 그 안으로 한정(날 끝 밖 배경 행 제외)."""
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
    #  스네이크는 "세로 밴드 좌/우" 전제라 2날 전용. N≠2 날이면 건너뛰고
    #  measure_blades 의 방사형 후퇴·외곽선을 그대로 사용한다.
    snakes = []
    if len(blades) == 2:
        edge_side = ['left', 'right']
        for i in range(len(blades)):
            eside = edge_side[i]
            wref = refine_blade_local(new_v, worn_v, axis, r, eside, log=log)  # 날별 국소정렬
            nx, ny = snake_edge(new_v, axis, r, eside)
            wx, wy = snake_edge(wref, axis, r, eside)
            # 인식된 날 영역 밖(배경/보케 행) 제외 — 두 스네이크는 같은 밴드 행을
            # 공유하므로 같은 마스크로 걸러도 짝이 유지된다
            if (row_limits is not None and i < len(row_limits) and row_limits[i]
                    and len(ny) >= 2 and len(wy) == len(ny)):
                y0r, y1r = row_limits[i]
                keep = (ny >= y0r) & (ny <= y1r)
                if int(keep.sum()) >= 2:
                    nx, ny = nx[keep], ny[keep]
                    wx, wy = wx[keep], wy[keep]
            s_in = 1.0 if eside == 'left' else -1.0    # 몸통 안쪽(+x=오른쪽/-x=왼쪽)=마모 방향
            rec = 0.0
            if len(nx) >= 2 and len(wx) >= 2 and len(nx) == len(wx):
                wear = np.maximum(0.0, s_in * (wx - nx))  # 안쪽 방향 후퇴[px], 역전은 0 보정
                wx = nx + s_in * wear                     # 마모 에지를 새 기준 안쪽으로만 클램프
                rec = float(np.median(wear)) * um_per_px
            snakes.append({'eside': eside, 'nx': nx, 'ny': ny, 'wx': wx, 'wy': wy,
                           's_in': s_in, 'rec': rec, 'wref': wref})
            blades[i]['recession_um'] = rec            # 후퇴 숫자를 스네이크 기준으로 통일

    W = int(0.30 * r)
    tips = {}
    for i, bl in enumerate(blades):                    # 날 끝 = 그 날 방향의 OD 지점
        th = math.radians(bl['center_deg'])            # measure_blades 가 항상 채움
        # round 필수: int() 절삭은 cos(270°)=-1e-16 같은 미세 음수로 1px 어긋남
        tips[i] = (int(round(cx + r * math.cos(th))), int(round(cy + r * math.sin(th))))

    # === 종합 도표 ===
    ncols = max(2, len(blades))
    fig = plt.figure(figsize=(6.5 * ncols, 14))
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.15, 1.05, 1.7], hspace=0.30, wspace=0.18)

    # 1행: 정렬 사진 나란히
    axtop = fig.add_subplot(gs[0, :])
    axtop.imshow(cv2.cvtColor(side, cv2.COLOR_BGR2RGB))
    axtop.set_title("회전 정렬한 두 실제 사진  (왼쪽=새 공구 / 오른쪽=마모 공구)", fontsize=12, fontweight='bold')
    axtop.axis('off')

    # 2행: 날 끝 크롭 + 외곽선 (파랑=새, 빨강=마모).
    #  2날: 배경=새+그 날 국소정렬 worn 합성 + 스네이크 외곽선
    #  N날: 배경=새+마모 합성 + 방사형 외곽선(measure_blades)
    def _poly(xs, ys):
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        ok = ~(np.isnan(xs) | np.isnan(ys))
        return np.c_[xs[ok], ys[ok]].astype(np.int32)

    for i, bl in enumerate(blades):
        ax = fig.add_subplot(gs[1, i])
        if snakes:
            sn = snakes[i]
            bc = cv2.cvtColor(cv2.addWeighted(new_v, 0.5, sn['wref'], 0.5, 0),
                              cv2.COLOR_GRAY2BGR)
            curves = [((sn['nx'], sn['ny']), (255, 80, 0)),
                      ((sn['wx'], sn['wy']), (0, 0, 255))]
        else:
            bc = cv2.cvtColor(cv2.addWeighted(new_v, 0.5, worn_v, 0.5, 0),
                              cv2.COLOR_GRAY2BGR)
            curves = [(bl['xy_new'], (255, 80, 0)), (bl['xy_worn'], (0, 0, 255))]
        for (xs, ys), col in curves:
            pts = _poly(xs, ys)
            if len(pts) >= 2:
                cv2.polylines(bc, [pts], False, col, 3)
        crop, _ = _crop_tip(bc, tips[i], W)
        if crop.size:
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            # 50µm 스케일바 (bottom_align 역이식). 고배율에서 바가 크롭보다 길면 생략,
            # plot 이 imshow 축을 autoscale 로 늘리지 않게 축 범위를 이미지에 고정.
            bar = 50.0 / um_per_px
            ch, cw2 = crop.shape[:2]
            if bar <= 0.9 * cw2:
                bx, by = 0.06 * cw2, 0.94 * ch
                ax.plot([bx, bx + bar], [by, by], color='white', lw=3)
                ax.text(bx, by - 0.03 * ch, '50 µm', color='white',
                        fontsize=9, fontweight='bold')
            ax.set_xlim(-0.5, cw2 - 0.5)
            ax.set_ylim(ch - 0.5, -0.5)
        ax.set_title(f"{bl['name']} 끝  ·  후퇴 {bl['recession_um']:.1f} µm\n"
                     f"(배경=새+마모 합성 · 파랑=새 / 빨강=마모 외곽선)",
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    # 3행: 2행과 같은 몸통 에지(날1=왼쪽/날2=오른쪽)를 µm 축으로 플롯
    #  파랑(새)을 x=0 직선 기준으로 두고, 빨강(마모)은 그로부터의 가로 편차(µm).
    #  → 2행 외곽선과 동일한 선이며, 파랑↔빨강 벌어진 폭이 곧 마모(후퇴)량.
    #  x축은 두 날 공통 스케일(마모 큰 쪽 기준)로 고정 → 날1·날2 마모 크기 직접 비교.
    if snakes:
        dw_all = []
        for sn in snakes:
            nx, wx = sn['nx'], sn['wx']
            if len(nx) >= 2 and len(wx) == len(nx):
                dw_all.append(sn['s_in'] * (wx - nx) * um_per_px)
        xmax = max((float(np.max(d)) for d in dw_all if len(d)), default=100.0)
        xmax = max(xmax, 20.0) * 1.08                # 여유
        for i, bl in enumerate(blades):
            ax = fig.add_subplot(gs[2, i])
            sn = snakes[i]
            nx, ny, wx, wy = sn['nx'], sn['ny'], sn['wx'], sn['wy']
            if len(nx) >= 2 and len(wx) >= 2 and len(nx) == len(wx):
                y_um = (ny - axis[1]) * um_per_px        # 세로 위치(축 기준)
                dw = sn['s_in'] * (wx - nx) * um_per_px  # 새 에지 기준 마모 후퇴(+ = 몸통 안쪽)
                ax.plot(np.zeros_like(y_um), y_um, color='#1565C0', lw=2, label='새 공구')
                ax.plot(dw, y_um, color='#C62828', lw=2, label='마모 공구')
                ax.fill_betweenx(y_um, 0, dw, color='#C62828', alpha=0.18)
            ax.axvline(0, color='#1565C0', lw=0.8, alpha=0.4)
            ax.set_xlim(-0.05 * xmax, xmax)          # 두 날 공통 x 스케일
            ax.set_title(f"{bl['name']} 몸통 에지  ·  후퇴 {sn['rec']:.1f} µm", fontsize=11)
            ax.set_xlabel('마모 후퇴 (µm, 파랑=새 에지 기준 0)')
            ax.set_ylabel('세로 위치 (µm, 축 기준)')
            ax.invert_yaxis()                        # 이미지와 같은 방향(아래=+y)
            ax.set_box_aspect(2.4)                    # 좁고 세로로 긴 그래프
            ax.grid(alpha=0.3)
            ax.legend(loc='lower center', fontsize=9, ncol=2)
    else:
        # N≠2 날: 방사형 후퇴 프로파일(날 끝 기준 각도별) — 모든 날 공통 y 스케일
        rec_all = [(bl['r_new'] - bl['r_worn']) * um_per_px for bl in blades]
        ymax = max((float(np.nanmax(d)) for d in rec_all
                    if np.isfinite(d).any()), default=100.0)
        ymax = max(ymax, 20.0) * 1.08
        for i, bl in enumerate(blades):
            ax = fig.add_subplot(gs[2, i])
            dw = rec_all[i]
            ok = np.isfinite(dw)
            if ok.any():
                ax.plot(bl['rel_ang'][ok], dw[ok], color='#C62828', lw=2)
                ax.fill_between(bl['rel_ang'][ok], 0, dw[ok], color='#C62828', alpha=0.18)
            ax.axhline(0, color='#1565C0', lw=0.8, alpha=0.4)
            ax.set_ylim(-0.05 * ymax, ymax)
            ax.set_title(f"{bl['name']} 코너 후퇴  ·  중앙값 {bl['recession_um']:.1f} µm",
                         fontsize=11)
            ax.set_xlabel('날 끝 기준 각도 (deg)')
            ax.set_ylabel('후퇴 (µm)')
            ax.grid(alpha=0.3)

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
                 ref_axis_scale=None, stamp=None, n_blades=2, blade_hint=None,
                 use_region=True, log=print):
    """Initial(새공구) 기준으로 test_folder(마모) 밑면 진단.
    ref_axis_scale: Initial 의 find_axis_scale 결과(캐시). 없으면 계산.
    stamp: 진단 시각(YYYYMMDD_HHMMSS). 없으면 현재 시각으로 생성 → 결과 파일명에 포함.
    n_blades: 날 개수(bottom_align 역이식). 2가 아니면 스네이크 대신 방사형 측정 사용.
    blade_hint: 수직화용 날 방향 힌트[deg, mod 180].
    use_region: 날 영역 인식(blade_region)으로 측정 범위를 날 안으로 한정.
      ANTHROPIC_API_KEY 가 있으면 인식 결과를 LLM(Claude)이 추가 검증. 2날 전용.
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

    nv, wv = verticalize(ncrop, wcrop_al, axc, r, blade_hint=blade_hint, log=log)
    wv = register_body(nv, wv, axc, r, log=log)   # 몸통(비마모 기준) 정밀 정합 → 잔차=순수 마모

    # 옛 dist 방식: 진단마다 {시각}_{이름} 하위폴더로 묶음
    # (save_results 안에서 후퇴 숫자를 스네이크 외곽선 기준으로 통일해 덮어씀)
    out_sub = os.path.join(out_dir, f'{stamp}_{name}')

    # ---- 날 영역 인식: 측정을 실제 날 안으로 한정 (날 밖 배경 오검출 방지) ----
    sectors = row_limits = None
    p_region = None
    if use_region and n_blades == 2:
        try:
            import blade_region
            reg = blade_region.recognize(nv, axc, r)
            # 날 기준 수직 보정: 인식된 날별 중심선(RANSAC)으로 기울기를 실측하고,
            # 보정 부호는 ± 둘 다 적용해 잔여가 작아지는 쪽을 채택(실측 프로브).
            # 잔여 ≤0.5° 까지 최대 3회 반복. 개선 안 되면 자동 중단(안전).
            if reg is not None:
                tilt = blade_region.band_tilt(reg, axc, r)
                total_corr = 0.0
                for _ in range(3):
                    if tilt is None or abs(tilt) <= 0.5:
                        break
                    Sv = nv.shape[0]
                    cands = []
                    for A in (-tilt, +tilt):
                        Mrot = cv2.getRotationMatrix2D((axc[0], axc[1]), A, 1.0)
                        g2 = cv2.warpAffine(nv, Mrot, (Sv, Sv), flags=cv2.INTER_LINEAR)
                        r2 = blade_region.recognize(g2, axc, r)
                        t2 = blade_region.band_tilt(r2, axc, r) if r2 else None
                        if t2 is not None:
                            cands.append((abs(t2), A, g2, r2, t2))
                    if not cands:
                        break
                    cands.sort(key=lambda c: c[0])
                    if cands[0][0] >= abs(tilt):        # 개선 없음 → 중단
                        break
                    _, A, nv, reg, tilt = cands[0]
                    Mrot = cv2.getRotationMatrix2D((axc[0], axc[1]), A, 1.0)
                    wv = cv2.warpAffine(wv, Mrot, (Sv, Sv), flags=cv2.INTER_LINEAR)
                    total_corr += A
                if abs(total_corr) > 1e-6:
                    log(f"  [영역] 날 기준 수직 보정 {total_corr:+.2f}° 적용 (잔여 {tilt:+.2f}°)")
            # 수직(캐노니컬) 포즈 확정 후 최종 마스크 검출. 우선순위:
            #  1) SAM(MobileSAM) — 물체를 실제 인식해 잘라냄. 밝기 경계에 의존하는
            #     워터셰드와 달리 조명 밝든 어둡든 날 밴드를 정확·매끈하게 검출.
            #  2) Claude 주석(blade_annotations.json) — SAM 미설치 시
            #  3) 워터셰드(위에서 이미 계산된 reg) — 최종 폴백
            sam_ok = False
            try:
                import blade_sam
                reg_s = blade_sam.detect(nv, axc, r)
                if reg_s is not None:
                    reg = reg_s
                    sam_ok = True
                    log("  [영역] SAM(MobileSAM) 기반 검출 사용")
            except Exception as e:
                log(f"  [영역] SAM 사용 불가({type(e).__name__}) → 폴백 검출 사용")
            if not sam_ok:
                anno = blade_region.find_annotation(initial_folder)
                if anno is not None:
                    reg_a = blade_region.recognize_annotated(nv, anno)
                    if reg_a is not None:
                        reg = reg_a
                        log("  [영역] Claude 주석 기반 검출 사용 (blade_annotations.json)")
            if reg is None:
                log("  [영역] 날 영역 인식 실패 → 기본 측정 범위 사용")
            else:
                bnd = blade_region.bounds(reg, axc, r)
                ov = blade_region.overlay(nv, reg, axc, r)
                os.makedirs(out_sub, exist_ok=True)
                p_region = os.path.join(out_sub, f'{stamp}_{name}_region.png')
                imwrite_unicode(p_region, ov)
                verdict = blade_region.llm_verify(ov, cache_key=fnew, log=log)
                if verdict is not None and not verdict.get('ok', True):
                    log("  [영역] LLM 검증에서 문제 지적 → 측정 한정 없이 기본 범위 사용")
                else:
                    sectors = bnd
                    row_limits = [b['rows'] if b else None for b in bnd]
                    for i, b in enumerate(bnd):
                        if b:
                            lo, hi = b['sector']
                            log(f"  [영역] 날{i + 1} 측정 섹터 {lo:.1f}°~{hi:.1f}°"
                                f"  세로 {b['rows'][0]}~{b['rows'][1]}px (날 안으로 한정)")
        except Exception:
            log("  [영역] 인식 단계 오류 → 기본 측정 범위 사용")

    blades = measure_blades(nv, wv, axc, r, um_per_px, n_blades=n_blades,
                            sectors=sectors, log=log)
    # 방사형(코너) 후퇴는 이후 save_results 에서 스네이크(몸통 곧은 변) 기준으로
    # 덮어써지므로, 코너 마모 참고치를 여기서 로그로 남긴다 (곧은 변은 형상이라
    # 마모돼도 0 이 정상 — 실제 마모는 코너에 나타남)
    for b in blades:
        log(f"    {b['name']} 코너 후퇴(방사형 참고치) ≈ {b['recession_um']:.1f} µm")

    paths = save_results(name, nv, wv, axc, r, um_per_px, zncc, blades, out_sub,
                         row_limits=row_limits, log=log)
    if p_region:
        paths['region'] = p_region
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
