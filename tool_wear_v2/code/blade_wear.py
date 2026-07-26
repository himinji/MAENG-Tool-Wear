# -*- coding: utf-8 -*-
"""SAM 마스크 외곽선 차이로 마모량(VB) 측정.

원리 (사용자 지시: "검출만 제대로면 외곽선은 쉽다"):
  수직 캐노니컬 포즈에서 날은 세로 밴드. 각 행 y 에서 새/마모 마스크의
  좌·우 에지 x 를 잰다.
    좌 이동 L(y) = worn_left(y)  - new_left(y)
    우 이동 R(y) = worn_right(y) - new_right(y)
  정렬 잔차(전체 평행이동)는 두 변을 같은 방향으로 밀고(공통성분),
  마모(한 변이 안쪽으로 후퇴)는 두 변을 다르게 만든다(차등성분).
    공통(정렬) = (L+R)/2   ← 빼서 제거
    잔차 후퇴 = 공통 제거 후 각 변의 안쪽 이동량 = 순수 마모
  VB = 마모 변에서 안쪽 후퇴의 강건 대표값(상위 백분위) × µm/px.
"""
import cv2
import numpy as np


def _medfilt(a, k=21):
    n = len(a)
    return np.array([np.median(a[max(0, i - k):i + k + 1]) for i in range(n)])


def _edges(mask, axis, r):
    """행별 (y, left_x, right_x). 몸통~코너 존(0.30~1.05r)만.
    SAM 마스크 경계는 픽셀단위로 들쭉날쭉 → 강한 미디언 평활로 경계 노이즈 제거
    (실제 마모는 수십 행에 걸쳐 매끈, 노이즈는 행마다 튐 → 평활로 분리)."""
    cy = float(axis[1])
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    rows = {}
    for y, x in zip(ys, xs):
        d = abs(y - cy)
        if 0.30 * r <= d <= 1.05 * r:
            lo, hi = rows.get(y, (x, x))
            rows[y] = (min(lo, x), max(hi, x))
    if len(rows) < 40:
        return None
    yy = np.array(sorted(rows))
    L = _medfilt(np.array([rows[y][0] for y in yy], float))
    R = _medfilt(np.array([rows[y][1] for y in yy], float))
    return yy, L, R


def _align_2d(mask_new, mask_worn, axis, r, span=25):
    """마모 마스크를 새 마스크에 2D 평행이동으로 정렬(비마모 몸통대 IoU 최대).
    ★ 몸통대(0.30~0.75r)에만 한정: 마모가 몰리는 코너(0.90r+)를 정렬에 포함하면
    정렬이 마모를 흡수해 과소평가 → 형상(비마모) 구간으로만 정렬을 정한다.
    반환: (dx, dy) [worn 에 적용할 평행이동]."""
    H, W = mask_new.shape
    cx, cy = float(axis[0]), float(axis[1])
    yy, xx = np.ogrid[:H, :W]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    body = (d2 >= (0.30 * r) ** 2) & (d2 <= (0.75 * r) ** 2)
    Nfull = (mask_new & body)
    Wfull = (mask_worn & body)
    ys, xs = np.nonzero(Nfull | Wfull)
    if len(xs) < 100:
        return 0.0, 0.0
    x0, x1 = max(0, xs.min() - span - 2), min(W, xs.max() + span + 2)
    y0, y1 = max(0, ys.min() - span - 2), min(H, ys.max() + span + 2)
    N = Nfull[y0:y1, x0:x1].astype(bool)
    Wm = Wfull[y0:y1, x0:x1].astype(bool)
    best = (-1.0, 0, 0)
    for dy in range(-span, span + 1, 1):
        Ws = np.roll(Wm, dy, axis=0)
        for dx in range(-span, span + 1, 1):
            Wd = np.roll(Ws, dx, axis=1)
            inter = np.count_nonzero(N & Wd)
            union = np.count_nonzero(N | Wd)
            iou = inter / union if union else 0.0
            if iou > best[0]:
                best = (iou, dx, dy)
    return float(best[1]), float(best[2])


def _align_tx(mask_new, mask_worn, axis, r):
    """곧은 몸통변(0.35~0.70r) 좌우 에지 중앙값으로 x 평행이동만 추정 (152554 방식).
    반환 (tx, 0.0)."""
    en = _edges(mask_new, axis, r)
    ew = _edges(mask_worn, axis, r)
    if en is None or ew is None:
        return 0.0, 0.0
    yn, Ln, Rn = en
    yw, Lw, Rw = ew
    cy = float(axis[1])
    ys = np.arange(int(max(yn.min(), yw.min())), int(min(yn.max(), yw.max())) + 1)
    if len(ys) < 20:
        return 0.0, 0.0
    b = (np.abs(ys - cy) >= 0.35 * r) & (np.abs(ys - cy) <= 0.70 * r)
    if b.sum() < 10:
        return 0.0, 0.0
    dL = np.interp(ys, yn, Ln) - np.interp(ys, yw, Lw)
    dR = np.interp(ys, yn, Rn) - np.interp(ys, yw, Rw)
    return 0.5 * (float(np.median(dL[b])) + float(np.median(dR[b]))), 0.0


def measure(mask_new, mask_worn, axis, r, um_per_px, align='iou'):
    """새/마모 마스크 외곽선 후퇴로 마모 측정 (코너+플랭크 전체).
    align: 'iou' = 2D IoU 몸통 정렬(dx,dy) / 'tx' = x평행이동만(152554 방식).
    원리: 정렬 잔차 제거 후, 새 외곽선 각 점에서 '마모 마스크가 얼마나 안쪽으로
    물러났나'(마모까지 거리, 새가 마모 밖일 때만)를 거리변환으로 잰다.
    반환: {vb_um, tip_um, retreat_pts:[(x,y,um)], align_px} 또는 None."""
    cy = float(axis[1])
    if align == 'tx':
        dx, dy = _align_tx(mask_new, mask_worn, axis, r)
    else:
        dx, dy = _align_2d(mask_new, mask_worn, axis, r)
    H, W = mask_new.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    worn = cv2.warpAffine(mask_worn, M, (W, H), flags=cv2.INTER_NEAREST)
    tx = (dx ** 2 + dy ** 2) ** 0.5
    # 새 외곽선 각 점에서 worn 경계까지 거리 (새가 worn 밖 = 재료 제거된 곳만)
    dt_to_worn = cv2.distanceTransform((1 - worn).astype(np.uint8), cv2.DIST_L2, 3)
    cnts, _ = cv2.findContours(mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    retreats, pts = [], []
    tip_um = 0.0
    for x, y in c:
        rad = abs(y - cy)
        if rad < 0.45 * r:                 # 축쪽 안(중심 웹)은 마모 무관 → 제외
            continue
        if worn[y, x] == 0:                # 새 외곽선이 worn 밖 = 여기서 물러남
            d = float(dt_to_worn[y, x])
            if d > 1.0:
                retreats.append(d)
                pts.append((int(x), int(y), d * um_per_px))
                if rad >= 0.90 * r:        # 코너(날 끝) 존
                    tip_um = max(tip_um, d * um_per_px)
    if not retreats:
        return {'vb_um': 0.0, 'tip_um': 0.0, 'retreat_pts': [], 'align_px': tx}
    vb_px = float(np.percentile(retreats, 90))
    return {
        'vb_um': vb_px * um_per_px,
        'tip_um': tip_um,
        'retreat_pts': pts,
        'align_px': tx,
    }
