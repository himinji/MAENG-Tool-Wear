# -*- coding: utf-8 -*-
"""옆날 마모 측정 v2 — 서브픽셀 에지 스냅 추가 (v1: wear_side_compare.py).

v1 대비 달라진 점
-----------------
* SAM 마스크는 역광 번짐 구간을 못 덮어 실제 외곽보다 안쪽에서 끊긴다
  (실측: 새 6.7px/32µm, 마모 9.3px/45µm — 두 공구가 서로 달라 마모로 오인됨).
  → 아래 경계를 '내부 밝기와 배경 밝기의 중간값(50%)' 지점으로 서브픽셀 스냅.
  → 위 경계는 대비가 없어(-1~-5단계) 자동으로 스냅을 건너뛰고 SAM 값 유지.
* 정반사 하이라이트로 인한 스파이크 방지: 마스크 경계에서 25px 초과 이탈 시
  스냅 포기 + u 방향 21칸 이동중앙값에서 3px 초과 이탈분 대체(디스파이크).
* refine_mask() 로 마스크 경계 자체를 스냅 위치까지 갱신 → 마스크와 측정 에지가
  하나로 통합(그려지는 외곽선도 하나).
* 스냅/프로파일 범위를 이미지 오른쪽 끝까지 확장. v1은 1400px에서 끊겨 그 지점에
  10~11px 단차가 있었음. 측정 구간 6.26mm → 8.38mm.
* Initial vs Test44 결과: 날1 18.2µm, 날2 26.8µm (v1은 25.6 / 35.6µm).

방법
----
1. 두 데이터셋 각각 find_ref_angle.py의 CSV에서 피크 2개(0°/180°)를 플래토
   중심으로 찾고, 그 가운데 각도 + 날끝 고립 경계 오프셋으로 판별 프레임 선정.
2. **위상 정밀 정렬(마모공구)**: 스텝이 정확히 1°가 아니고 마모로 피크가
   들쭉날쭉해 phase-match 한 장이 최적이 아닐 수 있으므로, 마모공구 후보를
   앞뒤 ±search장(기본 ±2, 총 5장) 모두 SAM 검출 → 새 공구 외곽선과의
   **y값 차이 RMSE가 최소인 프레임**을 정답 위상으로 선택.
   - 각 x마다 외곽선 y가 위/아래 2개씩 있으므로 둘 다 사용
   - 날끝(x가 가장 작은 쪽) --tip-skip px 구간은 제외 — 마모 때문에 새 공구와
     매칭이 잘 안 되는 부위라 위상 판단을 흐림
3. 각 사진에서 SAM 날 검출(sam_blade_detect.detect_blade, 날끝 복원 포함).
4. 정렬/스케일: 몸통 실루엣 중심선 y로 상하 정렬, 몸통 높이비로 배율.
   µm/px = 직경 / 새 공구 레퍼런스 피크 y_diff.
5. 후퇴 측정: 코너 원점, 축방향 거리 u의 에지 반경 r(u) 비교.
   retreat(u) = r_new(u) - r_worn(u)  (+ = 마모로 에지가 축쪽으로 후퇴)

사용법
------
python wear_side_compare.py --new-dir "...\Initial\옆" --worn-dir "...\Test44\Toolwear\옆"
    --new-csv ref_angle_out_v6\ref_angle_scores.csv
    --worn-csv ref_angle_out_test44\ref_angle_scores.csv
    [--diam 8.0] [--pairing auto] [--search 2] [--out wear_side_out]
--pairing: 2날 180° 모호성 해소. auto(기본)=새×마모 4조합의 RMSE를 모두 재서
           합이 작은 매칭을 자동 선택. direct/swap 으로 강제 지정 가능
--search: 마모공구 위상 탐색 반경(프레임, 기본 2 = 앞뒤 ±2, 총 5장)
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sam_blade_detect import (detect_blade, get_predictor, imread_gray,
                              merge_height, tool_silhouette)


def load_scores(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    y = np.array([int(r["y_diff"]) for r in rows])
    files = [r["file"] for r in rows]
    return y, files


def plateau_center(y, i_peak, tol=5):
    """i_peak가 속한 연속 플래토(최대-tol 이내)의 중심 인덱스."""
    lvl = y[i_peak] - tol
    lo = hi = i_peak
    while lo > 0 and y[lo - 1] >= lvl:
        lo -= 1
    while hi < len(y) - 1 and y[hi + 1] >= lvl:
        hi += 1
    return (lo + hi) / 2


def peaks_and_mids(y, min_sep=90):
    i1 = int(np.argmax(y))
    far = np.abs(np.arange(len(y)) - i1) >= min_sep
    i2 = int(np.arange(len(y))[far][np.argmax(y[far])])
    p1, p2 = sorted([plateau_center(y, i1), plateau_center(y, i2)])
    hp = p2 - p1                      # 반주기(≈180°)
    mid_a = (p1 + p2) / 2             # 90° 지점
    mid_b = mid_a - hp if mid_a - hp >= 0 else mid_a + hp   # 270° 지점
    ref_ydiff = (y[int(round(p1))] + y[int(round(p2))]) / 2
    return p1, p2, int(round(mid_a)), int(round(mid_b)), ref_ydiff


def isolation_offset(folder, files, valley, thresh, max_scan=30):
    """계곡에서 몇 장 뒤에 날끝이 고립(M=0 2연속)되는지 반환 (경계 오프셋)."""
    consec = 0
    for k in range(0, max_scan):
        i = valley + k
        if i >= len(files):
            break
        gray = imread_gray(Path(folder) / files[i])
        tool, tip = tool_silhouette(gray, thresh)
        if merge_height(tool, tip) <= 0:
            consec += 1
            if consec >= 2:
                return k - 1
        else:
            consec = 0
    return 12


def body_axis(tool, w):
    """몸통 대역(x=w-400~w-100)에서 축 중심 y와 몸통 높이."""
    band = tool[:, w - 400:w - 100]
    tops, bots = [], []
    for c in range(band.shape[1]):
        ys = np.flatnonzero(band[:, c])
        if len(ys):
            tops.append(ys.min())
            bots.append(ys.max())
    return float(np.median([(t + b) / 2 for t, b in zip(tops, bots)])), \
        float(np.median([b - t for t, b in zip(tops, bots)]))


def _cross_scan(col, y_from, d, lvl, limit=400):
    """y_from 에서 d 방향으로 훑어 lvl 을 지나는 첫 지점(서브픽셀)."""
    for k in range(0, limit):
        y1, y2 = y_from + d * k, y_from + d * (k + 1)
        if not (0 <= y1 < len(col) and 0 <= y2 < len(col)):
            return None
        a, b = col[y1], col[y2]
        if (a - lvl) * (b - lvl) <= 0 and a != b:
            return y1 + d * (lvl - a) / (b - a)
    return None


def jig_height(gray, bright=250, min_area=20000):
    """왼쪽 지그의 밝은 구멍 세로지름(px, 서브픽셀). 배율 기준용.

    공구 몸통은 플루트 구간이라 회전 위상에 따라 실루엣 높이가 변해(같은 공구에서
    1443 vs 1318px) 배율 기준으로 불안정하다. 지그는 고정물이라 세션 내
    반복성이 0.03px 수준으로 훨씬 안정적이다.
    찾지 못하면 None → 호출측에서 몸통 기준으로 폴백.
    """
    h, w = gray.shape
    bw = (gray[:, :w // 3] > bright).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    best = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        top, hh = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
        if a < min_area or top <= 1 or top + hh >= h - 1:   # 배경/잘린 구멍 제외
            continue
        if best is None or a > best[0]:
            best = (a, int(round(cent[i][0])), int(round(cent[i][1])), top, hh)
    if best is None:
        return None
    _, cx, cy, top, hh = best
    hs = []
    for x in range(max(cx - 60, 0), min(cx + 61, w), 10):
        col = gray[:, x].astype(float)
        vin = float(np.median(col[max(cy - hh // 4, 0):cy + hh // 4]))      # 구멍 내부(밝음)
        vout = float(np.median(col[max(top - 80, 0):max(top - 20, 1)]))     # 지그 몸체(어두움)
        if vin - vout < 40:
            continue
        lvl = (vin + vout) / 2
        y1 = _cross_scan(col, cy, -1, lvl)
        y2 = _cross_scan(col, cy, +1, lvl)
        if y1 is not None and y2 is not None:
            hs.append(y2 - y1)
    return float(np.median(hs)) if len(hs) >= 5 else None


def auto_click(tool, tip, axis_y, side="bottom"):
    """날끝 오른쪽 150~450px 대역에서 지정한 쪽(기본 아래) 웨지의 무게중심."""
    ys, xs = np.nonzero(tool)
    band = (xs >= tip[0] + 150) & (xs <= tip[0] + 450)
    sel = band & ((ys > axis_y) if side == "bottom" else (ys < axis_y))
    if not sel.any():
        raise RuntimeError(f"{side}쪽 날 웨지를 찾지 못함")
    return (int(xs[sel].mean()), int(ys[sel].mean())), side


def trim_pale_tip(gray, tool, tip, click, r=30, fill=0.8, march_max=60, level=0.32):
    """연한 날끝 트림: 반지름 r 원을 날끝→클릭 방향으로 밀며,
    원 안 실루엣 픽셀 중 '진한' 비율이 fill 이상이 되면 멈춘다.
    통과하는 동안 원 안의 연한 픽셀을 제외.

    '진한' 기준 = 몸통 밝기에서 배경 쪽으로 level 만큼 간 지점.
    level=0.5 면 밝기 50% 지점(스냅과 같은 규약)이지만 마모 날끝의 뿌연
    영역이 거의 안 잘린다. 기본 0.32 는 새 공구가 전혀 안 잘리는 선에서
    가장 많이 자르는 값(코팅 데이터 실측: 0.32→약 130, 새 0px / 마모 최대 1509px).

    [주의] 이 방식은 픽셀을 직접 깎아내므로 많이 자를수록 경계가 울퉁불퉁해진다.
    매끄러운 경계가 필요하면 연한 영역을 '음성 프롬프트'로 SAM 에 알려주는
    쪽이 낫다 (기본 꺼짐인 이유).

    반환 (제외마스크|None, 새 날끝). 날끝이 처음부터 진하면 (None, tip).
    """
    h, w = gray.shape
    ys, xs = np.nonzero(tool)
    band = (xs >= tip[0] + 150) & (xs <= tip[0] + 450)
    body = float(np.median(gray[ys[band], xs[band]]))
    bg = float(np.median(gray[:, max(tip[0] - 300, 0):max(tip[0] - 60, 1)]))
    t_dark = body + level * (bg - body)
    dark = tool & (gray <= t_dark)
    v = np.array([click[0] - tip[0], click[1] - tip[1]], float)
    v /= np.linalg.norm(v)
    yy, xx = np.ogrid[:h, :w]
    removed = np.zeros((h, w), bool)
    for t in range(0, march_max + 1):
        cx, cy = tip[0] + t * v[0], tip[1] + t * v[1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        n_sil = int((disk & tool).sum())
        if n_sil and (disk & dark).sum() / n_sil >= fill:
            break
        removed |= disk & tool & ~dark
    if not removed.any():
        return None, tip
    kept = tool & ~removed
    ky, kx = np.nonzero(kept)
    x_min = int(kx.min())
    return removed, (x_min, int(np.median(ky[kx <= x_min + 3])))


def snap_edge(gray, mask, tip, side, u, contrast_min=25):
    """마스크 경계에서 출발해 밝기 50% 지점을 서브픽셀로 찾은 에지 y.

    SAM 마스크는 역광 번짐 구간을 못 덮어 실제 외곽보다 7~9px 안쪽에서 끊긴다.
    내부 밝기와 배경 밝기의 중간값을 지나는 지점이 물리적 외곽이므로 거기로 스냅.
    대비가 부족하면(위 경계처럼 내부 경계면) 마스크 경계를 그대로 반환.
    """
    h, w = gray.shape
    x = tip[0] + u
    if x >= w:
        return None
    ys = np.flatnonzero(mask[:, x])
    if len(ys) == 0:
        return None
    d = 1 if side == "bottom" else -1
    y_sam = int(ys.max()) if side == "bottom" else int(ys.min())

    col = gray[:, x].astype(float)
    ins = [y_sam - d * k for k in range(10, 46)]
    out = [y_sam + d * k for k in range(8, 41)]
    ins = [y for y in ins if 0 <= y < h]
    out = [y for y in out if 0 <= y < h]
    if len(ins) < 5 or len(out) < 5:
        return float(y_sam)
    vi, vo = float(np.median(col[ins])), float(np.median(col[out]))
    if vo - vi < contrast_min:                 # 대비 부족 → 스냅 불가
        return float(y_sam)
    lvl = (vi + vo) / 2
    for k in range(-5, 31):
        y1, y2 = y_sam + d * k, y_sam + d * (k + 1)
        if not (0 <= y1 < h and 0 <= y2 < h):
            continue
        a, b = col[y1], col[y2]
        if (a - lvl) * (b - lvl) <= 0 and a != b:
            y = y1 + d * (lvl - a) / (b - a)
            # 정반사 하이라이트 등으로 엉뚱한 곳이 잡히면 마스크 경계 유지
            return y if abs(y - y_sam) <= 25 else float(y_sam)
    return float(y_sam)


def _despike(us, ys, win=21, tol=3.0):
    """에지 시계열의 스파이크 제거 — 국소 중앙값에서 tol px 넘게 벗어나면 대체."""
    ys = np.asarray(ys, dtype=float)
    if len(ys) < win:
        return ys
    pad = win // 2
    ext = np.pad(ys, pad, mode="edge")
    med = np.array([np.median(ext[i:i + win]) for i in range(len(ys))])
    bad = np.abs(ys - med) > tol
    out = ys.copy()
    out[bad] = med[bad]
    return out


def refine_mask(mask, tip, side, us, ys):
    """마스크의 해당 쪽 경계를 스냅된 에지까지 확장/축소해 하나로 통합."""
    out = mask.copy()
    h, w = mask.shape
    for u, y in zip(us, ys):
        x = tip[0] + u
        if x >= w:
            break
        col = np.flatnonzero(out[:, x])
        if len(col) == 0:
            continue
        yy = int(round(y))
        if side == "bottom":
            y0 = int(col.max())
            if yy > y0:
                out[y0:min(yy + 1, h), x] = True
            elif yy < y0:
                out[max(yy + 1, 0):y0 + 1, x] = False
        else:
            y0 = int(col.min())
            if yy < y0:
                out[max(yy, 0):y0 + 1, x] = True
            elif yy > y0:
                out[y0:min(yy, h), x] = False
    return out


def snapped_series(gray, mask, tip, side, length=None):
    """스냅된 에지의 (u, 절대 y) 시계열 — 스파이크 제거 포함.

    length 를 주지 않으면 이미지 오른쪽 끝까지. (중간에서 끊으면 그 지점부터
    보정 안 된 원래 마스크 경계가 남아 단차가 생긴다.)
    """
    if length is None:
        length = gray.shape[1] - tip[0]
    us, ys = [], []
    for u in range(0, length):
        y = snap_edge(gray, mask, tip, side, u)
        if y is None:
            if tip[0] + u >= gray.shape[1]:
                break
            continue
        us.append(u)
        ys.append(y)
    return np.array(us), _despike(us, ys)


def edge_profile_snapped(gray, mask, tip, axis_y, side, length=None):
    """스냅된 에지 기준 반경 프로파일 r(u) [px, 서브픽셀]."""
    us, ys = snapped_series(gray, mask, tip, side, length)
    return us, np.abs(ys - axis_y)


def edge_profile(mask, tip, axis_y, side, length=None):
    """코너 기준 축방향 거리 u에서의 에지 반경 r(u)[px]. side='bottom'|'top'."""
    h, w = mask.shape
    if length is None:
        length = w - tip[0]
    us, rs = [], []
    for u in range(0, length):
        x = tip[0] + u
        if x >= w:
            break
        ys = np.flatnonzero(mask[:, x])
        if len(ys) == 0:
            continue
        edge_y = ys.max() if side == "bottom" else ys.min()
        us.append(u)
        rs.append(abs(edge_y - axis_y))
    return np.array(us), np.array(rs, dtype=float)


def _smooth(a, k=15):
    """이동 평균(창 k열). 위 경계는 스냅이 안 붙어 정수 픽셀 계단이 남으므로
    이웃 열을 평균해 양자화 노이즈를 평활한다."""
    pad = k // 2
    return np.convolve(np.pad(a, pad, mode="edge"), np.ones(k) / k, mode="valid")


def _contour_smooth(mask, w=41):
    """마스크 외곽선을 원형 이동평균(창 w점)으로 평활해 다시 채움.

    연한영역 게이트 마스크는 실루엣 교집합·열기 뒤에도 재료 윤곽의 잔물결이
    남는데, 경계선 자체를 곡선으로 평활하면 손으로 그린 듯 매끄러워진다.
    (통계 구간 반경 영향 0 실측)"""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c) < w * 3:
        return mask
    kern = np.ones(w) / w
    pad = w // 2
    sm = np.empty_like(c)
    for d in range(2):
        ext = np.concatenate([c[-pad:, d], c[:, d], c[:pad, d]])   # 원형 패딩
        sm[:, d] = np.convolve(ext, kern, mode="valid")[:len(c)]
    out = np.zeros_like(mask, np.uint8)
    cv2.fillPoly(out, [np.round(sm).astype(np.int32)], 1)
    return out.astype(bool)


def outline_yy(mask, tip, axis_y, length=None, gray=None):
    """각 x(코너 기준 u)에서 외곽선의 y 두 값(위/아래)을 축 기준으로 반환.

    gray 를 주면 밝기 50% 지점으로 스냅(대비 있는 경계만). 반환은 축 기준 상대 y.
    """
    h, w = mask.shape
    if length is None:
        length = w - tip[0]
    us, tops, bots = [], [], []
    for u in range(0, length):
        x = tip[0] + u
        if x >= w:
            break
        ys = np.flatnonzero(mask[:, x])
        if len(ys) == 0:
            continue
        if gray is None:
            yt, yb = float(ys.min()), float(ys.max())
        else:
            yt = snap_edge(gray, mask, tip, "top", u)
            yb = snap_edge(gray, mask, tip, "bottom", u)
            if yt is None or yb is None:
                continue
        us.append(u)
        tops.append(yt - axis_y)
        bots.append(yb - axis_y)
    return np.array(us), np.array(tops, dtype=float), np.array(bots, dtype=float)


def outline_rmse(n, wn, s, tip_skip, um, u_max=None, edge="both"):
    """새/마모 외곽선의 y값 차이 RMSE (µm).

    구간: tip_skip <= u <= u_max
      - 날끝(u < tip_skip)은 마모로 매칭이 흐려져 제외
      - u_max 를 주면 그 뒤(절삭과 무관한 뒷부분)를 잘라내고 계산 →
        마모가 실제로 일어나는 앞부분 형상만으로 위상을 고른다
    각 x마다 외곽선 y가 위/아래 2개씩 있으므로 둘 다 사용한다.
    마모 쪽은 배율 s를 곱해 새 공구 좌표계로 맞춘 뒤 비교.
    """
    un, nt, nb = outline_yy(n["det"]["final"], n["tip"], n["axis"], gray=n["gray"])
    uw, wt, wb = outline_yy(wn["det"]["final"], wn["tip"], wn["axis"], gray=wn["gray"])
    common = np.intersect1d(un, uw)
    common = common[common >= tip_skip]
    if u_max is not None:
        common = common[common <= u_max]
    if len(common) < 50:
        return float("inf"), 0
    i_n = np.searchsorted(un, common)
    i_w = np.searchsorted(uw, common)
    d_top = nt[i_n] - wt[i_w] * s
    d_bot = nb[i_n] - wb[i_w] * s
    if edge == "bottom":
        d = d_bot
    elif edge == "top":
        d = d_top
    else:
        d = np.concatenate([d_top, d_bot])
    rmse_px = float(np.sqrt(np.mean(d ** 2)))
    return rmse_px * um, len(common)


def process_frame(folder, files, idx, predictor, thresh, side, sam_win=None,
                  tip_trim=False, trim_level=0.32, pale_neg=True):
    """한 프레임 SAM 검출 + 기하/프로파일. 실패 시 None.

    sam_win: (x0, x1) 고정 크롭 좌표 — 새 공구 레퍼런스 날끝 기준
    [ref_x−R/2, ref_x+R] 로 main 에서 한 번 정해 모든 사진에 동일 적용.
    주어지면 SAM 을 전체 프레임 대신 이 창(세로는 그대로) 위에서 돌리고,
    결과 마스크/날끝을 원본 좌표로 되돌린다. 축·몸통·지그 계산과
    이후의 스냅·프로파일은 항상 원본 프레임에서 수행.
    tip_trim: 연한 날끝 트림(trim_pale_tip). 날끝/클릭이 트림된
    실루엣 기준으로 다시 잡히고, 최종 마스크에서도 트림 픽셀을 뺀다.
    pale_neg: 연한영역 게이트(A안). 날끝에서 이어진 연한 픽셀이 200개
    이상인 프레임에서만 음성점 추가 + 날끝 패치 생략. 크롭 모드에서만 동작.
    """
    gray = imread_gray(Path(folder) / files[idx])
    if gray is None:
        return None
    try:
        tool, tip0 = tool_silhouette(gray, thresh)
        axis_y, body_h = body_axis(tool, gray.shape[1])
        click, sd = auto_click(tool, tip0, axis_y, side)
        trim = None
        if tip_trim:
            trim, tip1 = trim_pale_tip(gray, tool, tip0, click, level=trim_level)
            if trim is not None:
                print(f"    날끝 트림 {files[idx]}: {int(trim.sum())}px 제외, "
                      f"날끝 {tip0} → {tip1}")
                tool = tool & ~trim
                tip0 = tip1
                click, sd = auto_click(tool, tip0, axis_y, side)
        if sam_win:
            x0 = max(int(sam_win[0]), 0)
            x1 = min(int(sam_win[1]), gray.shape[1])
            crop = gray[:, x0:x1]
            # 음성점은 크롭 좌표로 다시 잡는다: 기본값(w-150 등)은 크롭에선 날 위에 떨어짐.
            # 배경 위 음성점은 힘이 없어 반대쪽 날까지 마스크가 번질 수 있으므로
            # 반대쪽 웨지 무게중심(확실히 공구 위)을 음성점으로 쓴다.
            opp, _ = auto_click(tool, tip0, axis_y,
                                "top" if sd == "bottom" else "bottom")
            negs = [(opp[0] - x0, opp[1]),                     # 반대쪽 날 위
                    (max(40, tip0[0] - x0 - 300), click[1])]   # 날끝 왼쪽 배경
            # 연한영역 게이트: 날끝에서 이어진 연한(뿌연) 픽셀이 200개 이상이면
            # 그 무게중심에 음성점을 추가하고 날끝 패치를 생략(A안) →
            # SAM 이 확실히 진한 재료만 매끄럽게 잡는다.
            # 호출측에서 마모 공구에만 pale_neg=True 를 준다 — 새 공구는
            # 항상 날끝까지 잡아야 하므로 게이트 대상이 아니다.
            pale_gate = False
            if pale_neg:
                prem, _ = trim_pale_tip(gray, tool, tip0, click)
                if prem is not None and prem.sum() >= 200:
                    pry, prx = np.nonzero(prem)
                    negs.append((int(prx.mean()) - x0, int(pry.mean())))
                    pale_gate = True
                    print(f"    연한영역 게이트 {files[idx]}: {int(prem.sum())}px "
                          f"→ 음성점 추가 + 날끝 패치 생략")
            det = detect_blade(crop, (click[0] - x0, click[1]), predictor, thresh,
                               negs=negs)
            full = np.zeros(gray.shape, bool)
            full[:, x0:x1] = det["base"] if pale_gate else det["final"]
            if pale_gate:
                # SAM 경계가 날끝 옆 밝은 반사 얼룩(밝기>=210, 실루엣 밖)을 물고
                # 오는 것 제거 — 실루엣과 교집합해 재료 픽셀만 남긴다
                full &= tool
                # 교집합이 남기는 픽셀 톱니·얼룩 사이 가는 혀를 모폴로지 열기로
                # 정리 → 진한 재료의 매끄러운 포락선만 남김 (27px: 주머니
                # 아티팩트가 사라지는 최소 커널, 실측 통계 영향 0)
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
                full = cv2.morphologyEx(full.astype(np.uint8), cv2.MORPH_OPEN,
                                        kern).astype(bool)
                n_cc, cc_lab, cc_st, _ = cv2.connectedComponentsWithStats(
                    full.astype(np.uint8), 8)
                if n_cc > 2:                # 조각나면 최대 성분만 유지
                    full = cc_lab == (1 + int(np.argmax(cc_st[1:, cv2.CC_STAT_AREA])))
                full = _contour_smooth(full)   # 경계선 이동평균 → 잔물결 제거
            det["final"] = full
            det["tip"] = (det["tip"][0] + x0, det["tip"][1])
            det["sam_crop"] = (x0, x1)
        else:
            det = detect_blade(gray, click, predictor, thresh)
        if trim is not None:
            # detect_blade 내부의 날끝 패치가 연한 픽셀을 되살리므로 다시 빼고,
            # 날끝(u 원점)도 트림된 마스크 기준으로 재계산
            det["final"] &= ~trim
            fy, fx = np.nonzero(det["final"])
            x_min = int(fx.min())
            det["tip"] = (x_min, int(np.median(fy[fx <= x_min + 3])))
    except Exception as e:
        print(f"    [건너뜀] idx {idx} {files[idx]}: {e}")
        return None
    # 스냅된 에지로 마스크 경계를 갱신 → 마스크와 측정 에지가 하나로 통합
    us, ys = snapped_series(gray, det["final"], det["tip"], sd)
    det["final"] = refine_mask(det["final"], det["tip"], sd, us, ys)
    r = np.abs(ys - axis_y)
    return {"gray": gray, "det": det, "tip": det["tip"], "axis": axis_y,
            "body_h": body_h, "jig_h": jig_height(gray),
            "side": sd, "file": files[idx], "idx": idx, "u": us, "r": r}


def scale_ratio(n, wn, ref="jig"):
    """배율 s와 그 출처. 지그가 잡히면 지그, 아니면 공구 몸통으로 폴백."""
    if ref == "jig" and n.get("jig_h") and wn.get("jig_h"):
        return n["jig_h"] / wn["jig_h"], "지그"
    return n["body_h"] / wn["body_h"], "몸통"


def crop_view(img, tip, axis, u_max_px, margin=120, half=900):
    """날끝~u_max 구간만 잘라낸 보기용 이미지."""
    h, w = img.shape[:2]
    x0, x1 = max(int(tip[0] - margin), 0), min(int(tip[0] + u_max_px + margin), w)
    y0, y1 = max(int(axis - half), 0), min(int(axis + half), h)
    return img[y0:y1, x0:x1]


def align_profiles(n, wn, scale_ref="jig", u_max=None):
    """새/마모 프로파일을 코너·축 정렬 + 배율 s로 공통 u에 맞춤.

    반환: dict(s, s_src, common, rn, rw, u_lo, u_hi) 또는 None.
    u_lo..u_hi = 날끝(코너 램프)과 마스크 끝을 제외한 몸통 구간.
    """
    s, s_src = scale_ratio(n, wn, scale_ref)
    common = np.intersect1d(n["u"], wn["u"])
    if len(common) < 50:
        return None
    rn = n["r"][np.searchsorted(n["u"], common)]
    rw = wn["r"][np.searchsorted(wn["u"], common)] * s
    plateau = float(np.median(rn[common >= common.max() - 400]))
    reach = np.flatnonzero(rn >= 0.9 * plateau)
    u_lo = int(common[reach[0]]) + 30 if len(reach) else 30
    u_hi = int(common.max()) - 100
    if u_max is not None:                      # 크롭: 측정 구간도 같은 범위로
        u_hi = min(u_hi, int(u_max))
    return {"s": s, "s_src": s_src, "common": common, "rn": rn, "rw": rw,
            "u_lo": u_lo, "u_hi": u_hi}


def wear_stats(aln, um):
    """정렬 결과에서 마모 후퇴 통계(µm)를 계산. 반환: (vb_max, vb_mean, retreat_um)"""
    common, rn, rw = aln["common"], aln["rn"], aln["rw"]
    u_lo, u_hi = aln["u_lo"], aln["u_hi"]
    zone = (common >= u_lo) & (common <= u_hi)
    retreat_um = (rn - rw) * um
    if not zone.any():
        return float("nan"), float("nan"), retreat_um
    d = retreat_um[zone]
    return float(d.max()), float(d.mean()), retreat_um


def save_candidate_views(out_dir, tag, n, wn, aln, d, resid, vbmean, vbmax, scale=0.35):
    """위상 후보 1장의 확인용 이미지 2종을 저장.

    _outline: 후보 본인 사진 위에 본인 외곽선
    _overlay: 새 공구 사진 위에 새(파랑) + 이 후보(빨강, 코너·축 정렬) 겹침
    파일명·라벨에 오프셋과 형상잔차/후퇴를 찍어 표와 대조 가능하게 한다.
    """
    cdir = out_dir / f"{tag}_candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    stem = Path(wn["file"]).stem
    l1 = f"off{d:+d}  {wn['file']}"
    l2 = f"RMSE={resid:.1f}um  retreat mean/max={vbmean:.1f}/{vbmax:.1f}um"

    ind = cv2.cvtColor(wn["gray"], cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(wn["det"]["final"].astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ind, cnts, -1, (0, 0, 255), 3)
    cv2.circle(ind, wn["tip"], 20, (0, 255, 255), 3)
    cv2.putText(ind, l1, (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
    cv2.putText(ind, l2, (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
    cv2.imencode(".png", cv2.resize(ind, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA))[1].tofile(
        str(cdir / f"{tag}_off{d:+d}_{stem}_outline.png"))

    s = aln["s"]
    H, W = n["gray"].shape
    canvas = cv2.cvtColor(n["gray"], cv2.COLOR_GRAY2BGR)
    M = np.float32([[s, 0, n["tip"][0] - s * wn["tip"][0]],
                    [0, s, n["axis"] - s * wn["axis"]]])
    wm = cv2.warpAffine(wn["det"]["final"].astype(np.uint8), M, (W, H)) > 0
    for m, col in [(n["det"]["final"], (255, 120, 0)), (wm, (0, 0, 255))]:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, col, 3)
    # 배경이 흰색/검정이 섞여 있어 흰 글씨만 쓰면 흰 배경 위에서 안 보임 → 검은 테두리
    for txt, ypos in [(f"blue=new({n['file']})  red=worn({wn['file']})  {l1}", 110),
                      (l2, 190)]:
        cv2.putText(canvas, txt, (60, ypos), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 9)
        cv2.putText(canvas, txt, (60, ypos), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.imencode(".png", cv2.resize(canvas, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA))[1].tofile(
        str(cdir / f"{tag}_off{d:+d}_{stem}_overlay.png"))


def save_new_view(out_dir, tag, n, scale=0.35):
    """후보 폴더에 새 공구 본인 사진 + 외곽선 1장 저장 (후보들과 나란히 비교용)."""
    cdir = out_dir / f"{tag}_candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    ind = cv2.cvtColor(n["gray"], cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(n["det"]["final"].astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ind, cnts, -1, (255, 120, 0), 3)
    cv2.circle(ind, n["tip"], 20, (0, 255, 255), 3)
    cv2.putText(ind, f"NEW  {n['file']}", (60, 110), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (255, 120, 0), 4)
    cv2.imencode(".png", cv2.resize(ind, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA))[1].tofile(
        str(cdir / f"{tag}_new_{Path(n['file']).stem}_outline.png"))


def main():
    ap = argparse.ArgumentParser(description="옆날 새/마모 SAM 비교 마모 측정")
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--worn-dir", required=True)
    ap.add_argument("--new-csv", required=True)
    ap.add_argument("--worn-csv", required=True)
    ap.add_argument("--diam", type=float, default=8.0, help="공구 직경 mm (기본 8)")
    ap.add_argument("--pairing", choices=["auto", "direct", "swap"], default="auto",
                    help="새/마모 2장씩의 짝 결정 (기본 auto: 두 매칭의 RMSE 합이 "
                         "작은 쪽 자동 선택). direct=순서대로, swap=교차")
    ap.add_argument("--swap", action="store_true",
                    help="(구버전 호환) --pairing swap 과 동일")
    ap.add_argument("--side", choices=["bottom", "top"], default="bottom",
                    help="측정할 절삭날 쪽 (기본 bottom)")
    ap.add_argument("--search", type=int, default=3,
                    help="마모공구 위상 탐색 반경(프레임, 기본 3 = 총 7장)")
    ap.add_argument("--rmse-edge", choices=["bottom", "top", "both"], default="both",
                    help="위상 선택 RMSE 를 어느 경계에서 계산할지 (기본 both). "
                         "bottom 만 쓰면 회전에 둔감해 판별 마진이 0.2~3.8µm 로 "
                         "무너진다(실측) — both 권장")
    ap.add_argument("--adoc", type=float, default=2.0,
                    help="축방향 절삭깊이 mm (기본 2.0). RMSE 계산 구간을 "
                         "날끝~crop-factor×adoc 으로 제한하는 데 쓴다")
    ap.add_argument("--crop-factor", type=float, default=2.0,
                    help="RMSE 계산 범위 = adoc × 이 값 (기본 2.0 → 0~4mm). "
                         "0 이면 제한 없음(전 구간)")
    ap.add_argument("--tip-skip", type=int, default=300,
                    help="RMSE 계산에서 제외할 날끝 구간 길이(px, 기본 300). "
                         "날끝은 마모로 새 공구와 매칭이 잘 안 되므로 제외")
    ap.add_argument("--scale-ref", choices=["jig", "body"], default="jig",
                    help="배율 기준 (기본 jig: 왼쪽 고정 지그 구멍 세로지름. "
                         "body: 공구 몸통 실루엣 높이 — 플루트 구간이라 위상에 흔들림)")
    ap.add_argument("--thresh", type=int, default=210)
    ap.add_argument("--tip-trim", action="store_true",
                    help="연한 날끝 트림: 반지름 30px 원을 날끝→클릭 방향으로 밀며 "
                         "원 안 실루엣의 진한 비율이 80% 미만인 동안 연한 픽셀을 "
                         "제외 (기본 꺼짐)")
    ap.add_argument("--tip-trim-level", type=float, default=0.32,
                    help="트림의 '진한' 기준 = 몸통에서 배경 쪽으로 이 비율만큼 간 "
                         "밝기 (기본 0.32 ≈ 밝기 130). 값을 낮출수록 더 많이 잘리며, "
                         "0.25 부근부터는 새 공구까지 잘리기 시작한다")
    ap.add_argument("--pale-neg", choices=["on", "off"], default="on",
                    help="연한영역 게이트 (기본 on, 마모 공구 프레임에만 적용): "
                         "날끝에서 이어진 연한 픽셀이 200개 이상인 프레임만 그 "
                         "무게중심에 음성점을 추가하고 날끝 패치를 생략 → 확실히 "
                         "진한 재료만 마스킹(A안). 새 공구는 항상 날끝까지 잡음")
    ap.add_argument("--sam-crop", choices=["on", "off"], default="on",
                    help="SAM 을 고정 좌표 크롭 위에서 실행 (기본 on). "
                         "창 = x∈[ref_x−R/2, ref_x+R] — ref_x는 새 공구 "
                         "레퍼런스(피크 2장) 날끝 x, R=반경(피크 y_diff/2). "
                         "모든 사진을 같은 좌표로 자름. 세로는 자르지 않음")
    ap.add_argument("--offset", default="20",
                    help="계곡→판별 프레임 오프셋(장). 기본 20 고정 — 새 공구 판별 "
                         "프레임이 모든 테스트에서 동일해진다(테스트 간 비교 가능). "
                         "고립 경계+2 가 이를 넘는 계곡은 경고 출력. "
                         "'auto' 면 옛 방식(4개 고립 경계 최대 + 2)")
    ap.add_argument("--out", default="wear_side_out")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = {"new": args.new_dir, "worn": args.worn_dir}
    sel = {}
    for name, csv_path in [("new", args.new_csv), ("worn", args.worn_csv)]:
        y, files = load_scores(csv_path)
        p1, p2, ma, mb, ref = peaks_and_mids(y)
        sel[name] = {"files": files, "valleys": [ma, mb], "ref_ydiff": ref,
                     "peaks": [int(round(p1)), int(round(p2))]}
        print(f"[{name}] 피크 중심 {p1:.1f}/{p2:.1f} (ref y_diff={ref:.0f})  계곡 idx {ma}, {mb}")

    offsets = {}
    for name in sel:
        for v in sel[name]["valleys"]:
            offsets[(name, v)] = isolation_offset(dirs[name], sel[name]["files"], v, args.thresh)
    for (name, v), k in offsets.items():
        print(f"  [{name}] 계곡 idx {v}: 고립 경계 +{k}장")
    if str(args.offset).strip().lower() == "auto":
        # 옛 방식: 4개 고립 경계의 최댓값 + 2 — 마모 세트에 따라 새 공구
        # 판별 프레임이 흔들려 테스트 간 직접 비교가 어렵다
        common_off = max(offsets.values()) + 2
        print(f"  → 공통 오프셋 +{common_off}장 적용 (auto: 최대 고립 경계 + 2)")
    else:
        # 고정 오프셋: 새 공구 판별 프레임이 모든 테스트에서 동일해짐.
        # 고립 경계가 (오프셋-2)를 넘는 계곡은 반대날 겹침 위험 → 경고만.
        common_off = int(args.offset)
        print(f"  → 고정 오프셋 +{common_off}장 적용 (테스트 간 판별 프레임 동일)")
        for (name, v), k in offsets.items():
            if k + 2 > common_off:
                print(f"  [경고] {name} 계곡 idx {v}: 고립 경계 +{k}장 — 고정 오프셋 "
                      f"{common_off}로는 여유 {common_off - k}장 (반대날 겹침 위험). "
                      f"--offset {k + 2} 이상 또는 --offset auto 권장")
    for name in sel:
        sel[name]["mids"] = [v + common_off for v in sel[name]["valleys"]]
        f = sel[name]["files"]
        print(f"[{name}] 판별 프레임 idx {sel[name]['mids'][0]}({f[sel[name]['mids'][0]]}), "
              f"{sel[name]['mids'][1]}({f[sel[name]['mids'][1]]})")

    um = args.diam * 1000.0 / sel["new"]["ref_ydiff"]
    print(f"스케일: {um:.3f} µm/px  (직경 {args.diam}mm / 새 공구 피크 y_diff {sel['new']['ref_ydiff']:.0f}px)")

    # SAM 크롭 창(고정 좌표): 새 공구 레퍼런스(피크 2장)의 날끝 x 를 기준으로
    # [ref_x−R/2, ref_x+R] 을 한 번 정하고, 모든 사진(새·마모)을 같은 좌표로 자른다.
    # R = 새 공구 반경(px) = 피크 y_diff / 2.
    sam_win = None
    if args.sam_crop == "on":
        R = int(round(sel["new"]["ref_ydiff"] / 2.0))
        ref_tips = []
        for pk in sel["new"]["peaks"]:
            g = imread_gray(Path(args.new_dir) / sel["new"]["files"][pk])
            _, tp = tool_silhouette(g, args.thresh)
            ref_tips.append(tp[0])
        ref_x = int(round(np.mean(ref_tips)))
        sam_win = (max(ref_x - R // 2, 0), ref_x + R)
        print(f"SAM 크롭(고정 좌표): 새 공구 레퍼런스 날끝 x={ref_x} "
              f"(피크 2장 {ref_tips}) → x {sam_win[0]}~{sam_win[1]} "
              f"(R={R}px, 세로는 전체 유지)")

    # 위상 선택용 RMSE 계산 범위 — 날끝~crop_factor×adoc 으로 크롭
    if args.crop_factor > 0:
        u_max_rmse = args.crop_factor * args.adoc * 1000.0 / um
        print(f"RMSE 구간: u={args.tip_skip}~{u_max_rmse:.0f}px "
              f"= {args.tip_skip * um / 1000:.2f}~{args.crop_factor * args.adoc:.2f}mm "
              f"(adoc {args.adoc}mm × {args.crop_factor:g}),  경계={args.rmse_edge}")
    else:
        u_max_rmse = None
        print(f"RMSE 구간: u>={args.tip_skip}px 이후 전 구간 (크롭 없음)")

    worn_files = sel["worn"]["files"]
    predictor = get_predictor()
    print("SAM 로드 완료")

    # ── 새 공구 2장 검출 ──
    news = []
    for i, ni in enumerate(sel["new"]["mids"]):
        n = process_frame(args.new_dir, sel["new"]["files"], ni, predictor, args.thresh,
                          args.side, sam_win, args.tip_trim,
                          pale_neg=False)   # 새 공구는 게이트 없이 항상 날끝까지
        if n is None:
            sys.exit(f"새 공구 프레임 검출 실패: idx {ni}")
        news.append(n)
        jh = f"{n['jig_h']:.2f}" if n.get("jig_h") else "없음"
        print(f"  new[{i}]: {n['file']}  날끝={n['tip']}  축y={n['axis']:.0f}  "
              f"몸통H={n['body_h']:.0f}px  지그H={jh}px")

    # ── 마모 후보 검출 (mid별 ±search) — 매칭 평가에 재사용하므로 1회만 ──
    worn_cands = []
    for j, wmid in enumerate(sel["worn"]["mids"]):
        lst = []
        for d in range(-args.search, args.search + 1):
            wj = wmid + d
            if not (0 <= wj < len(worn_files)):
                continue
            wn = process_frame(args.worn_dir, worn_files, wj, predictor, args.thresh,
                               args.side, sam_win, args.tip_trim,
                               pale_neg=(args.pale_neg == "on"))
            if wn is not None:
                lst.append({"d": d, "wn": wn})
        worn_cands.append(lst)
        print(f"  worn[{j}]: idx {wmid} ±{args.search} → 후보 {len(lst)}장 검출")

    # ── 4개 조합(새 i × 마모 j) 각각 위상 탐색해 최소 RMSE 구하기 ──
    best_of = {}
    for i, n in enumerate(news):
        for j, lst in enumerate(worn_cands):
            ev = []
            for c in lst:
                aln = align_profiles(n, c["wn"], args.scale_ref, u_max_rmse)
                if aln is None:
                    continue
                rmse, _ = outline_rmse(n, c["wn"], aln["s"], args.tip_skip, um,
                                       u_max_rmse, args.rmse_edge)
                vbmax, vbmean, _ = wear_stats(aln, um)
                ev.append({"d": c["d"], "wn": c["wn"], "aln": aln, "resid": rmse,
                           "vbmax": vbmax, "vbmean": vbmean})
            if ev:
                best_of[(i, j)] = min(ev, key=lambda e: e["resid"])
                best_of[(i, j)]["all"] = ev

    if len(best_of) < 4:
        sys.exit("매칭 평가 실패 — 일부 조합에서 후보를 얻지 못함")

    # ── 매칭 결정: 두 조합의 RMSE 합 비교 ──
    print("\n  매칭 RMSE 행렬 (행=새, 열=마모, 각 칸은 ±탐색 최소값 µm):")
    print(f"    {'':>10} {'worn[0]':>10} {'worn[1]':>10}")
    for i in range(2):
        print(f"    {'new[' + str(i) + ']':>10} "
              f"{best_of[(i, 0)]['resid']:>10.1f} {best_of[(i, 1)]['resid']:>10.1f}")
    tot_direct = best_of[(0, 0)]["resid"] + best_of[(1, 1)]["resid"]
    tot_swap = best_of[(0, 1)]["resid"] + best_of[(1, 0)]["resid"]
    print(f"    direct(0-0,1-1) 합 = {tot_direct:.1f}µm   swap(0-1,1-0) 합 = {tot_swap:.1f}µm")

    mode = "swap" if args.swap else args.pairing
    if mode == "auto":
        mode = "swap" if tot_swap < tot_direct else "direct"
        margin = abs(tot_swap - tot_direct)
        lo = min(tot_swap, tot_direct)
        note = "" if margin > 0.3 * lo else "  [주의: 두 매칭 차이 작음 — 날 대응 불확실]"
        print(f"    → 자동 선택: {mode} (차이 {margin:.1f}µm){note}")
    else:
        print(f"    → 지정: {mode}")
    combos = [(0, 0), (1, 1)] if mode == "direct" else [(0, 1), (1, 0)]

    report = []
    top_pairs = []                 # 위(top) 경계 후퇴 그림용 (두 pair 공통 축으로 루프 뒤 생성)
    for k, (i, j) in enumerate(combos):
        tag = f"pair{k + 1}"
        n = news[i]
        best = best_of[(i, j)]
        print(f"\n  {tag}: new[{i}] {n['file']}  ↔  worn[{j}] 후보")
        print(f"    {'off':>4} {'file':>13} {'RMSEµm':>10} {'후퇴평균µm':>10} {'후퇴최대µm':>10}")
        cands = sorted(best["all"], key=lambda e: e["d"])
        save_new_view(out_dir, tag, n)          # 후보 폴더에 새 공구 1장 추가
        for c in cands:
            mark = " ←선택" if c["d"] == best["d"] else ""
            print(f"    {c['d']:>+4} {c['wn']['file']:>13} {c['resid']:>10.1f} "
                  f"{c['vbmean']:>10.1f} {c['vbmax']:>10.1f}{mark}")
            save_candidate_views(out_dir, tag, n, c["wn"], c["aln"], c["d"],
                                 c["resid"], c["vbmean"], c["vbmax"])
        resid_sorted = sorted(c["resid"] for c in cands)
        sep = (resid_sorted[1] - resid_sorted[0]) if len(resid_sorted) > 1 else float("inf")
        warn = []
        if sep < 1.0:
            warn.append("2위와 차이<1µm — 위상 애매")
        if abs(best["d"]) == args.search:
            warn.append(f"최소가 탐색 경계(off{best['d']:+d}) — --search 확대 필요")
        flag = ("  [주의: " + " / ".join(warn) + "]") if warn else ""
        print(f"  → 선택: off {best['d']:+d} ({best['wn']['file']}), "
              f"RMSE {best['resid']:.1f}µm (2위와 +{sep:.1f}µm){flag}")

        wn = best["wn"]
        aln = best["aln"]
        s = aln["s"]
        s_body, _ = scale_ratio(n, wn, "body")
        print(f"    배율 s={s:.5f} ({aln['s_src']} 기준)  |  몸통 기준이면 {s_body:.5f} "
              f"→ 차이 {(s_body-s)*100:+.3f}% = 반경 705px 에서 {(s_body-s)*705*um:+.1f}µm")
        if abs(1 - s) > 0.02:
            print(f"  [경고] 배율비 {s:.4f} — 차이 큼")
        vb_max, vb_mean, retreat_um = wear_stats(aln, um)
        common, rn_i, rw_i = aln["common"], aln["rn"], aln["rw"]
        u_lo, u_hi = aln["u_lo"], aln["u_hi"]
        report.append((tag, n["file"], wn["file"], best["d"], vb_max, vb_mean, u_lo, u_hi))
        print(f"  {tag}: 후퇴 최대 {vb_max:.1f}µm / 평균 {vb_mean:.1f}µm "
              f"(u={u_lo}~{u_hi}px = {u_lo * um / 1000:.2f}~{u_hi * um / 1000:.2f}mm)")

        # ── 위(top) 경계 후퇴 데이터 수집 ──
        tun, tnt, _ = outline_yy(n["det"]["final"], n["tip"], n["axis"], gray=n["gray"])
        tuw, twt, _ = outline_yy(wn["det"]["final"], wn["tip"], wn["axis"], gray=wn["gray"])
        tc = np.intersect1d(tun, tuw)
        tn_y = tnt[np.searchsorted(tun, tc)]
        tw_y = twt[np.searchsorted(tuw, tc)] * s
        top_raw = (tw_y - tn_y) * um
        top_pairs.append({"tag": tag, "nf": n["file"], "wf": wn["file"], "u": tc,
                          "tn": tn_y, "tw": tw_y, "raw": top_raw,
                          "sm": _smooth(top_raw, 15)})

        # ── 위상 탐색 결과 플롯 ──
        fig, axp = plt.subplots(figsize=(8, 5))
        ds = [c["d"] for c in cands]
        axp.plot(ds, [c["resid"] for c in cands], "o-", color="#4477aa",
                 label=f"외곽선 y차 RMSE (선택 지표, 날끝 {args.tip_skip}px 제외)")
        axp.plot(ds, [c["vbmean"] for c in cands], "s--", color="#ee7733", label="후퇴 평균 (측정)")
        axp.axvline(best["d"], color="#cc3311", lw=1.2, label=f"선택 off {best['d']:+d}")
        axp.set_xlabel("마모공구 프레임 오프셋 (frame)")
        axp.set_ylabel("µm")
        axp.set_title(f"{tag}: 마모공구 위상 탐색 (외곽선 y차 RMSE 최소 = 정답 위상)")
        axp.set_xticks(ds)
        axp.legend()
        axp.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_phasematch.png", dpi=120)
        plt.close(fig)

        # ── 에지 겹침 + 후퇴 프로파일 ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(common * um / 1000, rn_i * um / 1000, "-", color="#4477aa", label=f"새 ({n['file']})")
        ax1.plot(common * um / 1000, rw_i * um / 1000, "-", color="#cc3311", label=f"마모 ({wn['file']})")
        ax1.set_xlabel("코너에서 축방향 거리 (mm)")
        ax1.set_ylabel("축에서 에지까지 반경 (mm)")
        ax1.set_title(f"{tag}: 에지 반경 프로파일 ({n['side']} edge)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.plot(common * um / 1000, retreat_um, "-", color="#ee7733")
        ax2.axvspan(u_lo * um / 1000, u_hi * um / 1000, color="#44aa77", alpha=0.12, label="통계 구간")
        ax2.legend(loc="upper right")
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set_xlabel("코너에서 축방향 거리 (mm)")
        ax2.set_ylabel("후퇴 (µm, +=마모)")
        ax2.set_title(f"{tag}: 마모 후퇴  최대 {vb_max:.1f}µm / 평균 {vb_mean:.1f}µm")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_retreat.png", dpi=120)
        plt.close(fig)

        # ── 개별 외곽선 ──
        for role, dat, col in [("new", n, (255, 120, 0)), ("worn", wn, (0, 0, 255))]:
            # ── 외곽선 없는 크롭 원본 (선택 프레임 확인용) ──
            plain = dat["gray"]
            if sam_win:
                plain = plain[:, sam_win[0]:min(sam_win[1], plain.shape[1])]
            plain_s = cv2.resize(plain, None, fx=0.6, fy=0.6,
                                 interpolation=cv2.INTER_AREA)
            cv2.imencode(".png", plain_s)[1].tofile(
                str(out_dir / f"{tag}_{role}_{Path(dat['file']).stem}_photo.png"))

            ind = cv2.cvtColor(dat["gray"], cv2.COLOR_GRAY2BGR)
            cnts, _ = cv2.findContours(dat["det"]["final"].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ind, cnts, -1, col, 3)
            cv2.circle(ind, dat["tip"], 20, (0, 255, 255), 3)
            if sam_win:                       # 결과 사진도 SAM 크롭 창 그대로
                ind = ind[:, sam_win[0]:min(sam_win[1], ind.shape[1])]
                sc = 0.6
            elif u_max_rmse:
                ind = crop_view(ind, dat["tip"], dat["axis"], u_max_rmse)
                sc = 0.6
            else:
                sc = 0.35
            cv2.putText(ind, f"{tag} {role}: {dat['file']}", (20, ind.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
            ind_s = cv2.resize(ind, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            cv2.imencode(".png", ind_s)[1].tofile(
                str(out_dir / f"{tag}_{role}_{Path(dat['file']).stem}_outline.png"))

        # ── 겹침 오버레이 (배경=새 공구, 새=파랑, 마모=빨강 정렬) ──
        H, W = n["gray"].shape
        canvas = cv2.cvtColor(n["gray"], cv2.COLOR_GRAY2BGR)
        M = np.float32([[s, 0, n["tip"][0] - s * wn["tip"][0]],
                        [0, s, n["axis"] - s * wn["axis"]]])
        wm = cv2.warpAffine(wn["det"]["final"].astype(np.uint8), M, (W, H)) > 0
        for m, col in [(n["det"]["final"], (255, 120, 0)), (wm, (0, 0, 255))]:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, col, 3)
        if sam_win:                           # 결과 사진도 SAM 크롭 창 그대로
            canvas = canvas[:, sam_win[0]:min(sam_win[1], canvas.shape[1])]
            sc = 0.6
        elif u_max_rmse:
            canvas = crop_view(canvas, n["tip"], n["axis"], u_max_rmse)
            sc = 0.6
        else:
            sc = 0.35
        _t = f"{tag} blue=new({n['file']}) red=worn({wn['file']}, off{best['d']:+d})"
        yy = canvas.shape[0] - 25
        cv2.putText(canvas, _t, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 9)
        cv2.putText(canvas, _t, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        small = cv2.resize(canvas, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        cv2.imencode(".png", small)[1].tofile(str(out_dir / f"{tag}_overlay.png"))

    # ── 위(top) 경계 후퇴 그림 — 두 pair 공통 축, 표시 0~3mm, 잘림 없음 ──
    if top_pairs:
        x_show = 3.0
        vis = [p["sm"][p["u"] * um / 1000 <= x_show] for p in top_pairs]
        ylo = min(0.0, min(v.min() for v in vis)) - 5
        yhi = max(v.max() for v in vis) * 1.08 + 5
        prof = [np.concatenate([p["tn"], p["tw"]]) * um / 1000 for p in top_pairs]
        p_hi = min(v.min() for v in prof) - 0.1
        p_lo = max(v.max() for v in prof) + 0.1
        for p in top_pairs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            x = p["u"] * um / 1000
            ax1.plot(x, p["tn"] * um / 1000, "-", color="#4477aa", label=f"새 ({p['nf']})")
            ax1.plot(x, p["tw"] * um / 1000, "-", color="#cc3311",
                     label=f"마모 ({p['wf']}, s보정)")
            ax1.set_xlim(-0.1, 4.15)
            ax1.set_ylim(p_lo, p_hi)              # 그림 위 = 실제 위
            ax1.set_xlabel("코너에서 축방향 거리 (mm)")
            ax1.set_ylabel("위 경계 위치 (축 기준 mm, 그림 위=실제 위)")
            ax1.set_title(f"{p['tag']}: 위(top) 경계 프로파일")
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax2.plot(x, p["raw"], "-", color="#cccccc", lw=0.8, label="원시")
            ax2.plot(x, p["sm"], "-", color="#ee7733", lw=1.8, label="평활(이동평균 15열)")
            ax2.axhline(0, color="gray", lw=0.8)
            ax2.set_xlim(-0.05, x_show)
            ax2.set_ylim(ylo, yhi)
            ax2.set_xlabel("코너에서 축방향 거리 (mm)")
            ax2.set_ylabel("위 경계 후퇴 (µm, +=마모가 더 아래)")
            ax2.set_title(f"{p['tag']}: 위 경계 후퇴")
            ax2.legend(loc="upper right")
            ax2.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"{p['tag']}_top_retreat.png", dpi=120)
            plt.close(fig)

    print("\n===== 요약 =====")
    for tag, nf, wf, d, vb_max, vb_mean, u_lo, u_hi in report:
        print(f"{tag}  새 {nf} vs 마모 {wf}(off{d:+d}):  후퇴 최대 {vb_max:.1f}µm, "
              f"평균 {vb_mean:.1f}µm  (구간 {u_lo * um / 1000:.2f}~{u_hi * um / 1000:.2f}mm)")
    print(f"결과 저장: {out_dir}\\pair*_phasematch.png, pair*_retreat.png, "
          f"pair*_top_retreat.png, pair*_overlay.png")


if __name__ == "__main__":
    main()
