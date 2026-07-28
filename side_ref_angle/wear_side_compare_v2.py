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


def process_frame(folder, files, idx, predictor, thresh, side):
    """한 프레임 SAM 검출 + 기하/프로파일. 실패 시 None."""
    gray = imread_gray(Path(folder) / files[idx])
    if gray is None:
        return None
    try:
        tool, tip0 = tool_silhouette(gray, thresh)
        axis_y, body_h = body_axis(tool, gray.shape[1])
        click, sd = auto_click(tool, tip0, axis_y, side)
        det = detect_blade(gray, click, predictor, thresh)
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
    ap.add_argument("--search", type=int, default=2,
                    help="마모공구 위상 탐색 반경(프레임, 기본 2 = 총 5장)")
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
        sel[name] = {"files": files, "valleys": [ma, mb], "ref_ydiff": ref}
        print(f"[{name}] 피크 중심 {p1:.1f}/{p2:.1f} (ref y_diff={ref:.0f})  계곡 idx {ma}, {mb}")

    offsets = {}
    for name in sel:
        for v in sel[name]["valleys"]:
            offsets[(name, v)] = isolation_offset(dirs[name], sel[name]["files"], v, args.thresh)
    common_off = max(offsets.values()) + 2
    for (name, v), k in offsets.items():
        print(f"  [{name}] 계곡 idx {v}: 고립 경계 +{k}장")
    print(f"  → 공통 오프셋 +{common_off}장 적용")
    for name in sel:
        sel[name]["mids"] = [v + common_off for v in sel[name]["valleys"]]
        f = sel[name]["files"]
        print(f"[{name}] 판별 프레임 idx {sel[name]['mids'][0]}({f[sel[name]['mids'][0]]}), "
              f"{sel[name]['mids'][1]}({f[sel[name]['mids'][1]]})")

    um = args.diam * 1000.0 / sel["new"]["ref_ydiff"]
    print(f"스케일: {um:.3f} µm/px  (직경 {args.diam}mm / 새 공구 피크 y_diff {sel['new']['ref_ydiff']:.0f}px)")

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
        n = process_frame(args.new_dir, sel["new"]["files"], ni, predictor, args.thresh, args.side)
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
            wn = process_frame(args.worn_dir, worn_files, wj, predictor, args.thresh, args.side)
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
    for k, (i, j) in enumerate(combos):
        tag = f"pair{k + 1}"
        n = news[i]
        best = best_of[(i, j)]
        print(f"\n  {tag}: new[{i}] {n['file']}  ↔  worn[{j}] 후보")
        print(f"    {'off':>4} {'file':>13} {'RMSEµm':>10} {'후퇴평균µm':>10} {'후퇴최대µm':>10}")
        cands = sorted(best["all"], key=lambda e: e["d"])
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
            ind = cv2.cvtColor(dat["gray"], cv2.COLOR_GRAY2BGR)
            cnts, _ = cv2.findContours(dat["det"]["final"].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ind, cnts, -1, col, 3)
            cv2.circle(ind, dat["tip"], 20, (0, 255, 255), 3)
            if u_max_rmse:
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
        if u_max_rmse:
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

    print("\n===== 요약 =====")
    for tag, nf, wf, d, vb_max, vb_mean, u_lo, u_hi in report:
        print(f"{tag}  새 {nf} vs 마모 {wf}(off{d:+d}):  후퇴 최대 {vb_max:.1f}µm, "
              f"평균 {vb_mean:.1f}µm  (구간 {u_lo * um / 1000:.2f}~{u_hi * um / 1000:.2f}mm)")
    print(f"결과 저장: {out_dir}\\pair*_phasematch.png, pair*_retreat.png, pair*_overlay.png")


if __name__ == "__main__":
    main()
