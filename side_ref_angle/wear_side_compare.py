# -*- coding: utf-8 -*-
"""옆날 마모 측정: 새 공구 vs 마모 공구, 피크 가운데 각도(≈90°/270°) 두 장씩 비교.

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
    [--diam 8.0] [--swap] [--search 2] [--out wear_side_out]
--swap: 2날 180° 모호성 해소용 — 새/마모 프레임 짝을 반대로 묶음
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


def auto_click(tool, tip, axis_y, side="bottom"):
    """날끝 오른쪽 150~450px 대역에서 지정한 쪽(기본 아래) 웨지의 무게중심."""
    ys, xs = np.nonzero(tool)
    band = (xs >= tip[0] + 150) & (xs <= tip[0] + 450)
    sel = band & ((ys > axis_y) if side == "bottom" else (ys < axis_y))
    if not sel.any():
        raise RuntimeError(f"{side}쪽 날 웨지를 찾지 못함")
    return (int(xs[sel].mean()), int(ys[sel].mean())), side


def edge_profile(mask, tip, axis_y, side, length=1400):
    """코너 기준 축방향 거리 u에서의 에지 반경 r(u)[px]. side='bottom'|'top'."""
    h, w = mask.shape
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


def outline_yy(mask, tip, axis_y, length=1400):
    """각 x(코너 기준 u)에서 외곽선의 y 두 값(위/아래)을 축 기준으로 반환.

    반환: u[], y_top[], y_bot[]  (모두 축 중심선 기준 상대 y, px)
    """
    h, w = mask.shape
    us, tops, bots = [], [], []
    for u in range(0, length):
        x = tip[0] + u
        if x >= w:
            break
        ys = np.flatnonzero(mask[:, x])
        if len(ys) == 0:
            continue
        us.append(u)
        tops.append(ys.min() - axis_y)
        bots.append(ys.max() - axis_y)
    return np.array(us), np.array(tops, dtype=float), np.array(bots, dtype=float)


def outline_rmse(n, wn, s, tip_skip, um):
    """새/마모 외곽선의 y값 차이 RMSE (µm). 날끝 구간(u < tip_skip)은 제외.

    각 x마다 외곽선 y가 위/아래 2개씩 있으므로 둘 다 사용한다.
    마모 쪽은 몸통 배율 s를 곱해 새 공구 좌표계로 맞춘 뒤 비교.
    """
    un, nt, nb = outline_yy(n["det"]["final"], n["tip"], n["axis"])
    uw, wt, wb = outline_yy(wn["det"]["final"], wn["tip"], wn["axis"])
    common = np.intersect1d(un, uw)
    common = common[common >= tip_skip]
    if len(common) < 50:
        return float("inf"), 0
    i_n = np.searchsorted(un, common)
    i_w = np.searchsorted(uw, common)
    d_top = nt[i_n] - wt[i_w] * s
    d_bot = nb[i_n] - wb[i_w] * s
    rmse_px = float(np.sqrt(np.mean(np.concatenate([d_top, d_bot]) ** 2)))
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
    u, r = edge_profile(det["final"], det["tip"], axis_y, sd)
    return {"gray": gray, "det": det, "tip": det["tip"], "axis": axis_y,
            "body_h": body_h, "side": sd, "file": files[idx], "idx": idx,
            "u": u, "r": r}


def align_profiles(n, wn):
    """새/마모 프로파일을 코너·축 정렬 + 몸통 배율 s로 공통 u에 맞춤.

    반환: dict(s, common, rn, rw, u_lo, u_hi) 또는 None.
    u_lo..u_hi = 날끝(코너 램프)과 마스크 끝을 제외한 몸통 구간.
    """
    s = n["body_h"] / wn["body_h"]
    common = np.intersect1d(n["u"], wn["u"])
    if len(common) < 50:
        return None
    rn = n["r"][np.searchsorted(n["u"], common)]
    rw = wn["r"][np.searchsorted(wn["u"], common)] * s
    plateau = float(np.median(rn[common >= common.max() - 400]))
    reach = np.flatnonzero(rn >= 0.9 * plateau)
    u_lo = int(common[reach[0]]) + 30 if len(reach) else 30
    u_hi = int(common.max()) - 100
    return {"s": s, "common": common, "rn": rn, "rw": rw, "u_lo": u_lo, "u_hi": u_hi}


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
    ap.add_argument("--swap", action="store_true", help="새/마모 프레임 짝 반대로")
    ap.add_argument("--side", choices=["bottom", "top"], default="bottom",
                    help="측정할 절삭날 쪽 (기본 bottom)")
    ap.add_argument("--search", type=int, default=2,
                    help="마모공구 위상 탐색 반경(프레임, 기본 2 = 총 5장)")
    ap.add_argument("--tip-skip", type=int, default=300,
                    help="RMSE 계산에서 제외할 날끝 구간 길이(px, 기본 300). "
                         "날끝은 마모로 새 공구와 매칭이 잘 안 되므로 제외")
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

    worn_mids = sel["worn"]["mids"][::-1] if args.swap else sel["worn"]["mids"]
    pairs = list(zip(sel["new"]["mids"], worn_mids))
    worn_files = sel["worn"]["files"]

    predictor = get_predictor()
    print("SAM 로드 완료")

    report = []
    for k, (ni, wi) in enumerate(pairs):
        tag = f"pair{k + 1}"
        n = process_frame(args.new_dir, sel["new"]["files"], ni, predictor, args.thresh, args.side)
        if n is None:
            print(f"  [{tag}] 새 공구 프레임 검출 실패 — 건너뜀")
            continue
        print(f"  {tag}/new: {n['file']}  날끝={n['tip']}  축y={n['axis']:.0f}  몸통H={n['body_h']:.0f}px")

        # ── 마모공구 위상 탐색: ±search장 후보를 몸통 정합잔차로 평가 ──
        print(f"  {tag}/worn 위상 탐색 (idx {wi} ±{args.search}):")
        print(f"    {'off':>4} {'idx':>4} {'file':>13} {'RMSEµm':>10} {'후퇴평균µm':>10} {'후퇴최대µm':>10}")
        cands = []
        for d in range(-args.search, args.search + 1):
            wj = wi + d
            if not (0 <= wj < len(worn_files)):
                continue
            wn = process_frame(args.worn_dir, worn_files, wj, predictor, args.thresh, args.side)
            if wn is None:
                continue
            aln = align_profiles(n, wn)
            if aln is None:
                print(f"    {d:>+4} {wj:>4} {wn['file']:>13}  정렬 실패")
                continue
            rmse, npts = outline_rmse(n, wn, aln["s"], args.tip_skip, um)
            vbmax, vbmean, _ = wear_stats(aln, um)
            cands.append({"d": d, "wn": wn, "aln": aln, "resid": rmse,
                          "vbmax": vbmax, "vbmean": vbmean, "npts": npts})
            print(f"    {d:>+4} {wj:>4} {wn['file']:>13} {rmse:>10.1f} {vbmean:>10.1f} {vbmax:>10.1f}")
            save_candidate_views(out_dir, tag, n, wn, aln, d, rmse, vbmean, vbmax)
        if not cands:
            print(f"  [{tag}] 마모공구 후보 없음 — 건너뜀")
            continue

        best = min(cands, key=lambda c: c["resid"])
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
        if abs(1 - s) > 0.02:
            print(f"  [경고] 몸통 높이비 {s:.4f} — 배율 차이 큼")
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
        axp.set_title(f"{tag}: 마모공구 위상 탐색 (몸통 정합 최소 = 정답 위상)")
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
            cv2.putText(ind, f"{tag} {role}: {dat['file']}", (60, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 4)
            ind_s = cv2.resize(ind, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
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
        _t = f"{tag} bg=new({n['file']}) blue=new red=worn(off{best['d']:+d})"
        cv2.putText(canvas, _t, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 10)
        cv2.putText(canvas, _t, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        small = cv2.resize(canvas, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
        cv2.imencode(".png", small)[1].tofile(str(out_dir / f"{tag}_overlay.png"))

    print("\n===== 요약 =====")
    for tag, nf, wf, d, vb_max, vb_mean, u_lo, u_hi in report:
        print(f"{tag}  새 {nf} vs 마모 {wf}(off{d:+d}):  후퇴 최대 {vb_max:.1f}µm, "
              f"평균 {vb_mean:.1f}µm  (구간 {u_lo * um / 1000:.2f}~{u_hi * um / 1000:.2f}mm)")
    print(f"결과 저장: {out_dir}\\pair*_phasematch.png, pair*_retreat.png, pair*_overlay.png")


if __name__ == "__main__":
    main()
