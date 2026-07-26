# -*- coding: utf-8 -*-
"""옆날 마모 측정: 새 공구 vs 마모 공구, 피크 가운데 각도(≈90°/270°) 두 장씩 비교.

방법
----
1. 두 데이터셋 각각 find_ref_angle.py의 CSV에서 피크 2개(0°/180°)를 플래토
   중심(최대값-5px 이내 프레임들의 중앙)으로 찾고, 그 가운데 두 프레임을
   마모 판별 사진으로 선정 (mid_a = 두 피크의 중점, mid_b = mid_a ± 반주기).
2. 각 사진에서 SAM 날 검출(sam_blade_detect.detect_blade, 날끝 복원 포함).
   클릭점은 자동: 날끝 오른쪽 150~450px 대역에서 축 위/아래 중 픽셀이 많은
   쪽(=카메라를 향한 날 웨지)의 무게중심.
3. 정렬/스케일: 몸통(오른쪽 400~100px 대역) 실루엣의 중심선 y로 상하 정렬,
   몸통 높이비로 배율 확인. µm/px = 직경 / 새 공구 레퍼런스 피크 y_diff.
4. 후퇴 측정: 날끝(코너)을 원점으로, 코너에서 축방향 거리 u에서의 에지
   반경 r(u) = |에지 y - 축 y| 프로파일을 새/마모 비교.
   retreat(u) = r_new(u) - r_worn(u)  (+ = 마모로 에지가 축쪽으로 후퇴)

사용법
------
python wear_side_compare.py --new-dir "...\Initial\옆" --worn-dir "...\Test44\Toolwear\옆"
    --new-csv ref_angle_out_v6\ref_angle_scores.csv
    --worn-csv ref_angle_out_test44\ref_angle_scores.csv
    [--diam 8.0] [--swap] [--out wear_side_out]
--swap: 2날 180° 모호성 해소용 — 새/마모 프레임 짝을 반대로 묶음
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
    """계곡에서 몇 장 뒤에 날끝이 고립(M=0 2연속)되는지 반환 (경계 오프셋).

    max_scan 안에 못 찾으면 12장으로 폴백.
    """
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
    """날끝 오른쪽 150~450px 대역에서 지정한 쪽(기본 아래) 웨지의 무게중심.

    2날 공구는 θ와 θ+180°가 같은 포즈(날만 교대)이므로, 피크 가운데 각도에서
    카메라를 향한 옆 절삭날은 두 판별 프레임 모두 같은 쪽(아래)에 온다.
    반대쪽(위)은 끝면/뒷날 실루엣이라 마모 측정 대상이 아님.
    """
    ys, xs = np.nonzero(tool)
    band = (xs >= tip[0] + 150) & (xs <= tip[0] + 450)
    sel = band & ((ys > axis_y) if side == "bottom" else (ys < axis_y))
    if not sel.any():
        raise RuntimeError(f"{side}쪽 날 웨지를 찾지 못함")
    return (int(xs[sel].mean()), int(ys[sel].mean())), side


def edge_profile(mask, tip, axis_y, side, length=1200):
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

    # 판별 각도 = 계곡 + 공통 오프셋 (고립 경계 최댓값 + 여유 2장)
    # — 반대날 끝면 덩어리가 날끝에서 비켜나 절삭날이 배경에 고립되는 각도
    offsets = {}
    for name in sel:
        for v in sel[name]["valleys"]:
            offsets[(name, v)] = isolation_offset(dirs[name], sel[name]["files"], v, args.thresh)
    common = max(offsets.values()) + 2
    for (name, v), k in offsets.items():
        print(f"  [{name}] 계곡 idx {v}: 고립 경계 +{k}장")
    print(f"  → 공통 오프셋 +{common}장 적용")
    for name in sel:
        sel[name]["mids"] = [v + common for v in sel[name]["valleys"]]
        f = sel[name]["files"]
        print(f"[{name}] 판별 프레임 idx {sel[name]['mids'][0]}({f[sel[name]['mids'][0]]}), "
              f"{sel[name]['mids'][1]}({f[sel[name]['mids'][1]]})")

    um = args.diam * 1000.0 / sel["new"]["ref_ydiff"]
    print(f"스케일: {um:.3f} µm/px  (직경 {args.diam}mm / 새 공구 피크 y_diff {sel['new']['ref_ydiff']:.0f}px)")

    worn_mids = sel["worn"]["mids"][::-1] if args.swap else sel["worn"]["mids"]
    pairs = list(zip(sel["new"]["mids"], worn_mids))

    predictor = get_predictor()
    print("SAM 로드 완료")

    report = []
    for k, (ni, wi) in enumerate(pairs):
        tag = f"pair{k + 1}"
        data = {}
        for role, folder, idx, files in [("new", args.new_dir, ni, sel["new"]["files"]),
                                         ("worn", args.worn_dir, wi, sel["worn"]["files"])]:
            path = Path(folder) / files[idx]
            gray = imread_gray(path)
            tool, tip = tool_silhouette(gray, args.thresh)
            axis_y, body_h = body_axis(tool, gray.shape[1])
            click, side = auto_click(tool, tip, axis_y, args.side)
            det = detect_blade(gray, click, predictor, args.thresh)
            data[role] = {"gray": gray, "det": det, "tip": det["tip"], "axis": axis_y,
                          "body_h": body_h, "side": side, "file": files[idx]}
            print(f"  {tag}/{role}: {files[idx]}  날끝={det['tip']}  축y={axis_y:.0f}  "
                  f"몸통H={body_h:.0f}px  날방향={side}  클릭={click}")

        n, wn = data["new"], data["worn"]
        s = n["body_h"] / wn["body_h"]
        if abs(1 - s) > 0.02:
            print(f"  [경고] 몸통 높이비 {s:.4f} — 배율 차이 큼")
        un, rn = edge_profile(n["det"]["final"], n["tip"], n["axis"], n["side"])
        uw, rw = edge_profile(wn["det"]["final"], wn["tip"], wn["axis"], wn["side"])
        common = np.intersect1d(un, uw)
        rn_i = rn[np.searchsorted(un, common)]
        rw_i = rw[np.searchsorted(uw, common)] * s
        retreat_um = (rn_i - rw_i) * um
        # 통계 구간: 옆날 본체(플래토)에 도달한 뒤부터 — 앞쪽 V자(축 교차) 구간과
        # 마스크 끝 아티팩트(마지막 100px)는 측정 대상이 아님
        plateau = float(np.median(rn_i[common >= common.max() - 400]))
        reach = np.flatnonzero(rn_i >= 0.9 * plateau)
        u_lo = int(common[reach[0]]) + 30 if len(reach) else 30
        u_hi = int(common.max()) - 100
        zone = (common >= u_lo) & (common <= u_hi)
        vb_max = float(retreat_um[zone].max()) if zone.any() else float("nan")
        vb_mean = float(retreat_um[zone].mean()) if zone.any() else float("nan")
        report.append((tag, n["file"], wn["file"], vb_max, vb_mean, u_lo, u_hi))
        print(f"  {tag}: 후퇴 최대 {vb_max:.1f}µm / 평균 {vb_mean:.1f}µm "
              f"(u={u_lo}~{u_hi}px = {u_lo * um / 1000:.2f}~{u_hi * um / 1000:.2f}mm)")

        # ── 시각화: 에지 겹침 + 후퇴 프로파일 ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(common * um / 1000, rn_i * um / 1000, "-", color="#4477aa", label=f"새 ({n['file']})")
        ax1.plot(common * um / 1000, rw_i * um / 1000, "-", color="#cc3311", label=f"마모 ({wn['file']})")
        ax1.set_xlabel("코너에서 축방향 거리 (mm)")
        ax1.set_ylabel("축에서 에지까지 반경 (mm)")
        ax1.set_title(f"{tag}: 에지 반경 프로파일 ({n['side']} edge)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.plot(common * um / 1000, retreat_um, "-", color="#ee7733")
        ax2.axvspan(u_lo * um / 1000, u_hi * um / 1000, color="#44aa77", alpha=0.12,
                    label="통계 구간")
        ax2.legend(loc="upper right")
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set_xlabel("코너에서 축방향 거리 (mm)")
        ax2.set_ylabel("후퇴 (µm, +=마모)")
        ax2.set_title(f"{tag}: 마모 후퇴  최대 {vb_max:.1f}µm / 평균 {vb_mean:.1f}µm")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_retreat.png", dpi=120)
        plt.close(fig)

        # 각 프레임 개별 외곽선 — 본인 사진 위에 그려서 검출 품질 확인 용이하게
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

        # 마스크 오버레이(배경=새 공구 사진, 새=파랑, 마모=빨강을 코너·축 정렬해 겹침)
        H, W = n["gray"].shape
        canvas = cv2.cvtColor(n["gray"], cv2.COLOR_GRAY2BGR)
        nm = n["det"]["final"]
        # 마모 마스크를 새 좌표로: 코너 x 정렬 + 축 y 정렬 + 배율 s
        M = np.float32([[s, 0, n["tip"][0] - s * wn["tip"][0]],
                        [0, s, n["axis"] - s * wn["axis"]]])
        wm = cv2.warpAffine(wn["det"]["final"].astype(np.uint8), M, (W, H)) > 0
        for m, col in [(nm, (255, 120, 0)), (wm, (0, 0, 255))]:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, col, 3)
        cv2.putText(canvas, f"{tag} bg=new({n['file']}) blue=new red=worn(aligned)", (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        small = cv2.resize(canvas, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
        cv2.imencode(".png", small)[1].tofile(str(out_dir / f"{tag}_overlay.png"))

    print("\n===== 요약 =====")
    for tag, nf, wf, vb_max, vb_mean, u_lo, u_hi in report:
        print(f"{tag}  새 {nf} vs 마모 {wf}:  후퇴 최대 {vb_max:.1f}µm, "
              f"평균 {vb_mean:.1f}µm  (구간 {u_lo * um / 1000:.2f}~{u_hi * um / 1000:.2f}mm)")
    print(f"결과 저장: {out_dir}\\pair*_retreat.png, pair*_overlay.png")


if __name__ == "__main__":
    main()
