# -*- coding: utf-8 -*-
"""날 영역 인식 모듈 — "인식 → 측정 범위 한정" (botface_pipeline_v2 에서 사용).

문제: 측정(방사형 프로파일·스네이크)이 고정 범위(±38°, 밴드 0.30r)를 훑어서
날 끝 밖 배경/보케까지 에지로 판단함.
해법: 정렬·수직화된 기준 이미지에서 날 영역(길쭉한 세로 기둥)을 분할해
날별 각도 섹터와 세로 범위를 뽑고, 측정을 그 안으로 한정한다.

인식 방식 (이 프로젝트에서 검증된 순서):
  1) 기하 시드 — 수직화된 캐노니컬 포즈에서는 날=수직축 방향, 비날=대각선 방향이
     보장됨 → 시드 배치에 학습/튜닝 불필요, 노출 변화에 강건
  2) cv2.watershed 로 시드를 실제 면 경계에 스냅
  3) 기둥(두 라벨 합집합)을 축 기준 위/아래로 분할 = 날1/날2
  4) (선택) LLM(Claude) 검증 — 오버레이 이미지를 보고 인식이 맞는지 판정.
     ANTHROPIC_API_KEY 가 설정되고 anthropic SDK 가 있으면 활성화, 아니면 생략.

사용자 확정 정의(2026-07-20): 날 = 길쭉한 세로 기둥(줄무늬 랜드 밴드),
축 중심 위쪽 절반 = 날1, 아래쪽 절반 = 날2. 부채꼴(여유면)·삼각형(플루트)은 날 아님.
"""
import os
import json
import base64

import cv2
import numpy as np

# 진단 세션 내 LLM 검증 결과 캐시 (기준 프레임당 1회만 호출)
_LLM_CACHE = {}


# ---------------- 1) 기하 시드 + 워터셰드 분할 ----------------
def _gradmag(gray):
    f = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def recognize(gray, axis, r):
    """수직화된 기준 이미지에서 날1(위)/날2(아래) 마스크 반환. 실패 시 None.
    핵심: 날을 원(OD) 안으로 가정하지 않는다. 기둥이 위/아래로 끝까지 번지게 한 뒤
    행별 '날 면 증거'(줄무늬 선명도)의 경향을 따라가 실제 끝점을 판별한다.
    반환 dict: {'mask1','mask2','column','tip1','tip2','evidence1','evidence2'}"""
    H, W = gray.shape[:2]
    cx, cy = float(axis[0]), float(axis[1])

    markers = np.zeros((H, W), np.int32)
    seed_r = max(6, int(0.012 * r))
    # 배경 시드: 좌/우 가장자리 + 위/아래 가장자리 중 '기둥 통로' 밖.
    # 수직화된 포즈에서 날 기둥은 중앙 세로 통로(|x-cx|<0.35r)로만 지나가므로
    # 그 통로를 비워두면 형상이 프레임 끝까지 막힘 없이 번진다 (원/프레임에서 안 잘림).
    b = max(4, seed_r)
    ch_lo = int(max(0, cx - 0.35 * r))
    ch_hi = int(min(W, cx + 0.35 * r))
    markers[:, :b] = 3
    markers[:, -b:] = 3
    markers[:b, :ch_lo] = 3
    markers[:b, ch_hi:] = 3
    markers[-b:, :ch_lo] = 3
    markers[-b:, ch_hi:] = 3

    def put(deg, t, lab):
        th = np.radians(deg)
        x = int(round(cx + t * r * np.cos(th)))
        y = int(round(cy + t * r * np.sin(th)))
        if b <= x < W - b and b <= y < H - b:
            cv2.circle(markers, (x, y), seed_r, lab, -1)

    # 날 시드 = 이어진 세로 선(t 0.18~0.95). 점 시드는 어두운 노출에서 워터셰드가
    # 중간을 끊어 기둥을 조각낼 수 있음(0709 세션에서 실제 발생) → 선으로 연속 확보.
    for sgn in (-1, +1):
        y0 = int(round(cy + sgn * 0.18 * r))
        y1 = int(round(cy + sgn * 0.95 * r))
        cv2.line(markers, (int(cx), y0), (int(cx), y1), 1,
                 thickness=max(3, seed_r // 2))
    for deg in (45.0, 135.0, 225.0, 315.0):             # 비날 = 대각선(부채꼴/플루트)
        for t in (0.45, 0.75):
            put(deg, t, 2)
    for deg in (0.0, 180.0):                            # 좌우 수평 = 여유면(비날) 보강
        for t in (0.5, 0.8):
            put(deg, t, 2)

    bgr = cv2.cvtColor(cv2.GaussianBlur(gray, (5, 5), 0), cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers)

    column = (markers == 1).astype(np.uint8)
    column = cv2.morphologyEx(column, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if int(column.sum()) < 0.01 * np.pi * r * r:        # 기둥이 사실상 안 잡힘
        return None

    # 형상 끝까지: 마스크는 이미지에 보이는 날 형상이 끝나는 데까지 전부 포함한다
    # (사용자 지시 — 원/코너 반경에서 자르지 않음). 폭만 기둥 통로(|x-cx|≤0.35r)로
    # 제한해 흐린 배경으로의 옆 누수를 막는다. tip = 마스크가 실제로 끝나는 행.
    # 절삭 코너 반경 r(평균 영상 실측)은 측정 존 앵커로 bounds()에서 따로 쓴다.
    channel = np.zeros((H, W), np.uint8)
    channel[:, int(max(0, cx - 0.35 * r)):int(min(W, cx + 0.35 * r))] = 1
    column = (column & channel).astype(np.uint8)

    out = {'column': column}
    for name, side, tkey in (('mask1', 'top', 'tip1'),
                             ('mask2', 'bottom', 'tip2')):
        h = column.copy()
        if side == 'top':
            h[int(cy):, :] = 0
        else:
            h[:int(cy), :] = 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(h, 8)
        if n <= 1:
            return None
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8)
        ys = np.nonzero(m)[0]
        out[name] = m
        out[tkey] = int(ys.min()) if side == 'top' else int(ys.max())
    out['_gray'] = gray                                 # band_tilt(줄무늬 방향) 용
    return out


# (참고) 단일 프레임 외관 신호로 끝점을 찾는 시도는 tool_wear_v2/_region_dev/ 의
# dev_tip.py(행별 그라디언트), dev_metric.py(고주파+스텝 탐지), dev_radial.py(방사 에지)로
# 검증했고 전부 실패 — 마진이 코너 너머로 같은 질감으로 이어져 절벽이 없음.
# 따라서 끝점은 회전 평균 영상에서 실측된 코너 반경(fit_od 의 r)을 사용한다.


def band_tilt(region, axis, r):
    """날 기둥 중심선의 수직 대비 기울기[deg] 실측.
    (PCA 는 누수 덩어리에 오염돼 부정확 → 행별 밴드 중심의 중앙값으로 견고하게.)
    각 날 마스크의 몸통 존(0.30~0.78r)에서 행별 중심 x = 통로 내 픽셀 중앙값 →
    (y, x중심) 점들을 두 날에서 모아 1차 직선 적합 → 기울기 atan(dx/dy).
    반환: deg (양수 = 아래로 갈수록 +x 이동) 또는 None."""
    # 신호 = 날별 중심선 기울기 (0704 에서 육안 +5°와 일치·수렴 검증된 방식).
    # - 날별 개별 적합 필수: 두 기둥은 치즐 웹만큼 좌우 오프셋 → 합쳐 피팅하면 상쇄 오답
    # - RANSAC: 누수 블롭이 낀 행의 중앙값(이상치)을 자동 배제 (0709 발산 원인 해결)
    cy = float(axis[1])
    rng = np.random.default_rng(0)
    slopes, weights = [], []
    for name in ('mask1', 'mask2'):
        m = region.get(name)
        if m is None:
            continue
        ys, xs = np.nonzero(m)
        if len(xs) < 200:
            continue
        dist = np.abs(ys - cy)
        sel = (dist >= 0.30 * r) & (dist <= 0.78 * r)
        if int(sel.sum()) < 100:
            continue
        xs_s, ys_s = xs[sel], ys[sel]
        order = np.argsort(ys_s)
        ys_o, xs_o = ys_s[order], xs_s[order]
        uy, idx = np.unique(ys_o, return_index=True)
        py, px = [], []
        for i, y in enumerate(uy):
            seg = xs_o[idx[i]:(idx[i + 1] if i + 1 < len(uy) else len(xs_o))]
            py.append(float(y))
            px.append(float(np.median(seg)))
        if len(py) < 60:
            continue
        py = np.array(py)
        px = np.array(px)
        # RANSAC 직선 적합 (2점 표본 → 인라이어 3px)
        best_in, best_ab = 0, None
        for _ in range(200):
            i1, i2 = rng.choice(len(py), 2, replace=False)
            if abs(py[i1] - py[i2]) < 30:
                continue
            a = (px[i2] - px[i1]) / (py[i2] - py[i1])
            b = px[i1] - a * py[i1]
            inl = np.abs(px - (a * py + b)) < 3.0
            n_in = int(inl.sum())
            if n_in > best_in:
                best_in, best_ab = n_in, inl
        if best_ab is None or best_in < 0.5 * len(py):
            continue
        a = float(np.polyfit(py[best_ab], px[best_ab], 1)[0])
        slopes.append(a)
        weights.append(best_in)
    if not slopes:
        return None
    a = float(np.average(slopes, weights=weights))      # dx/dy
    return float(np.degrees(np.arctan(a)))


# ---------------- 1b) Claude 주석 기반 검출 (의미는 Claude, 정밀도는 스냅) ----------------
_ANNO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "blade_annotations.json")


def _load_annotations():
    try:
        with open(_ANNO_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_annotation(ref_folder):
    """기준 폴더 경로에 맞는 Claude 주석 반환 (없으면 None)."""
    annos = _load_annotations()
    p = str(ref_folder).replace('/', '\\')
    for key, a in annos.items():
        if key.startswith('_'):
            continue
        if key in p:
            return a
    return None


def _snap_edge(gray, pts, ytop, ybot, side, win=18):
    """주석 에지점들을 행별 보간 후 그라디언트 에지에 스냅.
    밴드 내부에 이중 톤 경계가 있을 수 있으므로 '창 안 최대'가 아니라
    '창 안 가장 바깥쪽의 유효 에지'(left=최좌측, right=최우측)를 잡는다."""
    f = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    gx = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3))
    pts = sorted(pts)
    ys = np.arange(ytop, ybot + 1)
    py = np.array([p[0] for p in pts], float)
    px = np.array([p[1] for p in pts], float)
    x0 = np.interp(ys, py, px)
    out = np.empty(len(ys))
    W = gray.shape[1]
    for i, y in enumerate(ys):
        a = int(max(0, x0[i] - win))
        b = int(min(W, x0[i] + win + 1))
        seg = gx[y, a:b]
        if not seg.size:
            out[i] = x0[i]
            continue
        # 문턱을 낮게(0.25×) — 밴드 이중톤에서 내부 경계가 강해도 외곽 경계가
        # 유효하면 항상 외곽을 잡는다 (0709 오른쪽 에지 지그재그 원인 해결)
        thr = max(10.0, 0.25 * float(seg.max()))
        cand = np.nonzero(seg >= thr)[0]
        if not len(cand):
            out[i] = x0[i]
        else:
            out[i] = a + (cand.min() if side == 'left' else cand.max())
    k = 7
    out = np.array([np.median(out[max(0, i - k):i + k + 1]) for i in range(len(out))])
    return ys, out


def recognize_annotated(gray, anno):
    """Claude 주석(blade_annotations.json)으로부터 날 마스크 생성.
    주석은 대략 경계, 정밀도는 행별 그라디언트 스냅이 담당. 워터셰드와 달리
    치즐/여유면으로 새지 않는다(의미를 Claude 가 판단했으므로)."""
    H, W = gray.shape[:2]
    out = {}
    for name, key in (('mask1', 'blade1'), ('mask2', 'blade2')):
        b = anno.get(key)
        if b is None:
            return None
        ytop, ybot = int(b['ytop']), int(min(b['ybot'], H - 1))
        ysL, xL = _snap_edge(gray, b['left'], ytop, ybot, 'left')
        ysR, xR = _snap_edge(gray, b['right'], ytop, ybot, 'right')
        m = np.zeros((H, W), np.uint8)
        for i, y in enumerate(ysL):
            l = int(min(xL[i], xR[i]))
            rr = int(max(xL[i], xR[i]))
            if rr > l:
                m[int(y), l:rr + 1] = 1
        out[name] = m
        ys = np.nonzero(m)[0]
        out['tip1' if name == 'mask1' else 'tip2'] = (
            int(ys.min()) if name == 'mask1' else int(ys.max()))
    out['column'] = (out['mask1'] | out['mask2']).astype(np.uint8)
    out['_gray'] = gray
    return out


# ---------------- 2) 날별 측정 범위 산출 ----------------
def bounds(region, axis, r, pad_deg=2.0):
    """마스크에서 측정 한계 추출.
    반환 [날1, 날2] 각 원소 dict: {'sector': (lo,hi)[deg 절대], 'rows': (ymin,ymax),
                                   'tip_rad': 감지된 날 끝 반경[px]}
    섹터는 '감지된 날 끝' 기준 끝 존(끝-0.22r ~ 끝)에 실제 날이 있는 각도 + 여유각.
    (고정 0.80r~1.02r 가 아니라 실측 끝점 기준 — 원 가정 없음)"""
    cx, cy = float(axis[0]), float(axis[1])
    res = []
    for name, tkey, center in (('mask1', 'tip1', 270.0), ('mask2', 'tip2', 90.0)):
        m = region[name]
        ys, xs = np.nonzero(m)
        if len(xs) < 50:
            res.append(None)
            continue
        rad = np.hypot(xs - cx, ys - cy)
        tipy = region.get(tkey)
        tip_rad = abs(float(tipy) - cy) if tipy is not None else float(np.percentile(rad, 99.5))
        # 측정 존은 절삭 코너 반경 r(평균 영상 실측)에 앵커 — 마스크(형상)는 그 너머까지
        # 이어지지만, 마모 후퇴 측정이 이뤄지는 곳은 코너 주변이다.
        zone = (rad >= 0.78 * r) & (rad <= 1.06 * r)
        # 누수 방어: 날은 곧은 세로 밴드 → 경계가 선명한 몸통 구간(0.4~0.75r)에서
        # 밴드 x-통로(p5~p95)를 실측하고, 코너 존에서 통로 밖 픽셀(흐린 배경 누수)은 제외
        body = (rad >= 0.40 * r) & (rad <= 0.75 * r)
        if int(body.sum()) >= 50:
            xl = float(np.percentile(xs[body], 5)) - 0.03 * r
            xr = float(np.percentile(xs[body], 95)) + 0.03 * r
            zone = zone & (xs >= xl) & (xs <= xr)
        if int(zone.sum()) < 20:
            res.append(None)
            continue
        ang = np.degrees(np.arctan2(ys[zone] - cy, xs[zone] - cx)) % 360.0
        da = (ang - center + 180.0) % 360.0 - 180.0     # 날 방향 기준 상대각
        lo, hi = float(da.min()) - pad_deg, float(da.max()) + pad_deg
        res.append({'sector': (center + lo, center + hi),
                    'rows': (int(ys.min()), int(ys.max())),
                    'tip_rad': tip_rad})
    return res


# ---------------- 3) 오버레이 (검증·기록용) ----------------
def overlay(gray, region, axis, r):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for name, col in (('mask1', (60, 60, 255)), ('mask2', (60, 190, 255))):
        m = region[name]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        ov = vis.copy()
        cv2.drawContours(ov, [c], -1, col, -1)
        vis = cv2.addWeighted(ov, 0.28, vis, 0.72, 0)
        cv2.drawContours(vis, [c], -1, col, 3)
    cv2.circle(vis, (int(axis[0]), int(axis[1])), int(r), (0, 220, 220), 1)
    cv2.putText(vis, "red=blade1(top)  orange=blade2(bottom)", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


# ---------------- 4) LLM(Claude) 검증 — 선택 레이어 ----------------
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean",
               "description": "오버레이가 두 날(세로 기둥의 위/아래 절반)을 정확히 덮으면 true"},
        "problem": {"type": "string",
                    "description": "문제가 있으면 무엇이 잘못 포함/누락됐는지 한 문장. 없으면 빈 문자열"},
    },
    "required": ["ok", "problem"],
    "additionalProperties": False,
}

_VERIFY_PROMPT = (
    "2날 엔드밀을 아래에서 찍은 밑면 사진에 날 영역 인식 결과를 겹친 이미지입니다.\n"
    "날의 정의: 연삭 줄무늬가 있는 길쭉한 세로 기둥이 날이며, 사진 중심 기준 위쪽 절반이 "
    "날1(빨강), 아래쪽 절반이 날2(주황)입니다. 어두운 큰 부채꼴(여유면)과 밝은 부채꼴(플루트)은 "
    "날이 아니므로 색칠되면 안 됩니다. 노란 원은 공구 외경입니다.\n"
    "오버레이가 이 정의에 맞게 두 날만 정확히 덮는지 판정해 JSON으로 답하세요."
)


def llm_verify(overlay_bgr, cache_key=None, log=print, model="claude-opus-4-8"):
    """오버레이 이미지를 Claude 에 보내 인식 결과를 검증.
    반환 dict {'ok','problem'} 또는 None(비활성/실패 — 파이프라인은 인식 결과를 그대로 씀).
    비활성 조건: ANTHROPIC_API_KEY 미설정 또는 anthropic SDK 미설치."""
    if cache_key is not None and cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("    [영역] LLM 검증 생략 (ANTHROPIC_API_KEY 미설정 — 기하 인식만 사용)")
        return None
    try:
        import anthropic
    except ImportError:
        log("    [영역] LLM 검증 생략 (anthropic SDK 미설치: venv\\Scripts\\pip install anthropic)")
        return None
    try:
        h, w = overlay_bgr.shape[:2]
        scale = min(1.0, 1024.0 / max(h, w))            # 토큰 절약용 축소
        img = cv2.resize(overlay_bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA) if scale < 1.0 else overlay_bgr
        ok_enc, buf = cv2.imencode(".png", img)
        if not ok_enc:
            return None
        b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": _VERIFY_PROMPT},
                ],
            }],
        )
        if resp.stop_reason == "refusal":
            log("    [영역] LLM 검증 실패(refusal) — 기하 인식 결과 사용")
            return None
        text = next((b.text for b in resp.content if b.type == "text"), "")
        verdict = json.loads(text)
        log(f"    [영역] LLM 검증: {'OK' if verdict.get('ok') else '문제 발견'}"
            + (f" — {verdict.get('problem')}" if verdict.get("problem") else ""))
        if cache_key is not None:
            _LLM_CACHE[cache_key] = verdict
        return verdict
    except Exception as e:                              # 네트워크/429 등 — 진단은 계속
        log(f"    [영역] LLM 검증 오류({type(e).__name__}) — 기하 인식 결과 사용")
        return None
