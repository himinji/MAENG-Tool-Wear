# -*- coding: utf-8 -*-
"""SAM(Segment Anything) 기반 날 영역 검출 — "지능으로 찾기".

워터셰드(밝기 경계만 따라가는 단순 알고리즘)의 한계를 넘기 위해, 물체를 실제로
'인식해서' 잘라내는 세그멘테이션 모델(MobileSAM, CPU 경량)을 사용한다.

방식 = 기하 프라이어 + SAM:
  - 우리가 아는 것(축 중심, 날은 수직 방향 위/아래): SAM 에 '여기가 날'이라고 점 프롬프트
  - 우리가 아는 비-날(플루트=대각선 밝은 부채꼴, 여유면=대각선 어두운 부채꼴, 치즐=중앙):
    '여기는 날 아님' 음성 점 프롬프트
  - SAM 이 텍스처/형상/문맥으로 실제 날 경계를 통째로 검출 (조명 밝든 어둡든)

기준 이미지당 1회 실행 → 마스크 PNG 저장 → 파이프라인이 재사용.

준비물(setup_sam.py 참고): torch(CPU) + mobile_sam + tool_wear_v2/code/sam_models/mobile_sam.pt
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# model_id → (registry_key, 체크포인트 파일명, import 패키지)
_MODELS = {
    'mobile': ('vit_t', 'mobile_sam.pt',        'mobile_sam'),       # 경량 CPU
    'vit_b':  ('vit_b', 'sam_vit_b_01ec64.pth', 'segment_anything'), # 정식 SAM(소)
    'vit_l':  ('vit_l', 'sam_vit_l_0b3195.pth', 'segment_anything'), # 정식 SAM(중)
    'vit_h':  ('vit_h', 'sam_vit_h_4b8939.pth', 'segment_anything'), # 정식 SAM(대, 최고정확)
}
_PREDICTORS = {}   # model_id → SamPredictor (캐시)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_predictor(model='mobile'):
    if model in _PREDICTORS:
        return _PREDICTORS[model]
    import importlib
    key, ckpt_name, pkg = _MODELS[model]
    mod = importlib.import_module(pkg)      # mobile_sam / segment_anything (API 동일)
    ckpt = os.path.join(HERE, "sam_models", ckpt_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[SAM] 체크포인트 없음: {ckpt}  (model={model})")
    sam = mod.sam_model_registry[key](checkpoint=ckpt)
    sam.to(device=_device())
    sam.eval()
    _PREDICTORS[model] = mod.SamPredictor(sam)
    return _PREDICTORS[model]


def _prompts(axis, r, tip_deg):
    """한 날에 대한 (양성점, 음성점) 좌표. tip_deg = 날 방향(270=위 / 90=아래)."""
    cx, cy = axis
    pos, neg = [], []
    th = np.radians(tip_deg)
    for t in (0.28, 0.45, 0.62, 0.80, 0.95):            # 날 중심선 따라 양성
        pos.append([cx + t * r * np.cos(th), cy + t * r * np.sin(th)])
    # 음성: 대각선(플루트/여유면) 4방향 + 중앙(치즐)
    for d in (tip_deg - 45, tip_deg + 45, tip_deg - 135, tip_deg + 135):
        for t in (0.5, 0.8):
            dr = np.radians(d)
            neg.append([cx + t * r * np.cos(dr), cy + t * r * np.sin(dr)])
    neg.append([cx, cy])                                # 축 중심(치즐 웹)
    return np.array(pos, np.float32), np.array(neg, np.float32)


def _neck_cut(m, axis, r):
    """축쪽 치즐/웹 혹 + 목 잔재 제거. 밴드는 축 근처에서 좁아졌다(목) 치즐로 부풂.
    안쪽 존(축 0.10~0.35r)의 최소폭 행(목)을 찾아 절단하되, 최소 0.16r 바닥을 둔다
    (큰 혹은 목에서, 목 근처 잔재는 바닥에서 정리. 측정은 0.45r 밖만 쓰므로 무영향)."""
    cy = float(axis[1])
    ys = np.unique(np.nonzero(m)[0])
    if len(ys) < 20:
        return m
    up = ys.max() < cy                      # True=날1(축 위), False=날2(축 아래)
    cut_rad = 0.16 * r                       # 기본 바닥
    inner = [y for y in ys if 0.10 * r <= abs(y - cy) <= 0.35 * r]
    if len(inner) >= 5:
        widths = {}
        for y in inner:
            xs = np.nonzero(m[y])[0]
            widths[y] = (xs.max() - xs.min()) if len(xs) else 0
        neck = min(widths, key=lambda y: widths[y])
        cut_rad = max(abs(neck - cy), 0.16 * r)   # 목과 바닥 중 축에서 먼 쪽
    m = m.copy()
    if up:                                   # 날1: 축(cy)에서 cut_rad 안쪽(큰 y) 제거
        m[int(cy - cut_rad):, :] = 0
    else:                                    # 날2: 축에서 cut_rad 안쪽(작은 y) 제거
        m[:int(cy + cut_rad) + 1, :] = 0
    return m


def _guard_mask(m, axis, r):
    """마스크 상식 검사: 날 밴드는 폭이 대체로 일정한 세로 띠. 어느 행이 몸통 중앙폭의
    2.2배 넘게 부풀면(치즐/플루트 오검출 혹) 그 행 이후(끝 방향)를 잘라낸다."""
    cy = float(axis[1])
    ys = np.unique(np.nonzero(m)[0])
    if len(ys) < 40:
        return m
    widths = {}
    for y in ys:
        xs = np.nonzero(m[y])[0]
        widths[y] = xs.max() - xs.min() if len(xs) else 0
    body = [widths[y] for y in ys if 0.30 * r <= abs(y - cy) <= 0.70 * r]
    if len(body) < 10:
        return m
    med = float(np.median(body))
    if med < 5:
        return m
    lim = 2.2 * med
    m = m.copy()
    # 축에서 바깥(끝) 방향으로 걸으며 폭이 lim 초과하는 첫 지점부터 제거
    inner = sorted(ys, key=lambda y: abs(y - cy))
    cut = None
    for y in inner:
        if abs(y - cy) >= 0.30 * r and widths[y] > lim:
            cut = y
            break
    if cut is not None:
        if cut > cy:
            m[cut:, :] = 0
        else:
            m[:cut + 1, :] = 0
    return m


def _snap_edges(m, gray, axis, r, win=9, thr=18.0):
    """마스크 좌/우 에지를 실제 이미지 최대 그래디언트로 스냅 → 픽셀 단위 정밀.
    날은 세로 밴드 → 각 행에서 좌/우 x 를 |수평 그래디언트| 최대 위치(±win)로 이동,
    행 방향 미디언 평활로 계단 제거. 강한 에지 없으면 원위치 유지(정반사 방어)."""
    cy = float(axis[1])
    f = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    gx = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3))
    H, W = m.shape
    ys = np.unique(np.nonzero(m)[0])
    if len(ys) < 20:
        return m
    Ld, Rd = {}, {}
    for y in ys:
        xs = np.nonzero(m[y])[0]
        if len(xs) < 3:
            continue
        lo, hi = int(xs.min()), int(xs.max())
        a, b = max(0, lo - win), min(W, lo + win + 1)
        seg = gx[y, a:b]
        Ld[y] = a + int(np.argmax(seg)) if seg.size and seg.max() > thr else lo
        a, b = max(0, hi - win), min(W, hi + win + 1)
        seg = gx[y, a:b]
        Rd[y] = a + int(np.argmax(seg)) if seg.size and seg.max() > thr else hi
    yy = np.array(sorted(Ld))
    Lv = np.array([Ld[y] for y in yy], float)
    Rv = np.array([Rd[y] for y in yy], float)
    k = 4
    Ls = np.array([np.median(Lv[max(0, i - k):i + k + 1]) for i in range(len(yy))])
    Rs = np.array([np.median(Rv[max(0, i - k):i + k + 1]) for i in range(len(yy))])
    m2 = np.zeros_like(m)
    for i, y in enumerate(yy):
        l, rr = int(round(Ls[i])), int(round(Rs[i]))
        if rr > l:
            m2[y, l:rr + 1] = 1
    return m2


def detect(gray, axis, r, glint_suppress=False, mask_guard=False, edge_snap=False,
           model='vit_h'):
    """SAM 으로 날1(위)/날2(아래) 마스크 검출.
    model: 'vit_h'|'vit_l'|'vit_b'(정식 SAM, GPU 권장, 기본) | 'mobile'(경량 CPU 폴백).
    glint_suppress: 포화픽셀(>=225) 제거 (152554 방식). 밝은 정반사가 플루트와 이어지면
        외곽선을 파먹을 수 있음 — 옵션.
    mask_guard: 폭 이상(치즐/플루트 오검출 혹) 자동 절단.
    반환 dict(mask1,mask2,tip1,tip2,column,_gray)."""
    predictor = _get_predictor(model)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    predictor.set_image(rgb)
    H, W = gray.shape
    out = {}
    for name, tkey, tip_deg in (('mask1', 'tip1', 270.0), ('mask2', 'tip2', 90.0)):
        pos, neg = _prompts(axis, r, tip_deg)
        pts = np.vstack([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=labels, multimask_output=True)
        cx, cy = axis
        probe = (int(cy + 0.5 * r * np.sin(np.radians(tip_deg))),
                 int(cx + 0.5 * r * np.cos(np.radians(tip_deg))))
        best = None
        for m, sc in zip(masks, scores):
            if m[probe[0], probe[1]] and (best is None or sc > best[1]):
                best = (m, sc)
        if best is None:
            best = (masks[int(np.argmax(scores))], float(np.max(scores)))
        m = best[0].astype(np.uint8)
        if glint_suppress:
            m = (m & (gray < 225)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        if tip_deg == 270.0:                             # 그 날 쪽 절반만 유지
            m[int(cy):, :] = 0
        else:
            m[:int(cy), :] = 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            m = (lab == big).astype(np.uint8)
        # 내부 구멍만 메움 (바깥 외곽선 불변, 정반사 구멍 되메움)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            m = np.zeros_like(m)
            cv2.drawContours(m, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
        if mask_guard:
            m = _guard_mask(m, axis, r)
        if edge_snap:                                    # 경계를 실제 에지로 스냅(픽셀 정밀)
            m = _snap_edges(m, gray, axis, r)
        out[name] = m
        ys = np.nonzero(m)[0]
        out[tkey] = int(ys.min()) if tip_deg == 270.0 else int(ys.max()) if len(ys) else int(cy)
    out['column'] = (out['mask1'] | out['mask2']).astype(np.uint8)
    out['_gray'] = gray
    return out


def draw_prompts(gray, axis, r):
    """프롬프트 위치 시각화(디버그용)."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for tip in (270.0, 90.0):
        pos, neg = _prompts(axis, r, tip)
        for p in pos:
            cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 255, 0), -1)
        for p in neg:
            cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
    return vis
