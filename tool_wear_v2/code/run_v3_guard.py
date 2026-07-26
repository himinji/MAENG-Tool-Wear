# -*- coding: utf-8 -*-
"""변형 3 — 152554 복원 + 마스크 가드.
워터셰드 수직화 + 글린트 억제 + tx-only + 단일 프레임 + 치즐/플루트 오검출 가드.
좋은 상태(v1)에 가장 지적된 사고(오검출 혹)만 방지. 중간 선택지.

  venv\Scripts\python tool_wear_v2\code\run_v3_guard.py --new "...\Initial" --worn "...\Test30"
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wear_pipeline

CFG = {'label': 'v3 복원+가드+스냅', 'vertical': 'watershed', 'align': 'tx',
       'glint_suppress': True, 'frames': 'single', 'mask_guard': True,
       'edge_snap': True, 'sam': 'vit_h'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--worn", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_wear_out"))
    ap.add_argument("--diam", type=float, default=8.0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    base = a.name or os.path.basename(os.path.normpath(a.worn))
    wear_pipeline.diagnose(a.new, a.worn, a.out, a.diam, f"{base}_v3", CFG)


if __name__ == "__main__":
    main()
