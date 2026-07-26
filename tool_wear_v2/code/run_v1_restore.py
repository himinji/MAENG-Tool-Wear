# -*- coding: utf-8 -*-
"""변형 1 — 152554 상태 복원.
워터셰드 수직화 + 글린트 억제 + tx-only 측정정렬 + 단일 프레임.
좋다고 하신 20260721_152554 결과를 만든 구성.

  venv\Scripts\python tool_wear_v2\code\run_v1_restore.py --new "...\Initial" --worn "...\Test30"
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wear_pipeline

CFG = {'label': 'v1 복원(152554)+스냅', 'vertical': 'watershed', 'align': 'tx',
       'glint_suppress': True, 'frames': 'single', 'mask_guard': False,
       'edge_snap': True, 'sam': 'vit_h'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--worn", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_wear_out"))
    ap.add_argument("--diam", type=float, default=8.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--sam", default="vit_h",
                    choices=["mobile", "vit_b", "vit_l", "vit_h"],
                    help="검출 모델(기본 vit_h): vit_h·vit_l·vit_b(정식 SAM, GPU) / mobile(경량 CPU 폴백)")
    a = ap.parse_args()
    cfg = dict(CFG, sam=a.sam, label=f"{CFG['label']} · {a.sam}")
    wear_pipeline.diagnose(a.new, a.worn, a.out, a.diam,
                           (a.name or None) and f"{a.name}_v1" or f"{os.path.basename(os.path.normpath(a.worn))}_v1",
                           cfg)


if __name__ == "__main__":
    main()
