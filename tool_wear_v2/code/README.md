# SAM 기반 밑면(엔드페이스) 마모 진단

2날 엔드밀의 **밑면 날 마모**를, 새 공구와 마모 공구의 **날 영역(SAM)** 외곽선
차이로 측정한다. 외곽선 추종(contour-following)이 못 잡는 **날 끝(코너) 마모**를
잡는 것이 목적.

## 실행

```bat
venv\Scripts\python tool_wear_v2\code\run_v1_restore.py ^
  --new  "G:\...\260709_SM45C\Initial" ^
  --worn "G:\...\260709_SM45C\Test30"
```

결과는 `tool_wear_v2\code\_wear_out\<타임스탬프>_<이름>\` 에 저장(git 무시됨).
`--out` 으로 위치 변경, `--diam` 으로 공구 지름(mm, 기본 8.0) 지정.

## 파이프라인 (새 vs 마모 한 쌍)

1. **축·스케일** `botface_pipeline_v2.find_axis_scale` — OD 원 피팅으로 중심·반지름·µm/px
2. **프레임 선택** `pick_frame` — 대비(std) 최대 프레임
3. **크롭·링필** 축 중심 정사각 크롭(S=2.4r) + 배경 링 채움
4. **거친 정렬** `align_rotation` 회전 위상 맞춤 → `verticalize` 대략 세움
5. **정밀 수직화** `blade_align.verticalize` — SAM 마스크 중심선 기준 ±8° 보정
6. **몸통 정합** `register_body` — 비마모 몸통 평행이동(ECC)
7. **SAM 날 검출** `blade_sam.detect` — 기하 프롬프트(날 중심선 +점 / 플루트·릴리프·치즐 −점)
8. **날별 후퇴 측정** `blade_wear.measure` — 날1(긴 날)·날2(짧은 날) 외곽선 후퇴 = 마모(µm)
   - `vb_um` 플랭크 후퇴(90퍼센타일), `tip_um` 코너 마모(rad≥0.90r 최대)

## 파일

| 파일 | 역할 |
|---|---|
| `botface_pipeline_v2.py` | 축/스케일·프레임·크롭·회전정렬·수직화·몸통정합 (원본 `botface_pipeline.py` 불변, 이건 복제+역이식본) |
| `blade_sam.py` | MobileSAM 날 영역 검출(기하 프롬프트, edge-snap) |
| `blade_align.py` | SAM 중심선 정밀 수직화 |
| `blade_region.py` | 워터셰드 날 인식·오버레이·주석 기반 인식 |
| `blade_wear.py` | 날별 외곽선 후퇴 → VB·코너 마모 측정 |
| `blade_annotations.json` | 기준 프레임 수동 주석 |
| `wear_pipeline.py` | 설정(cfg) 파라미터화된 진단 코어 |
| `run_v1_restore.py` | 기본 구성(워터셰드+글린트억제+tx정렬+단일프레임+스냅) |
| `run_v3_guard.py` | v1 + 오검출(치즐/플루트) 가드 |
| `run_diagnosis.py` | SAM 정렬·수직 통합판(2D IoU 정합) |

## 준비물

기본 검출 모델은 **정식 SAM ViT-H (GPU)**. 셋업:

```bat
:: CUDA torch (RTX 등 NVIDIA GPU, 드라이버 CUDA 12.x)
venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
:: 정식 SAM 패키지
venv\Scripts\python -m pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
:: 체크포인트(2.4GB) → tool_wear_v2\code\sam_models\
::   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

- 모델 선택: 러너 `--sam vit_h|vit_l|vit_b|mobile` (기본 `vit_h`).
- GPU 없거나 SAM 미설치면 `--sam mobile` — 경량 `mobile_sam` + `mobile_sam.pt`(40MB, CPU)로 폴백.
- 체크포인트 파일: `sam_vit_h_4b8939.pth` / `sam_vit_l_0b3195.pth` / `sam_vit_b_01ec64.pth` / `mobile_sam.pt` — 모두 `sam_models\`, git 무시.

## 무시(커밋 금지)

`tool_wear_v2/code/_*/` (결과), `tool_wear_v2/code/sam_models/` (모델). `.gitignore` 참고.
