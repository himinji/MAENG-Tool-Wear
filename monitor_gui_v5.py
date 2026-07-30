# -*- coding: utf-8 -*-
"""
공구 마모 진단 GUI ver3 (밑면 기하 방식)

ver2(v4)와 차이:
- 파이프라인: 세그멘테이션 CNN → **밑면 기하 방식**(botface_pipeline). 모델 파일 불필요.
- 입력: 공구 세트 상위 폴더 하나 선택. 그 안에 Initial / Test1 / Test2 ... 가 생기고,
        각 폴더의 `Toolwear/아래`(밑면)를 추적한다. Initial = 새 공구 기준.
- 추가 입력: **공구 직경(mm)** → OD 기준으로 px→µm 환산.
- 결과: (1) 회전 정렬한 두 실제 사진 나란히, (2) 두 날(짧은날/긴날) 크롭 비교 + 후퇴 프로파일 그래프.
"""
import os
import sys
import json
import time
import queue
import threading
import traceback
import subprocess

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

import matplotlib
matplotlib.use('Agg')

import botface_pipeline as bp

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'monitor_gui_v5_config.json')

REF_NAME = 'Initial'                               # 새 공구 폴더 이름
SIDE_DIR = os.path.join(                           # 옆날 판별 스크립트 폴더
    os.path.dirname(os.path.abspath(__file__)), 'side_ref_angle')


def _has_face_images(folder):
    """폴더 안에 밑면 이미지(img_*.png)가 직접 들어있는지."""
    try:
        for e in os.scandir(folder):
            if e.is_file() and e.name.lower().startswith(bp.FILE_PREFIX) \
                    and e.name.lower().endswith(bp.FILE_EXT):
                return True
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return False
    return False


def face_folder(test_folder):
    """Test 폴더 → 그 안의 밑면 이미지 폴더.
    'Toolwear/아래' → '아래' 순으로 찾고, 둘 다 없으면 폴더 자체에
    이미지(img_*.png)가 있으면 그 폴더를 그대로 진단한다
    (밑면/아래 구조가 아닌 폴더나, 하위폴더를 직접 지정한 경우 대응)."""
    a = os.path.join(test_folder, 'Toolwear', '아래')
    if os.path.isdir(a):
        return a
    b = os.path.join(test_folder, '아래')
    if os.path.isdir(b):
        return b
    if _has_face_images(test_folder):
        return test_folder
    return b   # 못 찾음: 없는 '아래' 경로 반환 → 상위에서 isdir False 로 에러 안내


def side_folder(test_folder):
    """Test 폴더 → 그 안의 옆면 이미지 폴더.
    'Toolwear/옆' → '옆' 순으로 찾고, 둘 다 없으면 폴더 자체에
    이미지(img_*.png)가 있으면 그 폴더를 그대로 사용한다."""
    a = os.path.join(test_folder, 'Toolwear', '옆')
    if os.path.isdir(a):
        return a
    b = os.path.join(test_folder, '옆')
    if os.path.isdir(b):
        return b
    if _has_face_images(test_folder):
        return test_folder
    return b


def tool_name(folder):
    """폴더의 공구 이름 ('옆'/'아래'/'Toolwear' 를 지정했으면 상위 이름)."""
    b = os.path.basename(os.path.normpath(folder))
    if b in ('옆', '아래', 'Toolwear'):
        return tool_name(os.path.dirname(os.path.normpath(folder)))
    return b or 'tool'


def folder_signature(folder):
    count = 0
    total = 0
    try:
        for e in os.scandir(folder):
            if e.is_file() and e.name.lower().startswith(bp.FILE_PREFIX) \
                    and e.name.lower().endswith(bp.FILE_EXT):
                count += 1
                try:
                    total += e.stat().st_size
                except OSError:
                    pass
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return (0, 0)
    return (count, total)


def list_tool_subfolders(parent):
    """상위 폴더 바로 아래의, 밑면 이미지 폴더를 가진 하위 폴더들(이름순)."""
    out = []
    try:
        for e in os.scandir(parent):
            if e.is_dir() and os.path.isdir(face_folder(e.path)):
                out.append(e.path)
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return []
    return sorted(out, key=lambda p: bp._natkey(p))


class StreamToQueue:
    def __init__(self, q):
        self.q = q
        self._buf = ''

    def write(self, text):
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self.q.put(line)

    def flush(self):
        if self._buf:
            self.q.put(self._buf)
            self._buf = ''


class MonitorApp:
    POLL_INTERVAL = 2.0

    BG = '#EEF1F6'; CARD = '#FFFFFF'; NAVY = '#243A5E'; NAVY_SUB = '#A9B7CC'
    TEXT = '#1F2A3D'; MUTED = '#6B7688'; BORDER = '#D7DEE8'
    GREEN = '#2E7D32'; GREEN_H = '#266A2A'; RED = '#C62828'; RED_H = '#A91F1F'
    NEUTRAL = '#E6EAF1'; NEUTRAL_H = '#D6DCE6'; LOG_BG = '#1E1E2E'; LOG_FG = '#D6D6E0'

    def __init__(self, root):
        self.root = root
        self.root.title("공구 마모 진단 v3 (밑면 기하)")
        self.root.geometry("920x980")
        self.root.minsize(900, 760)
        self.root.configure(bg=self.BG)

        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self._ref_cache = None          # Initial 축·스케일 캐시
        self._ref_diam = None
        self._preview_img = None        # PhotoImage 참조 유지

        self._setup_style()
        self._build_ui()
        self._load_config()

        self.root.after(100, self._drain_log)
        self.root.after(300, self._drain_preview)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- 스타일 ----------
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        base_font = ('Malgun Gothic', 10)
        style.configure('.', background=self.BG, foreground=self.TEXT, font=base_font)
        style.configure('TFrame', background=self.BG)
        style.configure('TLabel', background=self.BG, foreground=self.TEXT)
        style.configure('Muted.TLabel', background=self.CARD, foreground=self.MUTED, font=('Malgun Gothic', 9))
        style.configure('Field.TLabel', background=self.CARD, foreground=self.TEXT, font=('Malgun Gothic', 10))
        style.configure('Modern.TEntry', fieldbackground=self.CARD, bordercolor=self.BORDER,
                        lightcolor=self.BORDER, darkcolor=self.BORDER, padding=6)
        style.map('Modern.TEntry', bordercolor=[('focus', self.NAVY)])
        style.configure('Modern.TSpinbox', fieldbackground=self.CARD, bordercolor=self.BORDER, arrowsize=12, padding=4)
        style.configure('Card.TCheckbutton', background=self.CARD, foreground=self.TEXT)
        style.map('Card.TCheckbutton', background=[('active', self.CARD)])

        def btn(name, bg, bg_h, fg='white', font=('Malgun Gothic', 10, 'bold'), pad=(16, 9)):
            style.configure(name, background=bg, foreground=fg, borderwidth=0,
                            focusthickness=0, focuscolor=bg, padding=pad, font=font)
            style.map(name, background=[('disabled', '#C7CDD8'), ('active', bg_h), ('!active', bg)],
                      foreground=[('disabled', '#EFEFEF')])
        btn('Accent.TButton', self.GREEN, self.GREEN_H)
        btn('Danger.TButton', self.RED, self.RED_H)
        btn('Ghost.TButton', self.NEUTRAL, self.NEUTRAL_H, fg=self.TEXT, font=('Malgun Gothic', 10), pad=(14, 8))
        btn('Pick.TButton', self.NEUTRAL, self.NEUTRAL_H, fg=self.TEXT, font=('Malgun Gothic', 9), pad=(12, 5))

        style.configure('TNotebook', background=self.BG, borderwidth=0, tabmargins=(0, 6, 0, 0),
                        bordercolor=self.BG, lightcolor=self.BG, darkcolor=self.BG)
        try:
            style.layout('TNotebook.Tab', [
                ('Notebook.tab', {'sticky': 'nswe', 'children': [
                    ('Notebook.padding', {'side': 'top', 'sticky': 'nswe', 'children': [
                        ('Notebook.label', {'side': 'top', 'sticky': ''})
                    ]})
                ]})
            ])
        except tk.TclError:
            pass
        # 비선택 탭: 작고 흐리게 / 선택 탭: 크고 진하게(위로만 확장해 좌우 정렬 유지)
        style.configure('TNotebook.Tab', background=self.NEUTRAL, foreground=self.MUTED,
                        padding=(20, 7), font=('Malgun Gothic', 10), borderwidth=1, bordercolor=self.BORDER,
                        lightcolor=self.BORDER, darkcolor=self.BORDER)
        style.map('TNotebook.Tab',
                  background=[('selected', self.CARD), ('!selected', self.NEUTRAL)],
                  foreground=[('selected', self.NAVY), ('!selected', self.MUTED)],
                  font=[('selected', ('Malgun Gothic', 11, 'bold')), ('!selected', ('Malgun Gothic', 10))],
                  padding=[('selected', (30, 13)), ('!selected', (20, 7))],
                  expand=[('selected', [0, 3, 0, 0])])
        style.configure('Hint.TLabel', background=self.BG, foreground=self.MUTED, font=('Malgun Gothic', 9))

    def _card(self, parent, title, pad=(16, 8, 12)):
        border = tk.Frame(parent, bg=self.BORDER)
        inner = tk.Frame(border, bg=self.CARD)
        inner.pack(fill='both', expand=True, padx=1, pady=1)
        tk.Label(inner, text=title, bg=self.CARD, fg=self.NAVY,
                 font=('Malgun Gothic', 10, 'bold')).pack(anchor='w', padx=pad[0], pady=(10, 0))
        content = tk.Frame(inner, bg=self.CARD)
        content.pack(fill='both', expand=True, padx=pad[0], pady=(pad[1], pad[2]))
        return border, content

    # ---------- UI ----------
    def _build_ui(self):
        header = tk.Frame(self.root, bg=self.NAVY, height=72)
        header.pack(fill='x', side='top'); header.pack_propagate(False)
        hl = tk.Frame(header, bg=self.NAVY); hl.pack(side='left', padx=22, pady=12)
        tk.Label(hl, text="공구 마모 진단", bg=self.NAVY, fg='white',
                 font=('Malgun Gothic', 17, 'bold')).pack(anchor='w')
        tk.Label(hl, text="Tool Wear Diagnosis  ·  v3 (밑면 기하 · 옆날 판별 · 정렬/후퇴 측정)",
                 bg=self.NAVY, fg=self.NAVY_SUB, font=('Malgun Gothic', 9)).pack(anchor='w')
        self.status_lbl = tk.Label(header, text="●  대기 중", bg=self.NAVY, fg=self.NAVY_SUB,
                                   font=('Malgun Gothic', 10, 'bold'))
        self.status_lbl.pack(side='right', padx=22)

        body = ttk.Frame(self.root)
        body.pack(fill='both', expand=True, padx=16, pady=14)

        self.parent_var = tk.StringVar()
        self.diag_ref_var = tk.StringVar()       # 폴더 진단: 새 공구(기준) 폴더
        self.diag_target_var = tk.StringVar()    # 폴더 진단: 진단할(마모) 폴더
        self.side_ref_var = tk.StringVar()       # 옆날 판별: 새 공구(기준) 폴더
        self.side_target_var = tk.StringVar()    # 옆날 판별: 판별할(마모) 폴더
        self.side_rescan_var = tk.BooleanVar(value=False)
        self.output_var = tk.StringVar(value=os.path.join(os.getcwd(), 'results_botface'))
        self.diam_var = tk.StringVar(value='8.0')
        self.stable_var = tk.StringVar(value='5')
        self.process_existing_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="대기 중")

        nb = ttk.Notebook(body); nb.pack(fill='x')

        # 탭 1: 모니터링
        tab_mon = ttk.Frame(nb, padding=(0, 12)); nb.add(tab_mon, text='  실시간 모니터링  ')
        m1b, m1 = self._card(tab_mon, '설정'); m1b.pack(fill='x'); m1.columnconfigure(1, weight=1)
        self._field_row(m1, 0, "공구 세트 상위 폴더", self.parent_var, self._pick_parent)
        self._field_row(m1, 1, "결과 폴더", self.output_var, self._pick_output)
        self._diam_row(m1, 2)
        m2b, m2 = self._card(tab_mon, '옵션'); m2b.pack(fill='x', pady=(12, 0))
        ttk.Label(m2, text="안정화 대기(초)", style='Field.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Spinbox(m2, from_=1, to=600, textvariable=self.stable_var, width=6,
                    style='Modern.TSpinbox').grid(row=0, column=1, sticky='w', padx=(10, 24))
        tk.Checkbutton(m2, text="시작 시 기존 Test 폴더도 처리", variable=self.process_existing_var,
                       bg=self.CARD, fg=self.TEXT, activebackground=self.CARD, activeforeground=self.TEXT,
                       selectcolor='white', font=('Malgun Gothic', 10),
                       highlightthickness=0, bd=0).grid(row=0, column=2, sticky='w')
        ttk.Label(m2, text="상위 폴더 안 'Initial' = 새 공구 기준. 각 폴더의 Toolwear/아래(밑면)를 추적합니다.",
                  style='Muted.TLabel').grid(row=1, column=0, columnspan=3, sticky='w', pady=(8, 0))
        mb = ttk.Frame(tab_mon); mb.pack(fill='x', pady=(12, 0))
        self.start_btn = ttk.Button(mb, text="▶  모니터링 시작", style='Accent.TButton', command=self._start)
        self.start_btn.pack(side='left')
        self.stop_btn = ttk.Button(mb, text="■  중지", style='Danger.TButton', command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=(10, 0))

        # 탭 2: 폴더 진단 (직접 지정: 새 공구 폴더 + 진단할 폴더)
        tab_diag = ttk.Frame(nb, padding=(0, 12)); nb.add(tab_diag, text='  폴더 진단  ')
        d1b, d1 = self._card(tab_diag, '설정'); d1b.pack(fill='x'); d1.columnconfigure(1, weight=1)
        self._field_row(d1, 0, "새 공구(기준) 폴더", self.diag_ref_var, self._pick_diag_ref)
        self._field_row(d1, 1, "진단할 폴더", self.diag_target_var, self._pick_diag_target)
        self._field_row(d1, 2, "결과 폴더", self.output_var, self._pick_output)
        self._diam_row(d1, 3)
        ttk.Label(tab_diag,
                  text="• 밑면 하위폴더(Toolwear/아래 또는 아래)를 가진 폴더를 지정하면 자동으로 그 안을 진단합니다 (예: Initial, Test14).\n"
                       "• 하위폴더가 없으면 지정한 폴더 안의 img_*.png 를 바로 진단합니다 (밑면 폴더를 직접 지정해도 됨).",
                  style='Hint.TLabel', justify='left').pack(anchor='w', pady=(10, 0))
        db = ttk.Frame(tab_diag); db.pack(fill='x', pady=(12, 0))
        self.diag_btn = ttk.Button(db, text="▶  진단 시작", style='Accent.TButton', command=self._diagnose)
        self.diag_btn.pack(side='left')

        # 탭 3: 옆날 판별 (옆면 회전 촬영 → SAM 크롭 · 날 후퇴 측정)
        tab_side = ttk.Frame(nb, padding=(0, 12)); nb.add(tab_side, text='  옆날 판별  ')
        s1b, s1 = self._card(tab_side, '설정'); s1b.pack(fill='x'); s1.columnconfigure(1, weight=1)
        self._field_row(s1, 0, "새 공구(기준) 폴더", self.side_ref_var, self._pick_side_ref)
        self._field_row(s1, 1, "판별할(마모) 폴더", self.side_target_var, self._pick_side_target)
        self._field_row(s1, 2, "결과 폴더", self.output_var, self._pick_output)
        self._diam_row(s1, 3)
        tk.Checkbutton(s1, text="레퍼런스 각도 스캔 다시 하기 (평소엔 기존 스캔 재사용)",
                       variable=self.side_rescan_var,
                       bg=self.CARD, fg=self.TEXT, activebackground=self.CARD,
                       activeforeground=self.TEXT, selectcolor='white',
                       font=('Malgun Gothic', 10), highlightthickness=0, bd=0
                       ).grid(row=4, column=1, sticky='w', pady=(2, 0))
        ttk.Label(tab_side,
                  text="• 각 폴더의 Toolwear/옆(옆면 1° 회전 촬영, img_*.png)을 사용합니다. 옆 폴더를 직접 지정해도 됩니다.\n"
                       "• 처음 1회는 레퍼런스 각도 스캔(420장, 폴더당 수 분)이 자동 실행되고 이후 재사용됩니다.\n"
                       "• SAM 날 검출 포함 전체 판별에 수 분이 걸립니다. 결과: 날 2개의 후퇴(µm) 그래프·겹침 사진.",
                  style='Hint.TLabel', justify='left').pack(anchor='w', pady=(10, 0))
        sb2 = ttk.Frame(tab_side); sb2.pack(fill='x', pady=(12, 0))
        self.side_btn = ttk.Button(sb2, text="▶  옆날 판별 시작", style='Accent.TButton',
                                   command=self._side_diagnose)
        self.side_btn.pack(side='left')

        util = ttk.Frame(body); util.pack(fill='x', pady=(12, 0))
        ttk.Button(util, text="결과 폴더 열기", style='Ghost.TButton', command=self._open_output).pack(side='left')
        ttk.Button(util, text="로그 지우기", style='Ghost.TButton', command=self._clear_log).pack(side='left', padx=(10, 0))

        # 하단: 로그 + 결과 미리보기 (좌우)
        bottom = ttk.Frame(body); bottom.pack(fill='both', expand=True, pady=(12, 0))
        bottom.rowconfigure(0, weight=1)
        bottom.columnconfigure(0, weight=1, uniform='pane')   # 로그
        bottom.columnconfigure(1, weight=1, uniform='pane')   # 결과 미리보기 (1:1)
        lc_b, lc = self._card(bottom, '로그', pad=(10, 6, 10)); lc_b.grid(row=0, column=0, sticky='nsew')
        # 로그는 고정폭 글꼴(D2Coding: 한글도 정렬 OK). 없으면 Consolas→맑은 고딕 폴백.
        import tkinter.font as tkfont
        _fams = set(tkfont.families())
        log_family = next((f for f in ('D2Coding', 'Consolas', 'Malgun Gothic') if f in _fams), 'Malgun Gothic')
        self.log_text = scrolledtext.ScrolledText(lc, height=22, width=40, state='disabled', wrap='word', relief='flat',
                                                   borderwidth=0, bg=self.LOG_BG, fg=self.LOG_FG,
                                                   insertbackground=self.LOG_FG, font=(log_family, 10), padx=10, pady=8)
        self.log_text.pack(fill='both', expand=True)
        self.log_text.tag_config('hl', foreground='#5CC8FF')
        self.log_text.tag_config('ok', foreground='#7FD17F')
        self.log_text.tag_config('err', foreground='#FF8A75')

        pv_b, pv = self._card(bottom, '결과 미리보기', pad=(10, 6, 10)); pv_b.grid(row=0, column=1, sticky='nsew', padx=(12, 0))
        self.preview_lbl = tk.Label(pv, bg=self.CARD, fg=self.MUTED,
                                    text="진단이 끝나면 결과 그래프가 여기 표시됩니다." if _HAS_PIL
                                    else "(미리보기에 Pillow 필요: pip install pillow)")
        self.preview_lbl.pack(fill='both', expand=True)
        self.preview_queue = queue.Queue()

    def _field_row(self, parent, row, label, var, cmd):
        ttk.Label(parent, text=label, style='Field.TLabel').grid(row=row, column=0, sticky='w', pady=5, padx=(0, 12))
        ttk.Entry(parent, textvariable=var, style='Modern.TEntry').grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Button(parent, text="찾기", style='Pick.TButton', command=cmd).grid(row=row, column=2, sticky='w', padx=(10, 0), pady=5)

    def _diam_row(self, parent, row):
        ttk.Label(parent, text="공구 직경 (mm)", style='Field.TLabel').grid(row=row, column=0, sticky='w', pady=5, padx=(0, 12))
        ttk.Spinbox(parent, from_=0.5, to=50.0, increment=0.5, textvariable=self.diam_var, width=8,
                    style='Modern.TSpinbox').grid(row=row, column=1, sticky='w', pady=5)

    def _set_status(self, text, color):
        self.status_lbl.configure(text="●  " + text, fg=color)
        self.status_var.set(text)

    # ---------- 파일 선택 ----------
    def _pick_parent(self):
        d = filedialog.askdirectory(title="공구 세트 상위 폴더 선택 (Initial/Test... 포함)")
        if d:
            self.parent_var.set(d)

    def _pick_diag_ref(self):
        d = filedialog.askdirectory(title="새 공구(기준) 폴더 선택 (밑면/옆면 하위폴더 포함)",
                                    initialdir=self.diag_ref_var.get() or os.getcwd())
        if d:
            self.diag_ref_var.set(d)

    def _pick_diag_target(self):
        d = filedialog.askdirectory(title="진단할 폴더 선택 (밑면/옆면 하위폴더 포함)",
                                    initialdir=self.diag_target_var.get() or os.getcwd())
        if d:
            self.diag_target_var.set(d)

    def _pick_side_ref(self):
        d = filedialog.askdirectory(title="새 공구(기준) 폴더 선택 (Toolwear/옆 포함)",
                                    initialdir=self.side_ref_var.get() or os.getcwd())
        if d:
            self.side_ref_var.set(d)

    def _pick_side_target(self):
        d = filedialog.askdirectory(title="판별할 폴더 선택 (Toolwear/옆 포함)",
                                    initialdir=self.side_target_var.get() or os.getcwd())
        if d:
            self.side_target_var.set(d)

    def _pick_output(self):
        d = filedialog.askdirectory(title="결과 폴더 선택")
        if d:
            self.output_var.set(d)

    def _open_output(self):
        out = self.output_var.get().strip()
        if out and os.path.isdir(out):
            try:
                os.startfile(out)
            except Exception as e:
                self.log(f"폴더 열기 실패: {e}")
        else:
            messagebox.showinfo("안내", "결과 폴더가 아직 없습니다.")

    # ---------- 로그/미리보기 ----------
    def log(self, msg):
        self.log_queue.put(str(msg))

    def _drain_log(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                tag = None
                if '오류' in line or 'Error' in line or 'Traceback' in line:
                    tag = 'err'
                elif '★' in line or '완료' in line or '[OK]' in line:
                    tag = 'ok'
                elif line.lstrip().startswith('>>>') or line.startswith('==='):
                    tag = 'hl'
                self.log_text.configure(state='normal')
                self.log_text.insert('end', line + '\n', (tag,) if tag else ())
                self.log_text.see('end')
                self.log_text.configure(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    def _drain_preview(self):
        try:
            while True:
                path = self.preview_queue.get_nowait()
                self._show_preview(path)
        except queue.Empty:
            pass
        self.root.after(300, self._drain_preview)

    def _show_preview(self, path):
        if not _HAS_PIL or not path or not os.path.isfile(path):
            return
        try:
            self.preview_lbl.update_idletasks()
            maxw = max(320, self.preview_lbl.winfo_width() - 12)
            maxh = max(320, self.preview_lbl.winfo_height() - 12)
            im = Image.open(path)
            im.thumbnail((maxw, maxh))
            self._preview_img = ImageTk.PhotoImage(im)
            self.preview_lbl.configure(image=self._preview_img, text='')
        except Exception as e:
            self.log(f"미리보기 실패: {e}")

    def _clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

    # ---------- 공통 검증 ----------
    def _read_params(self):
        parent = self.parent_var.get().strip()
        output = self.output_var.get().strip()
        if not parent or not os.path.isdir(parent):
            messagebox.showerror("오류", "공구 세트 상위 폴더가 올바르지 않습니다.")
            return None
        if not output:
            messagebox.showerror("오류", "결과 폴더를 지정하세요.")
            return None
        try:
            diam = float(self.diam_var.get())
            if diam <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("오류", "공구 직경(mm)은 양수여야 합니다.")
            return None
        ref = os.path.join(parent, REF_NAME)
        if not os.path.isdir(face_folder(ref)):
            messagebox.showerror("오류", f"기준 폴더가 없습니다:\n{face_folder(ref)}\n"
                                         f"(상위 폴더 안에 '{REF_NAME}/Toolwear/아래'가 필요)")
            return None
        return {'parent': parent, 'output': output, 'diam': diam, 'ref': ref}

    # ---------- 시작/중지 ----------
    def _start(self):
        if self._busy():
            return
        p = self._read_params()
        if not p:
            return
        try:
            p['stable_sec'] = float(self.stable_var.get())
        except ValueError:
            p['stable_sec'] = 5.0
        p['process_existing'] = self.process_existing_var.get()
        self._launch(self._monitor_loop, p, allow_stop=True, status="실행 중")

    def _diagnose(self):
        if self._busy():
            return
        ref = self.diag_ref_var.get().strip()
        target = self.diag_target_var.get().strip()
        output = self.output_var.get().strip()
        if not ref or not os.path.isdir(face_folder(ref)):
            messagebox.showerror("오류", f"새 공구(기준) 폴더에 이미지가 없습니다:\n{ref}\n"
                                         "(Toolwear/아래 · 아래 하위폴더 또는 폴더 자체의 img_*.png)")
            return
        if not target or not os.path.isdir(face_folder(target)):
            messagebox.showerror("오류", f"진단할 폴더에 이미지가 없습니다:\n{target}\n"
                                         "(Toolwear/아래 · 아래 하위폴더 또는 폴더 자체의 img_*.png)")
            return
        if not output:
            messagebox.showerror("오류", "결과 폴더를 지정하세요.")
            return
        try:
            diam = float(self.diam_var.get())
            if diam <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("오류", "공구 직경(mm)은 양수여야 합니다.")
            return
        p = {'ref': ref, 'target': target, 'output': output, 'diam': diam}
        self._launch(self._oneshot, p, allow_stop=True, status="진단 중")

    def _side_diagnose(self):
        if self._busy():
            return
        ref = self.side_ref_var.get().strip()
        target = self.side_target_var.get().strip()
        output = self.output_var.get().strip()
        if not os.path.isfile(os.path.join(SIDE_DIR, 'wear_side_compare_v2.py')):
            messagebox.showerror("오류", f"옆날 판별 스크립트 폴더가 없습니다:\n{SIDE_DIR}")
            return
        if not ref or not os.path.isdir(side_folder(ref)):
            messagebox.showerror("오류", f"새 공구(기준) 폴더에 옆면 이미지가 없습니다:\n{ref}\n"
                                         "(Toolwear/옆 · 옆 하위폴더 또는 폴더 자체의 img_*.png)")
            return
        if not target or not os.path.isdir(side_folder(target)):
            messagebox.showerror("오류", f"판별할 폴더에 옆면 이미지가 없습니다:\n{target}\n"
                                         "(Toolwear/옆 · 옆 하위폴더 또는 폴더 자체의 img_*.png)")
            return
        if not output:
            messagebox.showerror("오류", "결과 폴더를 지정하세요.")
            return
        try:
            diam = float(self.diam_var.get())
            if diam <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("오류", "공구 직경(mm)은 양수여야 합니다.")
            return
        p = {'ref': ref, 'target': target, 'output': output, 'diam': diam,
             'rescan': self.side_rescan_var.get()}
        self._launch(self._side_oneshot, p, allow_stop=True, status="옆날 판별 중")

    def _busy(self):
        return self.worker is not None and self.worker.is_alive()

    def _launch(self, fn, params, allow_stop, status):
        self._save_config()
        self.stop_event.clear()
        self._ref_cache = None
        self._set_busy(True, allow_stop)
        self._set_status(status, '#7FD17F')
        self.worker = threading.Thread(target=self._run_worker, args=(fn, params), daemon=True)
        self.worker.start()

    def _set_busy(self, busy, allow_stop=False):
        st = 'disabled' if busy else 'normal'
        self.start_btn.configure(state=st)
        self.diag_btn.configure(state=st)
        self.side_btn.configure(state=st)
        self.stop_btn.configure(state='normal' if (busy and allow_stop) else 'disabled')

    def _stop(self):
        self.stop_event.set()
        self._set_status("중지 중...", '#FFD27F')
        self.stop_btn.configure(state='disabled')

    def _worker_done(self):
        self._set_busy(False)
        self._set_status("대기 중", self.NAVY_SUB)

    def _run_worker(self, fn, p):
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = StreamToQueue(self.log_queue)
        sys.stderr = StreamToQueue(self.log_queue)
        try:
            fn(p)
        except Exception:
            self.log("워커 오류:\n" + traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            self.root.after(0, self._worker_done)

    # ---------- 처리 ----------
    def _get_ref(self, ref_folder, diam):
        """Initial 축·스케일 캐시 (직경 바뀌면 재계산)."""
        if self._ref_cache is not None and self._ref_diam == diam:
            return self._ref_cache
        self.log("=" * 50)
        self.log(f">>> 기준(Initial) 축·스케일 계산: {face_folder(ref_folder)}")
        self._ref_cache = bp.find_axis_scale(face_folder(ref_folder), diam, log=self.log)
        self._ref_diam = diam
        a = self._ref_cache
        self.log(f"★ 기준 완료: 축=({a['axis'][0]:.0f},{a['axis'][1]:.0f}) "
                 f"OD_r={a['od_r']:.1f}px  스케일={a['um_per_px']:.3f}µm/px")
        return self._ref_cache

    def _process(self, p, test_folder):
        name = os.path.basename(os.path.normpath(test_folder))
        try:
            self.log(f"\n>>> 처리: {name}")
            ref = self._get_ref(p['ref'], p['diam'])
            res = bp.process_tool(face_folder(p['ref']), face_folder(test_folder),
                                  p['diam'], p['output'], name, ref_axis_scale=ref, log=self.log)
            recs = "  ".join(f"{b['name']}={b['recession_um']:.1f}µm" for b in res['blades'])
            self.log(f"[{name}] 완료 ▶ {recs}  (정렬 edge-NCC {res['zncc']:.3f})")
            gp = res['paths'].get('graph')
            if gp:
                self.preview_queue.put(gp)
            return True
        except Exception:
            self.log(f"[{name}] 처리 중 오류:\n" + traceback.format_exc())
            return False

    def _oneshot(self, p):
        """폴더 진단: 새 공구(기준) 폴더 vs 진단할 폴더 1회 비교."""
        name = os.path.basename(os.path.normpath(p['target']))
        refname = os.path.basename(os.path.normpath(p['ref']))
        try:
            self.log("=" * 50)
            self.log(f">>> 진단: {name}  (기준 = {refname})")
            res = bp.process_tool(face_folder(p['ref']), face_folder(p['target']),
                                  p['diam'], p['output'], name, ref_axis_scale=None, log=self.log)
            recs = "  ".join(f"{b['name']}={b['recession_um']:.1f}µm" for b in res['blades'])
            self.log(f"[{name}] 완료 ▶ {recs}  (정렬 edge-NCC {res['zncc']:.3f})")
            gp = res['paths'].get('graph')
            if gp:
                self.preview_queue.put(gp)
            self.log("\n>>> 진단 완료.")
        except Exception:
            self.log(f"[{name}] 진단 중 오류:\n" + traceback.format_exc())

    # ---------- 옆날 판별 ----------
    def _run_script(self, args_list, cwd):
        """side_ref_angle 스크립트를 서브프로세스로 실행, 출력을 로그로 스트리밍."""
        env = {**os.environ, 'PYTHONUTF8': '1'}
        proc = subprocess.Popen([sys.executable, '-u'] + args_list, cwd=cwd,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', env=env)
        for line in proc.stdout:
            self.log(line.rstrip('\n'))
            if self.stop_event.is_set():
                proc.terminate()
                self.log(">>> 중지 요청 — 프로세스 종료")
                break
        proc.wait()
        return proc.returncode

    def _ensure_ref_scan(self, side_path, name, output, rescan):
        """레퍼런스 각도 스캔 CSV 확보 (없거나 재스캔 요청 시 find_ref_angle 실행)."""
        scan_dir = os.path.join(output, f'refscan_{name}')
        csvp = os.path.join(scan_dir, 'ref_angle_scores.csv')
        if os.path.isfile(csvp) and not rescan:
            self.log(f"레퍼런스 스캔 재사용: {csvp}")
            return csvp
        self.log(f"\n>>> 레퍼런스 각도 스캔 ({name}): {side_path}")
        rc = self._run_script([os.path.join(SIDE_DIR, 'find_ref_angle.py'),
                               side_path, '--out', scan_dir], cwd=SIDE_DIR)
        if rc != 0 or not os.path.isfile(csvp):
            raise RuntimeError(f"레퍼런스 각도 스캔 실패 (종료 코드 {rc})")
        return csvp

    def _side_oneshot(self, p):
        """옆날 판별: 레퍼런스 스캔(캐시) → wear_side_compare_v2 실행."""
        name = tool_name(p['target'])
        refname = tool_name(p['ref'])
        try:
            self.log("=" * 50)
            self.log(f">>> 옆날 판별: {name}  (기준 = {refname})")
            new_csv = self._ensure_ref_scan(side_folder(p['ref']),
                                            f'{refname}_side', p['output'], p['rescan'])
            if self.stop_event.is_set():
                return
            worn_csv = self._ensure_ref_scan(side_folder(p['target']),
                                             f'{name}_side', p['output'], p['rescan'])
            if self.stop_event.is_set():
                return
            out_dir = os.path.join(p['output'], f'side_{name}')
            self.log(f"\n>>> 옆날 마모 측정 시작 (결과: {out_dir})")
            rc = self._run_script([os.path.join(SIDE_DIR, 'wear_side_compare_v2.py'),
                                   '--new-dir', side_folder(p['ref']),
                                   '--worn-dir', side_folder(p['target']),
                                   '--new-csv', new_csv, '--worn-csv', worn_csv,
                                   '--diam', str(p['diam']), '--tip-skip', '210',
                                   '--out', out_dir], cwd=SIDE_DIR)
            if rc == 0:
                self.log(f"\n[{name}] 옆날 판별 완료 ▶ 결과 폴더: {out_dir}")
                gp = os.path.join(out_dir, 'pair1_retreat.png')
                if os.path.isfile(gp):
                    self.preview_queue.put(gp)
            elif not self.stop_event.is_set():
                self.log(f"[{name}] 옆날 판별 실패 (종료 코드 {rc})")
        except Exception:
            self.log(f"[{name}] 옆날 판별 중 오류:\n" + traceback.format_exc())

    def _monitor_loop(self, p):
        self._get_ref(p['ref'], p['diam'])
        processed = set()
        pending = {}
        if not p['process_existing']:
            for s in list_tool_subfolders(p['parent']):
                processed.add(os.path.abspath(s))
            self.log("기존 Test 폴더는 건너뜁니다 (새로 생기는 폴더만).")
        else:
            self.log("기존 Test 폴더도 처리합니다.")
        processed.add(os.path.abspath(p['ref']))   # 기준은 대상 제외

        self.log(f"\n>>> 감시 시작: {p['parent']} (주기 {self.POLL_INTERVAL}s, 안정화 {p['stable_sec']}s)\n")
        while not self.stop_event.is_set():
            for sub in list_tool_subfolders(p['parent']):
                if self.stop_event.is_set():
                    break
                ap = os.path.abspath(sub)
                if ap in processed:
                    continue
                sig = folder_signature(face_folder(sub))
                if sig[0] == 0:
                    continue
                prev = pending.get(ap)
                if prev is None or prev[0] != sig:
                    if prev is None:
                        self.log(f"새 폴더 감지: {os.path.basename(sub)} (이미지 {sig[0]}장, 안정화 대기)")
                    pending[ap] = (sig, time.time())
                    continue
                if time.time() - prev[1] >= p['stable_sec']:
                    self._process(p, sub)
                    processed.add(ap)
                    pending.pop(ap, None)
            self.stop_event.wait(self.POLL_INTERVAL)
        self.log("\n>>> 모니터링 중지됨.")

    # ---------- 설정 ----------
    def _save_config(self):
        cfg = {'parent': self.parent_var.get(),
               'diag_ref': self.diag_ref_var.get(), 'diag_target': self.diag_target_var.get(),
               'side_ref': self.side_ref_var.get(), 'side_target': self.side_target_var.get(),
               'output': self.output_var.get(),
               'diam': self.diam_var.get(), 'stable': self.stable_var.get(),
               'process_existing': self.process_existing_var.get()}
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _load_config(self):
        if not os.path.isfile(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.parent_var.set(cfg.get('parent', ''))
        self.diag_ref_var.set(cfg.get('diag_ref', ''))
        self.diag_target_var.set(cfg.get('diag_target', ''))
        self.side_ref_var.set(cfg.get('side_ref', ''))
        self.side_target_var.set(cfg.get('side_target', ''))
        self.output_var.set(cfg.get('output', self.output_var.get()))
        self.diam_var.set(cfg.get('diam', '8.0'))
        self.stable_var.set(cfg.get('stable', '5'))
        self.process_existing_var.set(cfg.get('process_existing', True))

    def _on_close(self):
        self.stop_event.set()
        self._save_config()
        self.root.after(200, self.root.destroy)


def main():
    try:
        root = tk.Tk()
        MonitorApp(root)
        root.mainloop()
    except Exception:
        err = traceback.format_exc()
        try:
            base = os.path.dirname(sys.executable if getattr(sys, 'frozen', False)
                                   else os.path.abspath(__file__))
            with open(os.path.join(base, 'monitor_gui_v5_crash.log'), 'w', encoding='utf-8') as f:
                f.write(err)
        except OSError:
            pass
        try:
            messagebox.showerror("실행 오류", err)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
