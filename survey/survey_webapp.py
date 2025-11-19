
# survey_webapp.py
#
# Minimal web port of your DearPyGui + VLC survey to a browser.
# - Framework: NiceGUI (pip install nicegui==2.*)
# - URL format: http://<server>:8080/<pname>  (e.g., /p1, /alice, /u001)
# - Data files:
#     - Base CSV: result/user_rating.csv           (the source template without _<pname>)
#     - Per-user CSV: result/user_rating_<pname>.csv
#     - Videos: dataset/<scene_id>/RGB.mp4           (served as static files)
#
# Run:
#   pip install nicegui pandas
#   python survey_webapp.py
#   (then open http://localhost:8080/p1)
#
# Optional production tip:
#   uvicorn survey_webapp:app --host 0.0.0.0 --port 8080
#
# Notes:
# - Concurrency: each participant should use a unique <pname>, which writes a separate CSV.
# - Sorting and column semantics mirror your original script (removes column index 1 from sorting order).
# - This uses HTML5 <video> (no VLC); videos must be browser-compatible (e.g., H.264 .mp4).
#
from __future__ import annotations

import os
import threading
from typing import List, Dict, Any
from datetime import datetime
import time

import pandas as pd
from nicegui import ui, app
from pathlib import Path
import numpy as np
from itertools import combinations
from collections import defaultdict
from typing import List
import random

from questionnaire_list import questionnaire_lst

# ---- Config ----
RESULT_DIR        = 'result/'
DATASET_DIR       = 'data/output'
REQUIRED_COLUMNS  = ["A", "B", "question", "preferred"]

APP_DIR = Path(__file__).parent.resolve()
DATASET_ABS = (APP_DIR / DATASET_DIR).resolve()
app.add_static_files("/"+DATASET_DIR, str(DATASET_DIR))

MODE_TEXT = "text"
MODE_CI   = "correct_incorrect"
MODE_AB   = "ab"
# UI_ORDER  = [MODE_TEXT, MODE_CI, MODE_CI, MODE_AB]
UI_ORDER  = [MODE_AB]
SCENE_PREFIX = "Ground-truth scene description:"

BASE_DIR = Path(__file__).parent

# ---- Utilities to mirror original behavior ----
def save_to_csv(csv_path: str, df: pd.DataFrame) -> None:
    columns = df.columns.tolist()
    if len(columns) > 1:
        # match your "columns.pop(1)" before sorting
        columns.pop(1)
    df = df.sort_values(by=columns, ascending=True, ignore_index=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

def create_template(csv_path):
    groups = defaultdict(list)

    # 1. Walk and group file paths by their last folder name
    for subdir, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith('.png'):
                # full relative path from DATASET_DIR
                rel_dir = os.path.relpath(subdir, DATASET_DIR)  # e.g. "smiley" or "scene1/smiley"
                rel_path = os.path.join(rel_dir, file) if rel_dir != "." else file

                # use the *last* folder as group key (e.g. "smiley", "mountain")
                last_folder = os.path.basename(rel_dir) if rel_dir != "." else ""
                groups[last_folder].append(rel_path)

    # 2. Within each group (folder), make all combinations of 2
    pair_list = []
    for paths in groups.values():
        if len(paths) < 2:
            continue
        pair_list.extend(combinations(paths, 2))  # no cross-folder pairs

    # pairs = np.array(pair_list)
    # df_template = pd.DataFrame(pairs, columns=["A", "B"])

    rows = []
    for q in questionnaire_lst:
        for a, b in pair_list:
            rows.append({"A": a, "B": b, "question": q})

    df_new = pd.DataFrame(rows, columns=["A", "B", "question"])

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)

        # Ensure columns exist in old df
        for col in ["A", "B", "question"]:
            if col not in df_old.columns:
                df_old[col] = np.nan

        # Concatenate and drop duplicates by key (A,B,question)
        df_template = pd.concat([df_old, df_new], ignore_index=True)
        df_template = df_template.drop_duplicates(subset=["A", "B", "question"], keep="first")
    else:
        df_template = df_new

    return df_template

def open_saved_file(csv_path: str, pname: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # # Load or create
    # if os.path.exists(csv_path):
    #     df = pd.read_csv(csv_path)
    # else:
        # copy base, append required columns
    df = create_template(csv_path)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    save_to_csv(csv_path, df)

    return df

def get_imcomplete_row_indices(df: pd.DataFrame) -> List[int]:
    missing = df[REQUIRED_COLUMNS].isnull().any(axis=1)
    incomplete_rows = df[missing].copy()
    groups = [g for _, g in incomplete_rows.groupby(["A", "B"], sort=False)]
    rng = random.Random(int(time.time()))
    rng.shuffle(groups)
    shuffled = pd.concat(groups)
    return shuffled.index.to_list()

# ---- Per-participant state ----
class SessionState:
    def __init__(self, pname: str):
        self.pname = pname
        self.csv_path = f'{RESULT_DIR}/user_rating_{pname}.csv'
        self.df = open_saved_file(self.csv_path, pname)
        self.df_indices = get_imcomplete_row_indices(self.df)
        self.isswapped = False
        self.prev_a, prev_b = '', ''
        if not self.df_indices:
            self.df_indices = []  # no tasks left
            self.df_idx = None
        else:
            self.df_idx = self.df_indices.pop(0)
            if np.random.rand() < 0.5:
                self.isswapped = True
            else:
                self.isswapped = False
            self.prev_a, self.prev_b = self.current_pair()

        self.ui_idx = 0  # index into UI_ORDER
        self.lock = threading.Lock()  # file updates

    def current_pair(self):
        if self.df_idx is None:
            return None, None
        if not self.isswapped:
            a = self.df.iloc[self.df_idx]["A"]
            b = self.df.iloc[self.df_idx]["B"]
        else:
            b = self.df.iloc[self.df_idx]["A"]
            a = self.df.iloc[self.df_idx]["B"]
        return a, b

    def file_path_for_current(self) -> str | None:
        a, b = self.current_pair()
        if a is None or b is None:
            return None, None

        # URL for the browser
        return f'/{DATASET_DIR}/{a}', f'/{DATASET_DIR}/{b}'

    def prompt_for_current(self) -> str | None:
        a, _ = self.current_pair()
        return None if a is None else str(self.df[self.df_idx].iloc[2])

    def get_questionnaire(self) -> str | None:
        dat = self.df.iloc[self.df_idx]
        if dat is None:
            return None
        desc = str(dat.iloc[2])
        return desc

    def advance(self):
        if not self.df_indices:
            self.df_idx = None
            return
        self.ui_idx = (self.ui_idx + 1) % len(UI_ORDER)
        if self.ui_idx == 0:
            if not self.df_indices:
                self.df_idx = None
            else:
                self.df_idx = self.df_indices.pop(0)
                curr_a, curr_b = self.current_pair()
                if curr_a != self.prev_a or curr_b != self.prev_b:
                    if np.random.rand() < 0.5:
                        self.isswapped = True
                    else:
                        self.isswapped = False
                self.prev_a, self.prev_b = self.current_pair()

# ---- NiceGUI plumbing ----
# Serve dataset folder as /dataset so <video> can play files directly
app.add_static_files(f'/{DATASET_DIR}', DATASET_DIR)

# In-memory registry of sessions by participant name
sessions: Dict[str, SessionState] = {}

def get_session(pname: str) -> SessionState:
    if pname not in sessions:
        sessions[pname] = SessionState(pname)
    return sessions[pname]

def submit_event(state: SessionState, event: Dict[str, Any]) -> None:
    """Mirror your on_event() to update DF and save."""
    with state.lock:
        df = state.df
        idx = state.df_idx
        if idx is None:
            return

        pref = event['value']
        val = 0 if pref == 'A' else 1

        if state.isswapped:
            df.at[idx, "preferred"] = 1 - val
        else:
            df.at[idx, "preferred"] = val

        save_to_csv(state.csv_path, df)

        state.advance()

@ui.page('/{pname}')
def survey_page(pname: str):
    from datetime import datetime

    state = get_session(pname)

    # --- top-level layout elements (must NOT be nested) ---
    with ui.header().classes('p-3'):
        ui.label(f'Participant: {state.pname}').style('font-weight:600')
        ui.space()
        tasks_lbl = ui.label('')
        clock = ui.label(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    tasks_lbl.set_text(f'Tasks left: {len(state.df_indices) + (1 if state.df_idx is not None else 0)}')

    # main content container you will clear/re-render
    root = ui.column().classes('items-stretch w-full max-w-5xl mx-auto gap-3 p-2')

    with ui.footer().classes('justify-center p-2'):
        ui.label('By Daehwa Kim')

    # keep header clock ticking
    ui.timer(1.0, lambda: clock.set_text(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def render():
        root.clear()
        with root:
            # no header/footer here; only page content

            if state.df_idx is None:
                ui.markdown('## ✅ Thanks! There are no more items for you right now.')
                return

            step = UI_ORDER[state.ui_idx]
            question = state.get_questionnaire() or ''

            (img_a,img_b) = state.file_path_for_current()
            # prompt = state.prompt_for_current()
            with ui.card().classes('w-full p-4'):
                ui.label("Choose one between two drawings regarding to the question.").style('font-weight:600; white-space:normal;').classes('break-words')
                ui.label(f"\"{question}\"").style('color: black; font-size: 1.25rem;').classes('break-words')
                ui.separator()

                with ui.row().classes('w-full items-center justify-center gap-4 flex-nowrap'):
                    ui.image(img_a) \
                        .style('width:50%; max-height:70vh; object-fit:contain; background:#000; cursor:pointer;') \
                        .on('click', lambda: (submit_event(state, {'type': 'ab', 'value': 'A'}), render()))
                    ui.element('div').style(
                        'width:2px; align-self:stretch; background:#555; margin:0 12px;'
                    )
                    ui.image(img_b) \
                        .style('width:50%; max-height:70vh; object-fit:contain; background:#000; cursor:pointer') \
                        .on('click', lambda: (submit_event(state, {'type': 'ab', 'value': 'B'}), render()))

            tasks_lbl.set_text(f'Tasks left: {len(state.df_indices) + (1 if state.df_idx is not None else 0)}')

    render()

@ui.page('/')
def index():
    ui.markdown('# HRI Creative Co-Painting\nEnter your participant name to begin.')
    pid = ui.input('Participant name', placeholder='e.g., p1').classes('w-64')
    ui.button(
        'Start',
        on_click=lambda: ui.navigate.to(f'/{(pid.value or "p1").strip()}')
    )

# Expose ASGI app for uvicorn / gunicorn
app = app
if __name__ in {'__main__', '__mp_main__'}:
    ui.run(host='0.0.0.0', port=8080, title='HRI Creative Co-Painting', reload=False, show=False)
