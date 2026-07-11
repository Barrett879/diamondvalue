"""Read-only access to the offline prediction files, plus table formatting.

The Streamlit pages import ONLY this (never the model). It loads the per-date
predictions parquet and slate JSON the daily pipeline wrote, and formats the
canonical display tables with the SENTINEL for missing values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cache
from . import model as M
from .theme import SENTINEL

# Canonical batter display columns: (prediction column, header, decimals).
BAT_DISPLAY = [
    ("slot", "Slot", 0), ("fullName", "Player", None), ("PA", "PA", 1),
    ("H", "H", 2), ("b2", "2B", 2), ("b3", "3B", 2), ("HR", "HR", 2),
    ("TB", "TB", 2), ("R", "R", 2), ("RBI", "RBI", 2), ("BB", "BB", 2),
    ("SO", "SO", 2), ("SB", "SB", 2), ("AVG", "AVG", 3), ("OBP", "OBP", 3),
    ("SLG", "SLG", 3), ("OPS", "OPS", 3),
]
PIT_DISPLAY = [
    ("fullName", "Pitcher", None), ("IP", "IP", 1), ("K", "K", 2),
    ("BB", "BB", 2), ("H", "H", 2), ("HR", "HR", 2), ("ER", "ER", 2),
]


def predictions_path(date: str):
    return cache.dc_path(f"predictions_{date.replace('-', '_')}_{M.MODEL_VERSION}.parquet")


def slate_meta_path(date: str):
    return cache.dc_path(f"slate_pred_{date.replace('-', '_')}_v1.json")


def load_predictions(date: str) -> pd.DataFrame | None:
    return cache.read_parquet_or_none(predictions_path(date))


def load_slate_meta(date: str):
    p = slate_meta_path(date)
    if not p.exists():
        return None
    try:
        return cache.json_load(p)
    except Exception:  # noqa: BLE001
        return None


def _fmt(val, decimals) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return SENTINEL
    if decimals is None:
        return str(val)
    if decimals == 0:
        try:
            return str(int(round(float(val))))
        except (ValueError, TypeError):
            return SENTINEL
    if decimals == 3:  # rate stats: .xxx with no leading zero, baseball style
        try:
            return f"{float(val):.3f}".lstrip("0") or "0"
        except (ValueError, TypeError):
            return SENTINEL
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return SENTINEL


def format_batter_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({header: _fmt(r.get(col), dec) for col, header, dec in BAT_DISPLAY})
    return pd.DataFrame(rows, columns=[h for _, h, _ in BAT_DISPLAY])


def format_pitcher_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({header: _fmt(r.get(col), dec) for col, header, dec in PIT_DISPLAY})
    return pd.DataFrame(rows, columns=[h for _, h, _ in PIT_DISPLAY])
