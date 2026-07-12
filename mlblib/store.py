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

# Canonical display columns: (prediction column, header, decimals). Barrett's
# 2026-07-11 column decision: show exactly his 18 stats and hide the rest
# (AVG/OBP/SLG/OPS and HR-allowed hidden; 1B and Pitches added). The hidden
# derived rates are still computed in the predictions parquet.
BAT_DISPLAY = [
    ("slot", "Slot", 0), ("fullName", "Player", None), ("PA", "PA", 1),
    ("H", "H", 2), ("b1", "1B", 2), ("b2", "2B", 2), ("b3", "3B", 2),
    ("HR", "HR", 2), ("TB", "TB", 2), ("R", "R", 2), ("RBI", "RBI", 2),
    ("BB", "BB", 2), ("SO", "SO", 2), ("SB", "SB", 2),
]
PIT_DISPLAY = [
    ("fullName", "Pitcher", None), ("IP", "IP", 1), ("Pitches", "Pitches", 0),
    ("K", "K", 2), ("BB", "BB", 2), ("H", "H", 2), ("ER", "ER", 2),
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


# ── Themed HTML tables (replace the default st.dataframe widget) ─────────────
def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def html_stat_table(str_df: pd.DataFrame, label_cols: int = 1,
                    hero: tuple = ()) -> str:
    """Render a formatted string table as a themed, responsive HTML table.
    The first `label_cols` columns are left-aligned text (slot/name); the rest
    are right-aligned tabular numbers. `hero` column headers get a subtle
    emphasis so the eye lands on the marquee stat.
    """
    cols = list(str_df.columns)
    head = "".join(
        (f'<th class="l">{_esc(c)}</th>' if i < label_cols else f'<th>{_esc(c)}</th>')
        for i, c in enumerate(cols))
    name_idx = label_cols - 1   # the last label column is the player/pitcher name
    body = []
    for _, r in str_df.iterrows():
        cells = []
        for i, c in enumerate(cols):
            if i < label_cols:
                cls = "name l" if i == name_idx else "slot l"
            else:
                cls = "hero" if c in hero else ""
            cells.append(f'<td class="{cls}">{_esc(r[c])}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="dv-table-wrap"><table class="dv-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def html_batter_table(df: pd.DataFrame) -> str:
    return html_stat_table(format_batter_table(df), label_cols=2, hero=("HR", "TB"))


def html_pitcher_table(df: pd.DataFrame) -> str:
    return html_stat_table(format_pitcher_table(df), label_cols=1, hero=("K",))


def html_df(df: pd.DataFrame, label_cols: int = 1, hero: tuple = (),
            rename: dict | None = None) -> str:
    """Themed render of an arbitrary DataFrame (floats to 3 dp, ints plain).
    For the analytics tables (accuracy tracker) so every table on the site
    shares one look."""
    out = df.copy()
    if rename:
        out = out.rename(columns=rename)
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: SENTINEL if v != v else f"{v:.3f}")
        else:
            out[c] = out[c].astype(str)
    return html_stat_table(out, label_cols=label_cols, hero=hero)
