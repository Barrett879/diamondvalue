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


# ── Expandable stat tables (each player's row opens to their PrizePicks lines) ─
def _col_template(n_cols: int, label_cols: int) -> str:
    """CSS grid template matching the stat columns, plus a trailing caret col."""
    if label_cols >= 2:
        return f"2.6rem minmax(5rem, 1.7fr) repeat({n_cols - 2}, 1fr) 1.6rem"
    return f"minmax(6rem, 1.7fr) repeat({n_cols - 1}, 1fr) 1.6rem"


_DIR_LABEL = {"both": "More &amp; Less", "more": "More only", "less": "Less only"}


def _props_meta(p: dict) -> str:
    """The sub-line under a posted line: which side(s) PrizePicks offers, the
    Demon/Goblin payout tag, and a note when the model's own lean is a side the
    market does not offer. Informational market facts, not a wager prompt."""
    direction = str(p.get("Direction", "") or "")
    odds = str(p.get("OddsType", "") or "").lower()
    bits = []
    label = _DIR_LABEL.get(direction)
    if label:
        bits.append(f'<span class="pd">{label}</span>')
    if odds in ("demon", "goblin"):
        bits.append(f'<span class="podds {odds}">{odds.capitalize()}</span>')
    meta = (f'<div class="dv-pmeta">{"".join(bits)}</div>') if bits else ""
    # Cross-reference: More is the Over side, Less the Under side. If the model
    # leans a side that is not among the offered directions, flag it.
    lean = str(p.get("Lean", ""))
    model_side = "more" if lean == "Over" else "less" if lean == "Under" else ""
    if model_side and direction in ("more", "less") and direction != model_side:
        meta += (f'<div class="dv-pxwarn">model&rsquo;s {_esc(lean)} side '
                 f"not offered</div>")
    return meta


def _props_body(props: list) -> str:
    """The expanded panel for one player: their posted lines vs our model."""
    lines = []
    for p in props:
        edge = float(p["Edge"])
        d = "over" if edge > 0 else ("under" if edge < 0 else "")
        lines.append(
            f'<div class="dv-prow">'
            f'<div class="dv-pline">'
            f'<span class="ps">{_esc(p["Stat"])}</span>'
            f'<span class="pv">{float(p["Model"]):g} <i>vs</i> {float(p["Line"]):g}</span>'
            f'<span class="pe {d}">{edge:+g} {_esc(p["Lean"])}</span>'
            f"</div>{_props_meta(p)}</div>")
    return f'<div class="dv-xbody">{"".join(lines)}</div>'


def html_expandable_stat_table(str_df: pd.DataFrame, label_cols: int,
                               hero: tuple, props_by_name: dict) -> str:
    """Like html_stat_table, but each player whose name is in `props_by_name`
    becomes a <details> that opens to their PrizePicks lines. Players without
    posted lines stay plain rows. Native <details> = no JS."""
    cols = list(str_df.columns)
    name_idx = label_cols - 1
    name_col = cols[name_idx]
    tmpl = _col_template(len(cols), label_cols)
    head = "".join(
        (f'<span class="l">{_esc(c)}</span>' if i < label_cols else f'<span>{_esc(c)}</span>')
        for i, c in enumerate(cols)) + "<span></span>"
    rows = []
    for _, r in str_df.iterrows():
        cells = []
        for i, c in enumerate(cols):
            if i < label_cols:
                cls = "name l" if i == name_idx else "slot l"
            else:
                cls = "hero" if c in hero else ""
            cells.append(f'<span class="{cls}">{_esc(r[c])}</span>')
        props = props_by_name.get(str(r[name_col]))
        if props:
            cells.append(f'<span class="xcaret has">{len(props)}</span>')
            rows.append(f'<details class="dv-xrow"><summary>{"".join(cells)}'
                        f"</summary>{_props_body(props)}</details>")
        else:
            cells.append('<span class="xcaret"></span>')
            rows.append(f'<div class="dv-xrow norow">{"".join(cells)}</div>')
    return (f'<div class="dv-table-wrap"><div class="dv-xtable" '
            f'style="--xt:{tmpl}"><div class="dv-xhead">{head}</div>'
            f'{"".join(rows)}</div></div>')


def html_expandable_batter_table(df: pd.DataFrame, props_by_name: dict) -> str:
    return html_expandable_stat_table(format_batter_table(df), 2, ("HR", "TB"),
                                      props_by_name)


def html_expandable_pitcher_table(df: pd.DataFrame, props_by_name: dict) -> str:
    return html_expandable_stat_table(format_pitcher_table(df), 1, ("K",),
                                      props_by_name)


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
