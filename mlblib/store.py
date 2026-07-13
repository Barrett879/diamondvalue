"""Read-only access to the offline prediction files, plus table formatting.

The Streamlit pages import ONLY this (never the model). It loads the per-date
predictions parquet and slate JSON the daily pipeline wrote, and formats the
canonical display tables with the SENTINEL for missing values.
"""
from __future__ import annotations

import functools

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


@functools.lru_cache(maxsize=1)
def _uni_name_map() -> dict:
    """personId -> fullName from the committed player universe (latest year).
    Repairs rows where the daily pipeline left a raw MLB id as the name (a
    projected-lineup player who was off the current roster fetch)."""
    for y in range(2027, 2017, -1):
        u = cache.read_parquet_or_none(cache.dc_path(f"player_universe_{y}_v1.parquet"))
        if u is not None and {"personId", "fullName"} <= set(u.columns):
            return {int(pid): nm for pid, nm in zip(u["personId"], u["fullName"])
                    if pd.notna(pid) and isinstance(nm, str)}
    return {}


def _repair_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace any all-digits fullName (a leaked personId) with the real name
    from the universe, so tables AND props read correctly even on already-
    generated prediction files."""
    if df is None or "fullName" not in df.columns:
        return df
    digits = df["fullName"].astype(str).str.fullmatch(r"\d+")
    if not digits.any():
        return df
    names = _uni_name_map()
    df = df.copy()
    pid = df.get("personId")
    fixed = []
    for i, (nm, bad) in enumerate(zip(df["fullName"], digits)):
        key = None
        if bad:
            key = int(pid.iloc[i]) if pid is not None and pd.notna(pid.iloc[i]) else \
                (int(nm) if str(nm).isdigit() else None)
        fixed.append(names.get(key, nm) if bad and key is not None else nm)
    df["fullName"] = fixed
    return df


def load_predictions(date: str) -> pd.DataFrame | None:
    return _repair_names(cache.read_parquet_or_none(predictions_path(date)))


def load_actuals(date: str) -> pd.DataFrame | None:
    """Per-game ACTUAL box-score counts for `date` (from the committed gamelogs),
    keyed by (personId, gamePk). Present only once the daily bot has scored the
    day, so it doubles as the "game is final" signal. None when unscored."""
    season = int(date[:4])
    g = cache.read_parquet_or_none(cache.dc_path(f"gamelogs_{season}_v1.parquet"))
    if g is None or "officialDate" not in g.columns:
        return None
    day = g[(g["officialDate"] == date) & g.get("played", True)]
    return day if not day.empty else None


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


def _headshot(pid) -> str:
    """MLB headshot <img> for a personId. The generic-default URL never 404s
    (an unknown id returns a silhouette), so no onerror is needed. Empty string
    when no id."""
    if pid is None or (isinstance(pid, float) and pid != pid):
        return ""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return ""
    url = ("https://img.mlbstatic.com/mlb-photos/image/upload/"
           "d_people:generic:headshot:67:current.png/w_96,q_auto:best/"
           f"v1/people/{pid}/headshot/67/current")
    return f'<img class="dv-hs" src="{url}" alt="">'


def html_stat_table(str_df: pd.DataFrame, label_cols: int = 1,
                    hero: tuple = (), pids: list | None = None) -> str:
    """Render a formatted string table as a themed, responsive HTML table.
    The first `label_cols` columns are left-aligned text (slot/name); the rest
    are right-aligned tabular numbers. `hero` column headers get a subtle
    emphasis so the eye lands on the marquee stat. `pids` (personId per row, in
    str_df order) prepends a headshot to the name cell.
    """
    cols = list(str_df.columns)
    head = "".join(
        (f'<th class="l">{_esc(c)}</th>' if i < label_cols else f'<th>{_esc(c)}</th>')
        for i, c in enumerate(cols))
    name_idx = label_cols - 1   # the last label column is the player/pitcher name
    body = []
    for ri, (_, r) in enumerate(str_df.iterrows()):
        cells = []
        for i, c in enumerate(cols):
            if i < label_cols:
                cls = "name l" if i == name_idx else "slot l"
            else:
                cls = "hero" if c in hero else ""
            val = _esc(r[c])
            if i == name_idx and pids is not None and ri < len(pids):
                val = _headshot(pids[ri]) + val
            cells.append(f'<td class="{cls}">{val}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="dv-table-wrap"><table class="dv-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def _pids_of(df: pd.DataFrame) -> list | None:
    return df["personId"].tolist() if "personId" in df.columns else None


def html_batter_table(df: pd.DataFrame) -> str:
    return html_stat_table(format_batter_table(df), label_cols=2, hero=("HR", "TB"),
                           pids=_pids_of(df))


def html_pitcher_table(df: pd.DataFrame) -> str:
    return html_stat_table(format_pitcher_table(df), label_cols=1, hero=("K",),
                           pids=_pids_of(df))


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


def _props_actual(p: dict) -> str:
    """The result line under a posted line once the game is scored: what the
    player actually did vs the line and whether the model's lean was right.
    Informational results, never a wager outcome."""
    a = p.get("Actual")
    if a is None or (isinstance(a, float) and a != a):  # None or NaN: did not play
        return ""
    line, actual = float(p["Line"]), float(a)
    went = "over" if actual > line else ("under" if actual < line else "push")
    lean = str(p.get("Lean", "")).lower()
    right = (lean == "over" and went == "over") or (lean == "under" and went == "under")
    cls = "hit" if right else ("push" if went == "push" else "miss")
    verdict = ("model right" if right else "push" if went == "push" else "model off")
    return (f'<div class="dv-pactual {cls}">Actual <b>{actual:g}</b> '
            f'&middot; went {went} &middot; {verdict}</div>')


def _props_body(props: list) -> str:
    """The expanded panel for one player: their posted lines vs our model, plus
    the actual result once the game is final."""
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
            f"</div>{_props_meta(p)}{_props_actual(p)}</div>")
    return f'<div class="dv-xbody">{"".join(lines)}</div>'


# Display header -> (gamelog actual column, gamelog-to-projection divisor). Only
# IP differs: the gamelog stores outs, so /3 recovers innings.
_ACT_BAT = {"PA": ("PA", 1), "H": ("H", 1), "1B": ("b1", 1), "2B": ("b2", 1),
            "3B": ("b3", 1), "HR": ("HR", 1), "TB": ("TB", 1), "R": ("R", 1),
            "RBI": ("RBI", 1), "BB": ("BB", 1), "SO": ("SO", 1), "SB": ("SB", 1)}
_ACT_PIT = {"IP": ("p_outs", 3), "Pitches": ("p_pitches", 1), "K": ("p_K", 1),
            "BB": ("p_BB", 1), "H": ("p_H", 1), "ER": ("p_ER", 1)}


def _actual_num(v) -> str:
    """An actual count as a clean number: int when whole (H = 1), else 1 dp
    (IP = 5.0)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return SENTINEL
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.1f}"


def _actuals_grid(raw_row, actual_row, display, act_map) -> str:
    """Compact 'projected -> actual' chips for one player's finished game."""
    chips = []
    for col, hdr, dec in display:
        m = act_map.get(hdr)
        if m is None:
            continue
        gcol, div = m
        av = actual_row.get(gcol)
        if av is None or (isinstance(av, float) and av != av):
            continue
        proj = _fmt(raw_row.get(col), dec)
        chips.append(f'<span class="dv-ag"><i>{_esc(hdr)}</i>{proj} &rarr; '
                     f'<b>{_actual_num(float(av) / div)}</b></span>')
    if not chips:
        return ""
    return ('<div class="dv-agrid"><div class="dv-ag-h">Projected &rarr; actual'
            f'</div>{"".join(chips)}</div>')


def html_expandable_stat_table(str_df: pd.DataFrame, label_cols: int,
                               hero: tuple, props_by_name: dict,
                               raw_df: pd.DataFrame | None = None,
                               actuals_by_pid: dict | None = None,
                               display: list | None = None,
                               act_map: dict | None = None) -> str:
    """Like html_stat_table, but a player becomes a <details> that opens to
    their PrizePicks lines and/or, once their game is final, a projected-vs-
    actual grid. Players with neither stay plain rows. Native <details> = no JS.
    raw_df (aligned row-for-row with str_df) supplies personId + projections."""
    cols = list(str_df.columns)
    name_idx = label_cols - 1
    name_col = cols[name_idx]
    tmpl = _col_template(len(cols), label_cols)
    head = "".join(
        (f'<span class="l">{_esc(c)}</span>' if i < label_cols else f'<span>{_esc(c)}</span>')
        for i, c in enumerate(cols)) + "<span></span>"
    raw = raw_df.reset_index(drop=True) if raw_df is not None else None
    rows = []
    for i, (_, r) in enumerate(str_df.iterrows()):
        row_pid = raw.iloc[i].get("personId") if (raw is not None and i < len(raw)) else None
        cells = []
        for j, c in enumerate(cols):
            if j < label_cols:
                cls = "name l" if j == name_idx else "slot l"
            else:
                cls = "hero" if c in hero else ""
            val = _esc(r[c])
            if j == name_idx and row_pid is not None:
                val = _headshot(row_pid) + val
            cells.append(f'<span class="{cls}">{val}</span>')
        props = props_by_name.get(str(r[name_col]))
        actual_row = None
        if raw is not None and i < len(raw) and actuals_by_pid:
            pid = raw.iloc[i].get("personId")
            if pd.notna(pid):
                actual_row = actuals_by_pid.get(int(pid))
        body = ""
        if actual_row is not None and display and act_map:
            body += _actuals_grid(raw.iloc[i], actual_row, display, act_map)
        if props:
            body += _props_body(props)
        if props or actual_row is not None:
            badge = (f'<span class="xcaret has">{len(props)}</span>' if props
                     else '<span class="xcaret fin"></span>')
            cells.append(badge)
            rows.append(f'<details class="dv-xrow"><summary>{"".join(cells)}'
                        f"</summary>{body}</details>")
        else:
            cells.append('<span class="xcaret"></span>')
            rows.append(f'<div class="dv-xrow norow">{"".join(cells)}</div>')
    return (f'<div class="dv-table-wrap"><div class="dv-xtable" '
            f'style="--xt:{tmpl}"><div class="dv-xhead">{head}</div>'
            f'{"".join(rows)}</div></div>')


def html_expandable_batter_table(df: pd.DataFrame, props_by_name: dict,
                                 actuals_by_pid: dict | None = None) -> str:
    return html_expandable_stat_table(
        format_batter_table(df), 2, ("HR", "TB"), props_by_name,
        raw_df=df, actuals_by_pid=actuals_by_pid, display=BAT_DISPLAY, act_map=_ACT_BAT)


def html_expandable_pitcher_table(df: pd.DataFrame, props_by_name: dict,
                                  actuals_by_pid: dict | None = None) -> str:
    return html_expandable_stat_table(
        format_pitcher_table(df), 1, ("K",), props_by_name,
        raw_df=df, actuals_by_pid=actuals_by_pid, display=PIT_DISPLAY, act_map=_ACT_PIT)


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
