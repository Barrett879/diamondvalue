"""DiamondValue home / slate page.

Pick a date, then click a game to open its detail page (both teams' full rosters
with every predicted metric). This page is the slate index; the per-game metrics
live on the Game page (pages/Game.py) at its own URL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlblib import fetch, store  # noqa: E402
from mlblib.theme import (  # noqa: E402
    SENTINEL,
    render_footer,
    render_nav,
    render_page_chrome,
)
from mlblib.util import game_time_et, game_url, parse_iso_date, today_iso  # noqa: E402

st.set_page_config(page_title="DiamondValue", page_icon="static/favicon.svg",
                   layout="wide")
render_page_chrome()
render_nav("Home")

st.markdown(
    '<div class="dv-brand">Diamond<span class="accent">Value</span></div>'
    '<div class="dv-tagline">Per-game projections for every MLB slate</div>'
    '<div class="dv-note" style="margin-bottom:0.8rem">Every number is an '
    "expected value, not a prediction of what will happen.</div>",
    unsafe_allow_html=True,
)


# ── Date picker: seed once from ?date=, mirror changes back to the URL ───────
def _mirror_date():
    d = st.session_state.get("slate_date")
    if d:
        st.query_params["date"] = d.isoformat()


if "slate_date" not in st.session_state:
    st.session_state["slate_date"] = (
        parse_iso_date(st.query_params.get("date")) or parse_iso_date(today_iso()))


def _jump(days_delta: int):
    import datetime as _dt

    base = parse_iso_date(today_iso())
    st.session_state["slate_date"] = base + _dt.timedelta(days=days_delta)
    _mirror_date()


c_date, c_yday, c_today = st.columns([5, 1, 1])
with c_date:
    st.date_input("Game date", key="slate_date", on_change=_mirror_date)
with c_yday:
    st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    st.button("Yesterday", key="jump_yday", on_click=_jump, args=(-1,),
              use_container_width=True)
with c_today:
    st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    st.button("Today", key="jump_today", on_click=_jump, args=(0,),
              use_container_width=True)
_mirror_date()
date_iso = st.session_state["slate_date"].isoformat()
dark = st.session_state.get("theme_dark", False)


def _slate_rows():
    """Unified game list [(gamePk, away, home, et, status)] for the date.

    Prefers the committed slate-meta JSON (offline, no network) when the day's
    predictions exist; otherwise fetches the live schedule.
    """
    meta = store.load_slate_meta(date_iso)
    preds = store.load_predictions(date_iso)
    has_numbers = (preds is not None and not preds.empty
                   and preds.get("PA", None) is not None and preds["PA"].notna().any())
    if meta and has_numbers:
        rows = []
        for m in meta:
            tinfo = m.get("teams", {})
            statuses = [tinfo.get(k, {}).get("lineup_status", "projected")
                        for k in ("home", "away")]
            posted = sum(1 for s in statuses if s == "confirmed")
            rows.append((m["gamePk"], m.get("away", "AWY"), m.get("home", "HOM"),
                         game_time_et(m.get("gameDate")), _status_text(posted)))
        return rows
    # Live schedule fallback.
    slate = fetch.get_slate(date_iso, today=today_iso())
    rows = []
    for g in slate:
        posted = sum(1 for t in (g["home"], g["away"]) if t.get("lineup"))
        rows.append((g["gamePk"], g["away"].get("abbr") or g["away"].get("name") or SENTINEL,
                     g["home"].get("abbr") or g["home"].get("name") or SENTINEL,
                     game_time_et(g.get("gameDate")), _status_text(posted)))
    return rows


def _status_text(posted: int) -> str:
    return ("lineups posted" if posted == 2
            else "1 lineup posted" if posted == 1
            else "lineups not posted")


with st.spinner("Loading slate..."):
    rows = _slate_rows()

if not rows:
    st.info(f"No MLB games scheduled for {date_iso}.")
    render_footer()
    st.stop()

st.markdown(f"### {len(rows)} game{'s' if len(rows) != 1 else ''} on {date_iso}")
st.caption("Click a game to see both teams' rosters and every predicted stat.")

cards = []
for gpk, away, home, et, status in rows:
    href = game_url(date_iso, gpk, dark)
    cards.append(
        f'<a class="dv-game-card" href="{href}" target="_self">'
        f'<span class="dv-game-match">{away}<span class="at">@</span>{home}</span>'
        f'<span class="dv-game-right">'
        f'<span class="dv-game-time">{et} &middot; {status}</span>'
        f'<span class="dv-game-arrow">&rarr;</span>'
        f"</span></a>"
    )
st.markdown("".join(cards), unsafe_allow_html=True)

st.caption("Pitcher strikeouts are the most predictable per-game stat. Batter "
           "single-game numbers are low-signal by nature; treat every value as "
           "a distribution mean.")

render_footer()
