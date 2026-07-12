"""DiamondValue home / slate page.

Pick a date, then click a game to open its detail page (both teams' full rosters
with every predicted metric). This page is the slate index; the per-game metrics
live on the Game page (pages/Game.py) at its own URL.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlblib import fetch, store  # noqa: E402
from mlblib.teams import team_color  # noqa: E402
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


# ── Date state: seed once from ?date=, mirror changes back to the URL ────────
def _mirror_date():
    d = st.session_state.get("slate_date")
    if d:
        st.query_params["date"] = d.isoformat()


if "slate_date" not in st.session_state:
    st.session_state["slate_date"] = (
        parse_iso_date(st.query_params.get("date")) or parse_iso_date(today_iso()))


def _jump(days_delta: int):
    """Absolute jump: Yesterday/Today relative to the real current date."""
    st.session_state["slate_date"] = (parse_iso_date(today_iso())
                                      + _dt.timedelta(days=days_delta))
    _mirror_date()


def _step(days_delta: int):
    """Relative step by one day (the chevrons)."""
    cur = st.session_state.get("slate_date") or parse_iso_date(today_iso())
    st.session_state["slate_date"] = cur + _dt.timedelta(days=days_delta)
    _mirror_date()


date_iso = st.session_state["slate_date"].isoformat()
dark = st.session_state.get("theme_dark", False)


def _status_text(posted: int) -> str:
    return ("lineups posted" if posted == 2
            else "1 lineup posted" if posted == 1
            else "lineups not posted")


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


def _slate_summary(rows):
    """(n_games, earliest first-pitch 'H:MM', lineups-posted %) for the readout."""
    n = len(rows)
    if not n:
        return 0, SENTINEL, 0
    times = []
    for r in rows:
        try:
            times.append(_dt.datetime.strptime(r[3].replace(" ET", ""),
                                               "%I:%M %p").time())
        except (ValueError, AttributeError):
            pass
    first = min(times).strftime("%-I:%M") if times else SENTINEL
    posted = sum({"lineups posted": 2, "1 lineup posted": 1}.get(r[4], 0) for r in rows)
    return n, first, round(100 * posted / (2 * n))


with st.spinner("Loading slate..."):
    rows = _slate_rows()

# ── Editorial masthead with a live slate readout ────────────────────────────
n_games, first_pitch, lineups_pct = _slate_summary(rows)
st.markdown(
    '<div class="dv-masthead"><div class="dv-mast-rule"></div>'
    '<div class="dv-mast-row"><div class="dv-mast-brand">'
    '<div class="dv-kicker"><span class="dv-diamond"></span>MLB Projections</div>'
    '<div class="dv-brand">Diamond<span class="accent">Value</span></div></div>'
    '<div class="dv-mast-summary">'
    f'<div class="dv-sum"><div class="dv-sum-num">{n_games}</div>'
    '<div class="dv-sum-lab">Games</div></div>'
    f'<div class="dv-sum"><div class="dv-sum-num">{first_pitch}</div>'
    '<div class="dv-sum-lab">First Pitch ET</div></div>'
    f'<div class="dv-sum"><div class="dv-sum-num">{lineups_pct}%</div>'
    '<div class="dv-sum-lab">Lineups In</div></div>'
    '</div></div>'
    '<div class="dv-deck">Per-game expected values for every hitter and pitcher '
    'on the slate. Every number is a distribution mean, not a prediction of '
    'what will happen.</div></div>',
    unsafe_allow_html=True,
)

# ── Date control bar: chevron step / date / Yesterday / Today ───────────────
with st.container(key="dv_datebar"):
    c_prev, c_date, c_next, c_yday, c_today = st.columns(
        [0.6, 4, 0.6, 1.5, 1.4], gap="small", vertical_alignment="bottom")
    with c_prev:
        st.button("‹", key="step_prev", on_click=_step, args=(-1,),
                  use_container_width=True, help="Previous day")
    with c_date:
        st.date_input("Slate date", key="slate_date", on_change=_mirror_date)
    with c_next:
        st.button("›", key="step_next", on_click=_step, args=(1,),
                  use_container_width=True, help="Next day")
    with c_yday:
        st.button("Yesterday", key="jump_yday", on_click=_jump, args=(-1,),
                  use_container_width=True)
    with c_today:
        st.button("Today", key="jump_today", on_click=_jump, args=(0,),
                  use_container_width=True)
st.markdown('<div class="dv-bar-rule"></div>', unsafe_allow_html=True)

if not rows:
    st.info(f"No MLB games scheduled for {date_iso}.")
    render_footer()
    st.stop()

# ── Slate-head divider + status legend ──────────────────────────────────────
head_date = st.session_state["slate_date"].strftime("%a %b %-d")
st.markdown(
    '<div class="dv-slate-head">'
    f'<div class="dv-slate-count"><b>{len(rows)}</b> game{"s" if len(rows) != 1 else ""}'
    f'<span class="sep">&middot;</span><span class="date">{head_date}</span></div>'
    '<div class="dv-legend">'
    '<span class="lg-set"><i></i>Lineups set</span>'
    '<span class="lg-part"><i></i>1 posted</span>'
    '<span class="lg-none"><i></i>Not posted</span>'
    '</div></div>', unsafe_allow_html=True)

_STATUS_CLS = {"lineups posted": "s-posted", "1 lineup posted": "s-partial",
               "lineups not posted": "s-none"}
cards = []
for gpk, away, home, et, status in rows:
    href = game_url(date_iso, gpk, dark)
    scls = _STATUS_CLS.get(status, "s-none")
    ap, _ = team_color(away)
    hp, _ = team_color(home)
    cards.append(
        f'<a class="dv-slate-card" href="{href}" target="_self" '
        f'style="--away:{ap};--home:{hp}">'
        f'<span class="dv-slate-away">{away}</span>'
        f'<span class="dv-slate-at">at</span>'
        f'<span class="dv-slate-home">{home}</span>'
        f'<span class="dv-slate-time">{et}</span>'
        f'<span class="dv-slate-status {scls}">{status}</span>'
        f"</a>"
    )
st.markdown(f'<div class="dv-slate-grid">{"".join(cards)}</div>',
            unsafe_allow_html=True)

st.caption("Pitcher strikeouts are the most predictable per-game stat. Batter "
           "single-game numbers are low-signal by nature; treat every value as "
           "a distribution mean.")

render_footer()
