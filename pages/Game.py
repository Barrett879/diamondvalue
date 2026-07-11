"""Game detail page: one game at its own URL (/Game?date=YYYY-MM-DD&gamePk=NNN),
showing both teams' full rosters with every predicted metric. When the day's
predictions have not been generated, it falls back to the posted lineups.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import fetch, store  # noqa: E402
from mlblib.theme import (  # noqa: E402
    SENTINEL,
    render_footer,
    render_nav,
    render_page_chrome,
)
from mlblib.util import game_time_et, today_iso  # noqa: E402

st.set_page_config(page_title="Game · DiamondValue", page_icon="static/favicon.svg",
                   layout="wide")
render_page_chrome()
render_nav("")

date = st.query_params.get("date")
game_pk = st.query_params.get("gamePk")

_dark = st.session_state.get("theme_dark", False)
_theme = "dark" if _dark else "light"
back_href = f"/?date={date}&theme={_theme}" if date else f"/?theme={_theme}"
st.markdown(f'<a class="dv-back" href="{back_href}" target="_self">&larr; Back to slate</a>',
            unsafe_allow_html=True)

if not date or not game_pk:
    st.info("No game selected. Return to the slate and pick a game.")
    render_footer()
    st.stop()

try:
    game_pk_int = int(game_pk)
except (TypeError, ValueError):
    st.error("Invalid game reference.")
    render_footer()
    st.stop()


def _badge(status: str) -> str:
    label = "Lineup posted" if status == "confirmed" else "Projected lineup"
    return f'<span class="dv-badge {status}">{label}</span>'


def _render_side(side_df: pd.DataFrame, team_name: str, probable: str | None,
                 status: str):
    st.markdown(f"### {team_name}", unsafe_allow_html=True)
    st.markdown(f"<span class='dv-note'>SP: {probable or 'TBD'}</span> {_badge(status)}",
                unsafe_allow_html=True)
    pit_df = side_df[side_df["role"] == "pit"]
    if not pit_df.empty:
        st.caption("Starting pitcher (expected)")
        st.dataframe(store.format_pitcher_table(pit_df), hide_index=True,
                     use_container_width=True)
    starters = side_df[(side_df["role"] == "bat") & (side_df["is_bench"] == False)].sort_values("slot")  # noqa: E712
    bench = side_df[(side_df["role"] == "bat") & (side_df["is_bench"] == True)]  # noqa: E712
    st.caption("Lineup (expected per game)")
    st.dataframe(store.format_batter_table(starters), hide_index=True,
                 use_container_width=True)
    if not bench.empty:
        st.caption(f"Bench ({len(bench)}) — expected per game if he starts")
        st.dataframe(store.format_batter_table(bench), hide_index=True,
                     use_container_width=True)


# ── Try the generated predictions first ──────────────────────────────────────
preds = store.load_predictions(date)
meta = store.load_slate_meta(date)
has_numbers = (preds is not None and not preds.empty
               and preds.get("PA", pd.Series(dtype=float)).notna().any())

if has_numbers and meta:
    gp = preds[preds["gamePk"] == game_pk_int]
    m = next((x for x in meta if x["gamePk"] == game_pk_int), {})
    if gp.empty or not m:
        st.info("That game is not on the selected date's slate.")
        render_footer()
        st.stop()
    et = game_time_et(m.get("gameDate"))
    away, home = m.get("away", "AWY"), m.get("home", "HOM")
    st.markdown(f'<div class="dv-brand">{away} <span class="accent">@</span> {home}</div>'
                f'<div class="dv-tagline">{date} &nbsp; {et}</div>',
                unsafe_allow_html=True)
    # Stack the two teams full-width so every predicted column is readable
    # (side-by-side would squeeze the 17-column batter tables).
    tinfo = m.get("teams", {})
    for is_home, key in ((False, "away"), (True, "home")):
        side_df = gp[gp["isHome"] == is_home]
        t = tinfo.get(key, {})
        _render_side(side_df, t.get("abbr", SENTINEL), t.get("probable"),
                     t.get("lineup_status", "projected"))
        st.divider()
    st.caption("Every number is an expected value, the mean of a distribution, "
               "not a prediction of what will happen. See the About page.")
    render_footer()
    st.stop()

# ── Fallback: no predictions for this date, show the posted lineups ───────────
with st.spinner("Loading game..."):
    slate = fetch.get_slate(date, today=today_iso())
g = next((x for x in slate if x["gamePk"] == game_pk_int), None)
if g is None:
    st.info(f"Game not found on {date}.")
    render_footer()
    st.stop()

et = game_time_et(g.get("gameDate"))
a = g["away"]["abbr"] or g["away"]["name"]
h = g["home"]["abbr"] or g["home"]["name"]
st.markdown(f'<div class="dv-brand">{a} <span class="accent">@</span> {h}</div>'
            f'<div class="dv-tagline">{date} &nbsp; {et}</div>', unsafe_allow_html=True)
st.info("Predictions for this date have not been generated yet. Showing the "
        "posted lineups and probable starters.")
for team in (g["away"], g["home"]):
    prob = team.get("probable")
    st.markdown(f"### {team.get('name')}")
    st.markdown(f"<span class='dv-note'>SP: {prob['name'] if prob else 'TBD'}</span>",
                unsafe_allow_html=True)
    lineup = team.get("lineup") or []
    if lineup:
        st.markdown("".join(
            f"<div class='dv-note'>{i + 1}. {p['name']} "
            f"<span style='color:var(--fg-5)'>{p.get('pos') or ''}</span></div>"
            for i, p in enumerate(lineup)), unsafe_allow_html=True)
    else:
        st.markdown("<div class='dv-note'>Lineup not yet posted.</div>",
                    unsafe_allow_html=True)
    st.divider()

render_footer()
