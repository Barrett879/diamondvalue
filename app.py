"""DiamondValue home / slate page.

Shows, for a chosen date, every game with predicted per-game stats for each
player. When the day's predictions file has not been built yet (or a model is
unavailable) it falls back to the live schedule view: games, probables, and
posted lineups. The date-picker machinery is the Phase 1.5 foundation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlblib import fetch, store  # noqa: E402
from mlblib.theme import (  # noqa: E402
    SENTINEL,
    render_footer,
    render_nav,
    render_page_chrome,
    render_theme_toggle,
)
from mlblib.util import game_time_et, parse_iso_date, today_iso  # noqa: E402

st.set_page_config(page_title="DiamondValue", page_icon="static/favicon.svg",
                   layout="wide")
render_page_chrome()
render_nav("Home")

st.markdown(
    '<div class="dv-brand">Diamond<span class="accent">Value</span></div>'
    '<div class="dv-tagline">Per-game player projections for every MLB slate. '
    "Every number is an expected value, not a prediction of what will happen."
    "</div>",
    unsafe_allow_html=True,
)
render_theme_toggle()


# ── Date picker: seed once from ?date=, mirror changes back to the URL ───────
def _mirror_date():
    d = st.session_state.get("slate_date")
    if d:
        st.query_params["date"] = d.isoformat()


if "slate_date" not in st.session_state:
    st.session_state["slate_date"] = (
        parse_iso_date(st.query_params.get("date")) or parse_iso_date(today_iso()))

st.date_input("Game date", key="slate_date", on_change=_mirror_date)
_mirror_date()
date_iso = st.session_state["slate_date"].isoformat()


def _badge(status: str) -> str:
    label = "Lineup posted" if status == "confirmed" else "Projected lineup"
    return f'<span class="dv-badge {status}">{label}</span>'


def _render_side(side_df: pd.DataFrame, team_abbr: str, probable: str | None,
                 status: str, pit_df: pd.DataFrame):
    st.markdown(
        f"<b>{team_abbr}</b> &nbsp; <span class='dv-note'>SP: {probable or 'TBD'}</span> "
        f"{_badge(status)}", unsafe_allow_html=True)
    if not pit_df.empty:
        st.caption("Starting pitcher (expected)")
        st.dataframe(store.format_pitcher_table(pit_df), hide_index=True,
                     use_container_width=True)
    starters = side_df[(side_df["role"] == "bat") & (side_df["is_bench"] == False)].sort_values("slot")  # noqa: E712
    bench = side_df[(side_df["role"] == "bat") & (side_df["is_bench"] == True)]  # noqa: E712
    st.caption("Lineup (expected)")
    st.dataframe(store.format_batter_table(starters), hide_index=True,
                 use_container_width=True)
    if not bench.empty:
        # Bench rendered inline (Streamlit forbids an expander nested inside the
        # game expander); the caption makes the conditional framing explicit.
        st.caption(f"Bench ({len(bench)}) — per game if he starts")
        st.dataframe(store.format_batter_table(bench), hide_index=True,
                     use_container_width=True)


def _render_predictions(preds: pd.DataFrame, meta: list) -> None:
    st.markdown(f"### {len(meta)} game{'s' if len(meta) != 1 else ''} on {date_iso}")
    meta_by_pk = {m["gamePk"]: m for m in meta}
    for gpk, gp in preds.groupby("gamePk"):
        m = meta_by_pk.get(gpk, {})
        et = game_time_et(m.get("gameDate"))
        away, home = m.get("away", "AWY"), m.get("home", "HOM")
        gnum = m.get("gameNumber", 1)
        dh = f"  (G{gnum})" if gnum and gnum > 1 else ""
        label = f"{away}  @  {home}      {et}{dh}"
        with st.expander(label):
            c_away, c_home = st.columns(2)
            for col, is_home in ((c_away, False), (c_home, True)):
                with col:
                    side_df = gp[gp["isHome"] == is_home]
                    if side_df.empty:
                        st.caption("No data")
                        continue
                    team_abbr = (m.get("teams", {}).get("home" if is_home else "away", {})
                                 .get("abbr", SENTINEL))
                    status = (m.get("teams", {}).get("home" if is_home else "away", {})
                              .get("lineup_status", "projected"))
                    probable = (m.get("teams", {}).get("home" if is_home else "away", {})
                                .get("probable"))
                    pit_df = side_df[side_df["role"] == "pit"]
                    _render_side(side_df, team_abbr, probable, status, pit_df)


def _render_schedule_only() -> None:
    """Fallback when no predictions file exists for the date."""
    with st.spinner("Loading slate..."):
        slate = fetch.get_slate(date_iso, today=today_iso())
    if not slate:
        st.info(f"No MLB games scheduled for {date_iso}.")
        return
    st.markdown(f"### {len(slate)} game{'s' if len(slate) != 1 else ''} on {date_iso}")
    st.caption("Predictions for this date have not been generated yet. Showing "
               "the schedule, probable starters, and posted lineups.")
    for g in slate:
        away, home = g["away"], g["home"]
        et = game_time_et(g.get("gameDate"))
        a = away.get("abbr") or away.get("name") or SENTINEL
        h = home.get("abbr") or home.get("name") or SENTINEL
        with st.expander(f"{a}  @  {h}      {et}"):
            c_away, c_home = st.columns(2)
            for col, team in ((c_away, away), (c_home, home)):
                with col:
                    prob = team.get("probable")
                    st.markdown(f"<b>{team.get('name')}</b> "
                                f"<span class='dv-note'>SP: {prob['name'] if prob else 'TBD'}</span>",
                                unsafe_allow_html=True)
                    lineup = team.get("lineup") or []
                    if lineup:
                        st.markdown("".join(
                            f"<div class='dv-note'>{i + 1}. {p['name']}</div>"
                            for i, p in enumerate(lineup)), unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='dv-note'>Lineup not yet posted.</div>",
                                    unsafe_allow_html=True)


preds = store.load_predictions(date_iso)
meta = store.load_slate_meta(date_iso)
has_numbers = preds is not None and not preds.empty and preds.get("PA") is not None \
    and preds["PA"].notna().any()

if has_numbers and meta:
    _render_predictions(preds, meta)
else:
    _render_schedule_only()

st.caption("Pitcher strikeouts are the most predictable per-game stat. Batter "
           "single-game numbers are low-signal by nature; treat every value as "
           "a distribution mean. See the About page for how this is built.")

render_footer()
