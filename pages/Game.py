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

import props_ui  # noqa: E402
from mlblib import fetch, store  # noqa: E402
from mlblib.teams import team_color  # noqa: E402
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


def _hero(away: str, home: str, date: str, et: str) -> None:
    """Team-colored game header: the AWAY @ HOME wordmark with each abbreviation
    underlined in its team color, over the date/time line."""
    ac, _ = team_color(away)
    hc, _ = team_color(home)
    st.markdown(
        f'<div class="dv-game-hero" style="--away:{ac};--home:{hc}">'
        f'<div class="dv-brand"><span class="dv-hero-away">{away}</span> '
        f'<span class="accent">@</span> '
        f'<span class="dv-hero-home">{home}</span></div>'
        f'<div class="dv-tagline">{date} &nbsp; {et}</div></div>',
        unsafe_allow_html=True)


def _badge(status: str) -> str:
    label = "Lineup posted" if status == "confirmed" else "Projected lineup"
    return f'<span class="dv-badge {status}">{label}</span>'


def _bat_table(df, pbn):
    return (store.html_expandable_batter_table(df, pbn) if pbn
            else store.html_batter_table(df))


def _render_side(side_df: pd.DataFrame, team_name: str, probable: str | None,
                 status: str, pbn: dict):
    sp = f"SP: <b>{probable}</b>" if probable else "SP: TBD"
    st.markdown(
        f'<div class="dv-team"><span class="dv-team-name">{team_name}</span>'
        f'<span class="dv-team-sp">{sp}</span>{_badge(status)}</div>',
        unsafe_allow_html=True)
    pit_df = side_df[side_df["role"] == "pit"]
    if not pit_df.empty:
        st.markdown('<div class="dv-eyebrow">Starting pitcher &middot; expected</div>',
                    unsafe_allow_html=True)
        tbl = (store.html_expandable_pitcher_table(pit_df, pbn) if pbn
               else store.html_pitcher_table(pit_df))
        st.markdown(tbl, unsafe_allow_html=True)
    starters = side_df[(side_df["role"] == "bat") & (side_df["is_bench"] == False)].sort_values("slot")  # noqa: E712
    bench = side_df[(side_df["role"] == "bat") & (side_df["is_bench"] == True)]  # noqa: E712
    st.markdown('<div class="dv-eyebrow">Lineup &middot; expected per game</div>',
                unsafe_allow_html=True)
    st.markdown(_bat_table(starters, pbn), unsafe_allow_html=True)
    if not bench.empty:
        st.markdown(f'<div class="dv-eyebrow">Bench ({len(bench)}) &middot; '
                    'expected if he starts</div>', unsafe_allow_html=True)
        st.markdown(_bat_table(bench, pbn), unsafe_allow_html=True)


def _render_market_section(gp: pd.DataFrame, date: str) -> None:
    """The one-stop PrizePicks section: this game's biggest-gaps highlight
    strip (the per-player detail now lives in the expandable roster rows), and
    the line input inline (expanded when nothing is loaded yet). Lines are
    already persisted by the caller so the roster expansions reflect them."""
    matched = props_ui.render_board(gp, date, scope_label="this game",
                                    show_ledger=False)
    with st.expander("Add / update PrizePicks lines", expanded=matched == 0):
        props_ui.render_input(date)


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
    _hero(away, home, date, et)
    # Persist any freshly-entered lines FIRST, so the expandable roster rows
    # reflect the latest paste; then group them by player for the rows.
    props_ui.resolve_and_persist(date)
    pbn = props_ui.props_by_name(gp, date)
    if pbn:
        st.caption("Players with a teal count have posted PrizePicks lines; "
                   "click the row to see them.")
    # Stack the two teams full-width so every predicted column is readable
    # (side-by-side would squeeze the 17-column batter tables).
    tinfo = m.get("teams", {})
    for is_home, key in ((False, "away"), (True, "home")):
        side_df = gp[gp["isHome"] == is_home]
        t = tinfo.get(key, {})
        _render_side(side_df, t.get("abbr", SENTINEL), t.get("probable"),
                     t.get("lineup_status", "projected"), pbn)
        st.divider()
    try:
        _render_market_section(gp, date)
    except Exception:  # noqa: BLE001 — the market section must never break the game view
        pass
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
_hero(a, h, date, et)
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
