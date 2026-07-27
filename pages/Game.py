"""Game detail page: one game at its own URL (/Game?date=YYYY-MM-DD&gamePk=NNN),
showing both teams' full rosters with every predicted metric. When the day's
predictions have not been generated, it falls back to the posted lineups.
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import props_ui  # noqa: E402
from mlblib import fetch, parse, store  # noqa: E402
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

# The public host for shareable links. Minted explicitly because Streamlit
# Community Cloud serves the app in an iframe: the browser's address bar never
# reflects in-app navigation, so there is nothing correct to copy from it.
SHARE_BASE = "https://diamondvalue.streamlit.app"

date = st.query_params.get("date")
game_pk = st.query_params.get("gamePk")
away_q = (st.query_params.get("away") or "").strip().upper()
home_q = (st.query_params.get("home") or "").strip().upper()
gnum_q = st.query_params.get("game")   # doubleheader disambiguator (1-based)

_dark = st.session_state.get("theme_dark", False)
_theme = "dark" if _dark else "light"
back_href = f"/?date={date}&theme={_theme}" if date else f"/?theme={_theme}"
st.markdown(f'<a class="dv-back" href="{back_href}" target="_self">&larr; Back to slate</a>',
            unsafe_allow_html=True)

def _resolve_by_teams(date: str, away: str, home: str, gnum) -> int | None:
    """gamePk for the away@home matchup on `date` -- the human-readable URL
    (?date=&away=&home=) someone can read before clicking. Slate meta first
    (offline), live schedule as fallback; `gnum` picks a doubleheader game."""
    cand = [x for x in (store.load_slate_meta(date) or [])
            if str(x.get("away", "")).upper() == away
            and str(x.get("home", "")).upper() == home]
    if not cand:
        try:
            slate = fetch.get_slate(date, today=today_iso())
        except Exception:  # noqa: BLE001
            slate = []
        cand = [{"gamePk": g["gamePk"], "gameNumber": i + 1}
                for i, g in enumerate(
                    g for g in slate
                    if (g["away"].get("abbr") or "").upper() == away
                    and (g["home"].get("abbr") or "").upper() == home)]
    if not cand:
        return None
    if gnum:
        try:
            want = int(gnum)
            for x in cand:
                if int(x.get("gameNumber") or 1) == want:
                    return int(x["gamePk"])
        except (TypeError, ValueError):
            pass
    return int(cand[0]["gamePk"])


if not date or not (game_pk or (away_q and home_q)):
    st.info("No game selected. Return to the slate and pick a game.")
    render_footer()
    st.stop()

if game_pk:
    try:
        game_pk_int = int(game_pk)
    except (TypeError, ValueError):
        st.error("Invalid game reference.")
        render_footer()
        st.stop()
else:
    _pk = _resolve_by_teams(date, away_q, home_q, gnum_q)
    if _pk is None:
        st.info(f"No {away_q} at {home_q} game found on {date}.")
        render_footer()
        st.stop()
    game_pk_int = _pk


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


def _game_started(game_date_utc: str | None) -> bool:
    """True once the scheduled first pitch has passed (UTC compare)."""
    try:
        s = (game_date_utc or "").replace("Z", "+00:00")
        return dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return False


def _bat_table(df, pbn, abp, team):
    return (store.html_expandable_batter_table(df, pbn or {}, abp, team) if (pbn or abp)
            else store.html_batter_table(df, team))


def _render_side(side_df: pd.DataFrame, team_name: str, probable: str | None,
                 status: str, pbn: dict, abp: dict, team_color: str):
    sp = f"SP: <b>{probable}</b>" if probable else "SP: TBD"
    st.markdown(
        f'<div class="dv-team" style="--team:{team_color}">'
        f'<span class="dv-team-name">{team_name}</span>'
        f'<span class="dv-team-sp">{sp}</span>{_badge(status)}</div>',
        unsafe_allow_html=True)
    pit_df = side_df[side_df["role"] == "pit"]
    if not pit_df.empty:
        st.markdown('<div class="dv-eyebrow">Starting pitcher &middot; expected</div>',
                    unsafe_allow_html=True)
        tbl = (store.html_expandable_pitcher_table(pit_df, pbn or {}, abp, team_color)
               if (pbn or abp) else store.html_pitcher_table(pit_df, team_color))
        st.markdown(tbl, unsafe_allow_html=True)
    starters = side_df[(side_df["role"] == "bat") & (side_df["is_bench"] == False)].sort_values("slot")  # noqa: E712
    bench = side_df[(side_df["role"] == "bat") & (side_df["is_bench"] == True)]  # noqa: E712
    st.markdown('<div class="dv-eyebrow">Lineup &middot; expected per game</div>',
                unsafe_allow_html=True)
    st.markdown(_bat_table(starters, pbn, abp, team_color), unsafe_allow_html=True)
    if not bench.empty:
        st.markdown(f'<div class="dv-eyebrow">Bench ({len(bench)}) &middot; '
                    'expected if he starts</div>', unsafe_allow_html=True)
        st.markdown(_bat_table(bench, pbn, abp, team_color), unsafe_allow_html=True)


def _render_market_section(gp: pd.DataFrame, date: str) -> None:
    """This game's biggest-gaps highlight strip (the per-player detail lives in
    the expandable roster rows). The line input itself is the Update lines
    button in the header. Lines are already persisted by the caller so the
    roster expansions reflect them."""
    matched = props_ui.render_board(gp, date, scope_label="this game",
                                    show_ledger=False)
    if matched == 0:
        st.caption("No PrizePicks lines loaded for this game. Use the "
                   "**Update lines** button under the matchup to add them.")


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
    # Guarded because Streamlit Cloud can hot-reload this page before the
    # imported props_ui module reimports on a deploy -- degrade to plain rows
    # rather than crash during that brief window.
    try:
        props_ui.resolve_and_persist(date)
        pbn = props_ui.props_by_name(gp, date)
    except Exception:  # noqa: BLE001
        pbn = {}
    # Actuals once the game is scored (keyed by personId), so each played
    # player's row opens to projected-vs-actual and each prop shows its result.
    def _abp_from(frame):
        ga = frame[frame["gamePk"] == game_pk_int]
        return {int(r["personId"]): r for _, r in ga.iterrows() if pd.notna(r["personId"])}

    abp: dict = {}
    day_act = store.load_actuals(date)
    if day_act is not None:
        abp = _abp_from(day_act)
    # ── Actuals exist for EVERY game at all times: once first pitch has
    # passed and the committed gamelogs have not scored the game yet, the live
    # box score is pulled AUTOMATICALLY (2-minute session cache, so reruns do
    # not refetch and an in-progress game keeps updating on its own). The
    # Update actuals button just forces an immediate refresh. ────────────────
    def _pull_live_actuals() -> pd.DataFrame:
        bs = fetch.get_boxscore_raw(game_pk_int, force=True)
        rows = parse.parse_boxscore(bs, {
            "gamePk": game_pk_int, "officialDate": date,
            "gameDate": m.get("gameDate"),
            "gameNumber": m.get("gameNumber", 1),
            "venue_id": m.get("venue_id"), "dayNight": None}) if bs else []
        return pd.DataFrame([r for r in rows if r.get("played")])

    _lk = f"live_actuals_{game_pk_int}"
    _pref = f"live_pref_{game_pk_int}"
    _started = _game_started(m.get("gameDate"))
    _live_actuals = False
    # Auto-pull when the game is on and unscored; ALSO when the user pressed
    # Update actuals on a scored game (a fresh pull is authoritative -- a
    # committed gamelog can have been scored mid-game).
    if _started and (not abp or st.session_state.get(_pref)):
        entry = st.session_state.get(_lk)
        if not (isinstance(entry, dict) and time.time() - entry["at"] < 120):
            with st.spinner("Pulling the box score..."):
                try:
                    entry = {"at": time.time(), "df": _pull_live_actuals()}
                except Exception:  # noqa: BLE001 -- a feed hiccup must not break the page
                    entry = {"at": time.time(), "df": pd.DataFrame()}
            st.session_state[_lk] = entry
        if entry["df"] is not None and not entry["df"].empty:
            _live = _abp_from(entry["df"])
            if _live:
                abp = _live
                _live_actuals = True
    # ── Nav-bar actions: both controls live ON the top bar as buttons (the
    #    dv_nav_actions container position:fixes next to the theme toggle).
    #    The popover holds the same paste flow the old bottom expander did;
    #    guarded like the market section so a deploy-swap can't crash the view.
    with st.container(key="dv_nav_actions"):
        with st.popover("Update lines", help="Add or update PrizePicks lines"):
            try:
                props_ui.render_input(date)
            except Exception:  # noqa: BLE001
                st.caption("Line input is briefly unavailable (app updating).")
        # Update actuals is ALWAYS on the bar. Started game: force a fresh
        # box-score pull now (on a scored game the fresh pull is preferred
        # from here on -- committed gamelogs can have been scored mid-game).
        # Unstarted game: nothing to pull, say so.
        if st.button("Update actuals", key="upd_act",
                     help="Refresh this game's box score now"):
            if _started:
                st.session_state.pop(_lk, None)
                st.session_state[_pref] = True
                st.rerun()
            else:
                st.toast("This game hasn't started yet. Actuals appear "
                         "automatically at first pitch.")
        # Shareable link with the TEAMS and DATE in it. Minted here because
        # the streamlit.app address bar never follows in-app navigation (the
        # app lives in an iframe), so this is the only correct thing to copy.
        with st.popover("Share", help="Copy a link to this game"):
            _same = [x for x in meta
                     if str(x.get("away", "")).upper() == str(away).upper()
                     and str(x.get("home", "")).upper() == str(home).upper()]
            _share = f"{SHARE_BASE}/Game?date={date}&away={away}&home={home}"
            if len(_same) > 1:   # doubleheader: pin which game of the day
                _share += f"&game={int(m.get('gameNumber') or 1)}"
            st.caption("Anyone with this link opens exactly this game. Copy "
                       "from here -- the browser address bar doesn't follow "
                       "in-app navigation.")
            st.code(_share, language=None)
    if abp:
        st.caption(("Live box score, refreshed automatically. " if _live_actuals
                    else "")
                   + "Click a player to see projected vs actual"
                   + (" and how the posted lines landed" if pbn else "") + ".")
    else:
        note = ("No box-score data posted yet; actuals appear here "
                "automatically." if _started else
                "Actuals appear here automatically once the game starts.")
        if pbn:
            note += (" Players with a teal count have posted PrizePicks lines; "
                     "click the row to see them.")
        st.caption(note)
    # Stack the two teams full-width so every predicted column is readable
    # (side-by-side would squeeze the 17-column batter tables).
    tinfo = m.get("teams", {})
    for is_home, key in ((False, "away"), (True, "home")):
        side_df = gp[gp["isHome"] == is_home]
        t = tinfo.get(key, {})
        tcol, _ = team_color(t.get("abbr", SENTINEL))
        _render_side(side_df, t.get("abbr", SENTINEL), t.get("probable"),
                     t.get("lineup_status", "projected"), pbn, abp, tcol)
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
