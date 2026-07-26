"""Shared PrizePicks props UI (line input + model-vs-market board).

Consolidated from the old standalone Props page so the whole feature lives
wherever the user already is -- the game detail page and the home slate -- with
no separate tab. Line input persists slate-wide (per date), so pasting/pulling
on any page flows to every game.

An informational model-vs-market view, never a wager recommendation.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd
import streamlit as st

from mlblib import props, store
from mlblib.util import today_iso


def _fmt_date(date_iso: str) -> str:
    """'2026-07-17' -> 'Thu Jul 17' (falls back to the raw string)."""
    try:
        return _dt.date.fromisoformat(date_iso).strftime("%a %b %-d")
    except ValueError:
        return date_iso

def resolve_and_persist(date_iso: str):
    """The pasted text (feed JSON, board text, or a bare list -- the paste box
    is the only source) MERGED into the previously-saved set and persisted, so
    every page/game reads the same accumulating board. PrizePicks has no All
    tab, so each paste ADDS to what is saved (a re-paste of the same tab just
    refreshes those lines).

    Lines are saved under the date of the GAME they are for (from each prop's
    start_time), not the slate date on screen: the pre-game board posted
    tonight is tomorrow's games, and saving it under today would strand it
    against the wrong slate. Lines with no start_time (board-text pastes, bare
    lists) stay on the on-screen date. render_input reads the per-date state
    back from DISK (props.saved_line_dates), so its pointers survive reruns
    and later pastes; only the no-player-names diagnostic rides session state
    (st.session_state['pp_parse_note']).

    Returns the saved frame for `date_iso` (or None). Safe to call at the top
    of a page before the input widgets render (it reads their committed
    session_state values). 'Clear all' (_clear_lines) resets the set."""
    lines = None
    note = ""
    txt = (st.session_state.get("pp_paste") or "").strip()
    if txt:
        got = props.parse_any(txt)   # JSON, board text, or a simple list
        if got is not None and got.empty and got.attrs.get("skipped_noname"):
            note = (f"That JSON has {got.attrs['skipped_noname']} props but "
                    "no player names (its player list is missing), so "
                    "nothing could be saved. Re-open the feed link and copy "
                    "the WHOLE page, or paste the board text instead.")
        lines = got if (got is not None and not got.empty) else None
    if lines is not None and not lines.empty:
        for d, batch in props.bucket_by_date(lines).items():
            dest = d or date_iso
            merged = props.merge_lines(props.load_lines(dest), batch)
            props.save_lines(dest, merged)   # accumulate across stat tabs
    st.session_state["pp_parse_note"] = note
    return props.load_lines(date_iso)        # what THIS page's date has saved


def props_by_name(scope_preds: pd.DataFrame, date_iso: str) -> dict:
    """{fullName: [ {Stat, Model, Line, Edge, Lean}, ... ]} for the players in
    scope_preds that have a posted, mappable line. Feeds the expandable roster
    rows on the game page."""
    lines = props.load_lines(date_iso)
    if lines is None or lines.empty:
        return {}
    table, _ = props.compare(lines, scope_preds, actuals=store.load_actuals(date_iso))
    out: dict = {}
    for _, r in table.iterrows():
        out.setdefault(r["Player"], []).append(
            {"Stat": r["Stat"], "Model": r["Model"], "Line": r["Line"],
             "Edge": r["Edge"], "Lean": r["Lean"],
             "Direction": r.get("Direction", ""), "OddsType": r.get("OddsType", ""),
             "Actual": r.get("Actual")})
    return out


def saved_count(date_iso: str) -> int:
    """How many PrizePicks lines are saved for the date (0 when none)."""
    saved = props.load_lines(date_iso)
    return 0 if saved is None or saved.empty else len(saved)


def line_counts_by_game(scope_preds: pd.DataFrame, date_iso: str) -> dict:
    """{gamePk: number of posted PrizePicks lines that map to a projected stat,
    summed across BOTH teams in that game}. Feeds the per-card count on the
    slate ("Dodgers 3 + Diamondbacks 1 -> 4"). A "line" is one posted (player,
    stat) prop that resolves to a stat we project, so it matches exactly what
    the game page can show. Empty dict when nothing is loaded, so cards with no
    lines render no badge."""
    lines = props.load_lines(date_iso)
    if (lines is None or lines.empty or scope_preds is None
            or scope_preds.empty or "gamePk" not in scope_preds.columns):
        return {}
    out: dict = {}
    for gpk, gp in scope_preds.groupby("gamePk"):
        _, meta = props.compare(lines, gp)
        if meta["matched"]:
            out[int(gpk)] = int(meta["matched"])
    return out


def render_board(scope_preds: pd.DataFrame, date_iso: str,
                 scope_label: str = "this game", show_ledger: bool = True,
                 warn_on_empty: bool = False) -> int:
    """Strip of the biggest gaps over the full model-vs-line ledger, for the
    given predictions frame (one game, or the whole slate). Both rank by
    ABSOLUTE gap so the leading chip is the ledger's top row. Renders nothing
    (returns 0) when no lines are stored or none of these players have a
    posted, mappable line -- except with warn_on_empty (the slate-wide call),
    where saved-but-matchless lines get an explanation instead of silence:
    a board that says nothing after "6,000 lines saved" reads as broken."""
    lines = props.load_lines(date_iso)
    if lines is None or lines.empty:
        return 0
    table, meta = props.compare(lines, scope_preds)
    if table.empty:
        if warn_on_empty:
            samples = meta.get("unmatched_names") or []
            if samples and all(s.isupper() and len(s) <= 4 for s in samples):
                hint = (" They have team codes instead of player names (a feed "
                        "paste missing its player list) -- click Clear all and "
                        "re-add them with the grabber.")
            elif samples:
                hint = (" None matched a player on this slate; unmatched "
                        f"examples: {', '.join(samples)}.")
            else:
                hint = (" They name players on this slate but none map to a "
                        "projected stat.")
            n = len(lines)
            st.warning(f"{n} PrizePicks line{'s' if n != 1 else ''} saved for "
                       f"{date_iso} could not be compared.{hint}")
        return 0
    n = meta["matched"]
    st.success(f"{n} PrizePicks line{'s' if n != 1 else ''} loaded and compared "
               f"for {scope_label}.")
    st.markdown('<div class="dv-eyebrow">Model vs the board &middot; '
                'PrizePicks lines</div>', unsafe_allow_html=True)

    # Only ACTIONABLE lines make the recommendations: a Demon/Goblin the model
    # leans Under on cannot be taken (both alt types are More-only), so it must
    # never top the edge board. Hidden lines still show on each player's
    # expandable row with their "side not offered" note.
    playable = table[table["Playable"]] if "Playable" in table.columns else table
    n_hidden = len(table) - len(playable)
    if playable.empty:
        st.info(f"All {len(table)} matched line(s) lean a side PrizePicks does "
                "not offer (Demons and Goblins are More-only), so there is "
                "nothing to act on.")
        return n

    strip = playable[playable["Edge"].abs() >= 0.005]
    chips = strip.head(6)
    if not chips.empty:
        top = float(chips["Edge"].abs().max()) or 1.0
        parts = []
        for _, r in chips.iterrows():
            d = "over" if r["Edge"] > 0 else "under"
            w = int(round(abs(float(r["Edge"])) / top * 100))
            parts.append(
                f'<div class="dv-edge-chip {d}">'
                f'<span class="ec-player">{store._esc(r["Player"])}</span>'
                f'<span class="ec-stat">{store._esc(r["Stat"])}</span>'
                f'<span class="ec-nums">{r["Model"]:g} '
                f'<span class="ec-vs">vs</span> {r["Line"]:g} &middot; '
                f'<span class="ec-lean-{d}">{r["Lean"]}</span></span>'
                f'<span class="ec-bar"><i style="width:{w}%"></i></span>'
                f"</div>")
        n_more = len(strip) - len(chips)
        if n_more > 0:
            parts.append(f'<span class="dv-edge-more">+{n_more} more</span>')
        st.markdown(f'<div class="dv-edge-strip">{"".join(parts)}</div>',
                    unsafe_allow_html=True)

    if show_ledger:
        # The per-line direction / odds ride on the expandable player rows, not
        # this compact ledger; drop them so the table stays Player..Lean wide.
        ledger = playable[[c for c in playable.columns
                           if c not in ("Direction", "OddsType", "Playable",
                                        "Actual")]]
        st.markdown(store.html_df(ledger, label_cols=3, hero=("Edge",)),
                    unsafe_allow_html=True)
    saved = props.saved_at_et(lines.attrs.get("saved_at"))
    note = f"{meta['matched']} posted line(s) for {scope_label}"
    if n_hidden:
        note += (f" · {n_hidden} hidden (model leans a side not offered; "
                 "Demons/Goblins are More-only)")
    if saved:
        note += f" · lines saved {saved}"
    note += (" · model means vs posted lines, informational, "
             "not a wager recommendation.")
    st.caption(note)
    return meta["matched"]


def _clear_lines(date_iso: str) -> None:
    """'Clear all' button callback: drop every saved line on EVERY date (a
    date-routed paste can live on several), delete any cached raw feed pull
    (resolve_and_persist re-reads it each run, so leaving it would resurrect
    the lines on the very next rerun and make Clear all a visible no-op), and
    reset the input state. Runs before the rerun, so clearing pp_paste here
    also stops resolve_and_persist from re-merging the last paste back in."""
    props.clear_lines(date_iso)
    for d, _n in props.saved_line_dates():
        props.clear_lines(d)
    try:
        for p in props.cache.CACHE_DIR.glob("prizepicks_raw_*.json"):
            p.unlink(missing_ok=True)
    except OSError:
        pass
    st.session_state["pp_paste"] = ""


# The exact request PrizePicks' own app shape expects (game_mode is required
# now; a bare request gets an application error). Must be opened WITHOUT a VPN:
# they reject VPN/datacenter IPs app-side.
FEED_URL = ("https://api.prizepicks.com/projections?"
            "league_id=2&per_page=250&single_stat=true&game_mode=pickem")


def render_input(date_iso: str) -> None:
    """The line-input controls: the paste box is the ONLY way lines come in.
    Open the feed link, copy the whole wall of code, paste once -- date
    routing puts every line on its game's slate. Persistence/comparison is
    handled by resolve_and_persist + render_board; this only draws widgets."""
    # ── A bordered container, NOT an expander -- render_input renders inside
    #    the header's "Update lines" popover, and Streamlit forbids nesting
    #    expanders inside popovers. ──
    with st.container(border=True):
        st.markdown("**Copy the feed page and paste it here**")
        st.markdown(
            f'1. Open the <a href="{FEED_URL}" target="_blank" rel="noopener">'
            "<b>PrizePicks feed &#8599;</b></a> in a new tab (VPN off -- "
            "PrizePicks blocks VPN traffic). It looks like a huge wall of "
            "code -- that IS the whole board, and exactly what you want.<br>"
            "2. Select all (&#8984;A) and copy (&#8984;C).<br>"
            "3. Come back here, paste in the box below, and click **Add these "
            "lines**. Every line lands on the date of its own game, so "
            "tonight's copy of tomorrow's board goes to tomorrow's slate.",
            unsafe_allow_html=True)
    st.caption("A stat tab copied from the PrizePicks board page, or a simple "
               "`Name, Stat, Line` list, pastes fine too.")
    st.text_area("Paste the feed JSON, board text, or a Name, Stat, Line list",
                 height=150, key="pp_paste",
                 placeholder=(
                     "Paste the copied feed page here, or type a simple list:\n"
                     "Ketel Marte, Total Bases, 1.5\n"
                     "Zac Gallen, Pitcher Strikeouts, 6.5"))
    # The paste is already merged into the saved set(s) by resolve_and_persist
    # at the top of the page. Per-date state comes from DISK, not session
    # state: a date-routed paste can live on several dates, and these pointers
    # (and Clear all's reach) must survive later pastes and reruns.
    n_saved = saved_count(date_iso)
    all_saved = props.saved_line_dates()
    # Pointers only for live dates (today on): a stale save from last week is
    # noise, though Clear all still reaches every date.
    elsewhere = {d: n for d, n in all_saved
                 if d != date_iso and d >= today_iso()}
    c_add, c_clear = st.columns([2.4, 1], gap="small")
    with c_add:
        compared = st.button("Add these lines", type="primary", key="pp_compare",
                             use_container_width=True)
    with c_clear:
        st.button("Clear all", key="pp_clear", on_click=_clear_lines,
                  args=(date_iso,), use_container_width=True,
                  disabled=n_saved == 0 and not all_saved,
                  help="Remove every saved line, on this and every other date")
    st.caption("A paste ADDS to your saved lines (re-pasting refreshes, no "
               "duplicates), so stat tabs can also come in one at a time. "
               "Fantasy Score, 1st-Inning and Combo props are not projected "
               "and are skipped. Lines apply to every game on their date.")
    # The no-player-names diagnostic must never be drowned out by older saved
    # lines reading as success -- that silence is the original bug.
    note = st.session_state.get("pp_parse_note")
    if note:
        st.warning(note)
    n_total = n_saved + sum(elsewhere.values())
    if compared:
        st.toast(("The last paste saved nothing. " if note else "")
                 + (f"{n_total} PrizePicks line(s) on file." if n_total
                    else "Couldn't read any lines from that paste."))
    if n_saved:
        st.success(f"{n_saved} PrizePicks line(s) loaded for this slate. They "
                   "show above and on every game page.")
    for d, n in sorted(elsewhere.items())[:3]:
        st.info(f"{n} line(s) are saved for games on **{_fmt_date(d)}**. Step "
                "the date there to see their board.")
    if not n_total and not note and (st.session_state.get("pp_paste") or "").strip():
        st.warning("Couldn't read any lines from that paste. Paste the grabber "
                   "output, the PrizePicks board or feed JSON, or a simple "
                   "`Name, Stat, Line` list.")

