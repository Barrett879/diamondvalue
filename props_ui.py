"""Shared PrizePicks props UI (line input + model-vs-market board).

Consolidated from the old standalone Props page so the whole feature lives
wherever the user already is -- the game detail page and the home slate -- with
no separate tab. Line input persists slate-wide (per date), so pasting/pulling
on any page flows to every game.

An informational model-vs-market view, never a wager recommendation.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import scripts.fetch_prizepicks as fp
from mlblib import props, store

# One-click grabber: runs in the user's OWN logged-in browser (where PrizePicks
# is not Cloudflare-blocked), fetches every MLB projection, and copies a clean
# "Name | Stat | Line" list to the clipboard ready to paste.
BOOKMARKLET = (
    "javascript:(async()=>{try{let o=[],P=1;for(let p=1;p<=P&&p<=8;p++){"
    "let r=await fetch('https://api.prizepicks.com/projections?league_id=2"
    "&per_page=250&page='+p+'&single_stat=true',{headers:{Accept:"
    "'application/json'},credentials:'include'});if(!r.ok){alert('PrizePicks "
    "returned '+r.status+'. Open prizepicks.com first, then click again.');"
    "return}let j=await r.json();P=(j.meta&&j.meta.total_pages)||1;let n={};"
    "(j.included||[]).forEach(i=>{if(i.type=='new_player')n[i.id]="
    "i.attributes.name});(j.data||[]).forEach(d=>{let a=d.attributes||{},"
    "x=((d.relationships||{}).new_player||{}).data,m=x?n[x.id]:0;"
    "if(m&&a.line_score!=null&&a.stat_type)o.push(m+' | '+a.stat_type+' | '"
    "+a.line_score)})}await navigator.clipboard.writeText(o.join(String."
    "fromCharCode(10)));alert('Copied '+o.length+' PrizePicks MLB lines. "
    "Paste them into DiamondValue.')}catch(e){alert('Could not fetch "
    "PrizePicks ('+e+'). Open prizepicks.com in this tab first, then click "
    "the bookmarklet.')}})();"
)


def resolve_and_persist(date_iso: str):
    """Current lines from any source -- a live pull or the pasted text (JSON,
    board, or list) -- MERGED into the previously-saved set and persisted, so
    every page/game reads the same accumulating board. PrizePicks has no All
    tab, so each paste ADDS to what is saved (a re-paste of the same tab just
    refreshes those lines). Returns the merged frame or None. Safe to call at
    the top of a page before the input widgets render (it reads their committed
    session_state values). 'Clear all' (clear_lines) resets the set."""
    lines = None
    payload = st.session_state.get("pp_payload") or fp.load_raw(date_iso)
    if payload:
        try:
            lines = props.parse_prizepicks_json(payload)
        except Exception:  # noqa: BLE001
            lines = None
    if lines is None or lines.empty:
        txt = (st.session_state.get("pp_paste") or "").strip()
        if txt:
            got = props.parse_any(txt)   # JSON, board text, or a simple list
            lines = got if (got is not None and not got.empty) else None
    if lines is not None and not lines.empty:
        merged = props.merge_lines(props.load_lines(date_iso), lines)
        props.save_lines(date_iso, merged)   # accumulate across stat tabs
        return merged
    return props.load_lines(date_iso)       # fall back to a prior save


def props_by_name(scope_preds: pd.DataFrame, date_iso: str) -> dict:
    """{fullName: [ {Stat, Model, Line, Edge, Lean}, ... ]} for the players in
    scope_preds that have a posted, mappable line. Feeds the expandable roster
    rows on the game page."""
    lines = props.load_lines(date_iso)
    if lines is None or lines.empty:
        return {}
    table, _ = props.compare(lines, scope_preds)
    out: dict = {}
    for _, r in table.iterrows():
        out.setdefault(r["Player"], []).append(
            {"Stat": r["Stat"], "Model": r["Model"], "Line": r["Line"],
             "Edge": r["Edge"], "Lean": r["Lean"],
             "Direction": r.get("Direction", ""), "OddsType": r.get("OddsType", "")})
    return out


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
                 scope_label: str = "this game", show_ledger: bool = True) -> int:
    """Strip of the biggest gaps over the full model-vs-line ledger, for the
    given predictions frame (one game, or the whole slate). Both rank by
    ABSOLUTE gap so the leading chip is the ledger's top row. Renders nothing
    (returns 0) when no lines are stored or none of these players have a
    posted, mappable line."""
    lines = props.load_lines(date_iso)
    if lines is None or lines.empty:
        return 0
    table, meta = props.compare(lines, scope_preds)
    if table.empty:
        return 0
    n = meta["matched"]
    st.success(f"{n} PrizePicks line{'s' if n != 1 else ''} loaded and compared "
               f"for {scope_label}.")
    st.markdown('<div class="dv-eyebrow">Model vs the board &middot; '
                'PrizePicks lines</div>', unsafe_allow_html=True)

    strip = table[table["Edge"].abs() >= 0.005]
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
        ledger = table[[c for c in table.columns
                        if c not in ("Direction", "OddsType")]]
        st.markdown(store.html_df(ledger, label_cols=3, hero=("Edge",)),
                    unsafe_allow_html=True)
    saved = props.saved_at_et(lines.attrs.get("saved_at"))
    note = f"{meta['matched']} posted line(s) for {scope_label}"
    if saved:
        note += f" · lines saved {saved}"
    note += (" · model means vs posted lines, informational, "
             "not a wager recommendation.")
    st.caption(note)
    return meta["matched"]


def _clear_lines(date_iso: str) -> None:
    """'Clear all' button callback: drop every accumulated line for the date and
    reset the input state. Runs before the rerun, so clearing pp_paste here also
    stops resolve_and_persist from re-merging the last paste back in."""
    props.clear_lines(date_iso)
    st.session_state["pp_paste"] = ""
    st.session_state["pp_payload"] = None


FEED_URL = ("https://api.prizepicks.com/projections?"
            "league_id=2&per_page=250&single_stat=true")


def render_input(date_iso: str) -> None:
    """The line-input controls. The reliable, all-browser path is: open the
    PrizePicks feed in a new tab, copy the JSON, paste it. A best-effort live
    pull and an optional bookmarklet are also offered. Persistence/comparison
    is handled by resolve_and_persist + render_board; this only draws widgets."""
    # Primary path -- works in Safari, Chrome, anywhere, no install.
    st.markdown(
        "**Get today's lines** (any browser, including Safari):<br>"
        f'1. Open the <a href="{FEED_URL}" target="_blank" rel="noopener">'
        "PrizePicks feed &#8599;</a> in a new tab (sign in to PrizePicks first "
        "if it asks).<br>"
        "2. Select all (&#8984;A) and copy (&#8984;C).<br>"
        "3. Paste it in the box below and click **Add these lines** (repeat per "
        "PrizePicks stat tab, they accumulate).",
        unsafe_allow_html=True)
    st.text_area("PrizePicks JSON or a Name, Stat, Line list", height=170,
                 key="pp_paste", placeholder=(
                     "Paste the copied feed here, or type a simple list:\n"
                     "Ketel Marte, Total Bases, 1.5\n"
                     "Zac Gallen, Pitcher Strikeouts, 6.5"))
    # The paste is already merged into the saved set by resolve_and_persist at
    # the top of the page, so n_saved here is the running total across tabs.
    saved = props.load_lines(date_iso)
    n_saved = 0 if saved is None or saved.empty else len(saved)
    c_add, c_clear = st.columns([2.4, 1], gap="small")
    with c_add:
        compared = st.button("Add these lines", type="primary", key="pp_compare",
                             use_container_width=True)
    with c_clear:
        st.button("Clear all", key="pp_clear", on_click=_clear_lines,
                  args=(date_iso,), use_container_width=True,
                  disabled=n_saved == 0,
                  help="Remove every line you have added for this date")
    st.caption("PrizePicks has no All tab, so each paste ADDS to your lines "
               "(re-pasting a tab just refreshes it, no duplicates). Paste one "
               "stat tab, click Add, switch tabs, repeat. You can skip the "
               "Fantasy Score, 1st-Inning and Combo tabs, those are not "
               "projected. Lines apply to every game on the slate.")
    if compared:
        st.toast(f"{n_saved} PrizePicks line(s) loaded." if n_saved
                 else "Couldn't read any lines from that paste.")
    if n_saved:
        st.success(f"{n_saved} PrizePicks line(s) loaded across the tabs you have "
                   "added. They show above and on every game page.")
    elif (st.session_state.get("pp_paste") or "").strip():
        st.warning("Couldn't read any lines from that paste. Paste the copied "
                   "PrizePicks board or the feed JSON, or a simple "
                   "`Name, Stat, Line` list.")

    # Best-effort automated pull (usually blocked by PrizePicks; harmless).
    if st.button("Try a live pull instead", key="pp_update",
                 help="Attempts to fetch directly; PrizePicks usually blocks this"):
        with st.spinner("Pulling PrizePicks lines..."):
            payload = fp.fetch(date_iso)
        if payload:
            st.session_state["pp_payload"] = payload
            st.rerun()
        else:
            st.warning("PrizePicks blocked the automated pull (normal). Use the "
                       "copy-and-paste steps above instead.")

    # Optional bookmarklet, with Safari-correct install steps.
    with st.expander("One-click bookmarklet (optional, advanced)", expanded=False):
        st.markdown(
            "Copies every line in one click, but the install is fiddly. "
            "**Install:**<br>"
            "1. Show your bookmarks bar first. In Safari: **View &rsaquo; Show "
            "Favorites Bar** (&#8984;&#8679;B). In Chrome: **&#8984;&#8679;B**.<br>"
            "2. Drag the button below onto that **bookmarks bar**, NOT the "
            "address/search bar (Safari turns a script dropped there into a "
            "search, which is what you saw).<br>"
            "3. If dragging will not stick in Safari: copy the code below, "
            "bookmark any page (&#8984;D), then **Bookmarks &rsaquo; Edit "
            "Bookmarks**, and paste the code into that bookmark's Address field "
            "(Safari accepts it there even though the address bar rejects it).<br>"
            "Then open **prizepicks.com** and click the bookmark; it copies the "
            "lines to your clipboard to paste above.", unsafe_allow_html=True)
        html = (
            '<a href="' + BOOKMARKLET.replace('"', "&quot;") + '" '
            'style="display:inline-block;padding:8px 16px;border-radius:20px;'
            'background:#0fae9d;color:#fff;font:600 14px Manrope,sans-serif;'
            'text-decoration:none;cursor:grab" '
            'onclick="event.preventDefault()">Grab PrizePicks lines</a>'
            '<div style="font:13px Manrope,sans-serif;color:#71757f;'
            'margin-top:8px">Drag me to your bookmarks bar (not the search bar).</div>')
        components.html(html, height=70)
        st.caption("Or copy this code and paste it as the new bookmark's Address:")
        st.code(BOOKMARKLET, language="javascript")
