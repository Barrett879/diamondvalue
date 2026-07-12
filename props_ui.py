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
    """Current lines from any source, in priority order -- a live pull, the
    pasted text (JSON or list), or the previously-saved file (sticky across
    reloads) -- persisted so every page/game reads the same set. Returns the
    lines DataFrame or None. Safe to call at the top of a page before the input
    widgets render (it reads their committed session_state values)."""
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
            if txt[:1] in "{[":
                try:
                    got = props.parse_prizepicks_json(txt)
                    lines = got if (got is not None and not got.empty) else None
                except Exception:  # noqa: BLE001
                    lines = None
            if lines is None or lines.empty:
                got = props.parse_line_list(txt)
                lines = got if (got is not None and not got.empty) else None
    if lines is not None and not lines.empty:
        props.save_lines(date_iso, lines)   # fresh input -> persist slate-wide
        return lines
    return props.load_lines(date_iso)       # fall back to a prior save


def render_board(scope_preds: pd.DataFrame, date_iso: str,
                 scope_label: str = "this game") -> int:
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

    st.markdown(store.html_df(table, label_cols=3, hero=("Edge",)),
                unsafe_allow_html=True)
    saved = props.saved_at_et(lines.attrs.get("saved_at"))
    note = f"{meta['matched']} posted line(s) for {scope_label}"
    if saved:
        note += f" · lines saved {saved}"
    note += (" · model means vs posted lines, informational, "
             "not a wager recommendation.")
    st.caption(note)
    return meta["matched"]


def render_input(date_iso: str) -> None:
    """The line-input controls: a best-effort live pull, the bookmarklet
    installer, and the paste box. Persistence/comparison is handled by
    resolve_and_persist + render_board; this only draws the widgets."""
    if st.button("Update now", type="primary",
                 help="Try a live PrizePicks pull for this date", key="pp_update"):
        with st.spinner("Pulling PrizePicks lines..."):
            payload = fp.fetch(date_iso)
        if payload:
            st.session_state["pp_payload"] = payload
            st.rerun()
        else:
            st.warning("PrizePicks blocked the automated request (this is "
                       "normal). Paste the lines below instead.")

    with st.expander("One-click grab (bookmarklet)", expanded=False):
        st.markdown(
            "Install this once, then it grabs every MLB line for you. "
            "**Drag the button below to your bookmarks bar** (or make a new "
            "bookmark and paste the code as its URL). Then open "
            "**prizepicks.com** in a tab and click the bookmark: it copies all "
            "the lines to your clipboard. Come back here and paste below.")
        html = (
            '<a href="' + BOOKMARKLET.replace('"', "&quot;") + '" '
            'style="display:inline-block;padding:8px 16px;border-radius:20px;'
            'background:#0fae9d;color:#fff;font:600 14px Manrope,sans-serif;'
            'text-decoration:none;cursor:grab" '
            'onclick="event.preventDefault()">Grab PrizePicks lines</a>'
            '<div style="font:13px Manrope,sans-serif;color:#71757f;'
            'margin-top:8px">Drag me up to your bookmarks bar.</div>')
        components.html(html, height=70)
        st.caption("Or copy the code and paste it as a new bookmark's URL:")
        st.code(BOOKMARKLET, language="javascript")

    st.markdown(
        "Open PrizePicks in your own browser, then either copy the JSON from "
        "`api.prizepicks.com/projections?league_id=2&per_page=250` and paste "
        "it below, **or** type a simple `Name, Stat, Line` list (one per row).")
    st.text_area("PrizePicks JSON or a Name, Stat, Line list", height=180,
                 key="pp_paste", placeholder=(
                     "Ketel Marte, Total Bases, 1.5\n"
                     "Corbin Carroll, Hits, 0.5\n"
                     "Zac Gallen, Pitcher Strikeouts, 6.5"))
    st.button("Compare pasted lines", type="primary", key="pp_compare")
    st.caption("Tip: after pasting, click **Compare** or click anywhere outside "
               "the box (pressing Enter alone just adds a line). Lines apply to "
               "every game on the slate.")
