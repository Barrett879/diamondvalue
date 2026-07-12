"""Props page: compare our per-game projections to posted PrizePicks lines.

An informational model-vs-market view, not betting advice. Lines come from a
manual paste (reliable: PrizePicks blocks automated requests) or a best-effort
"Update now" live pull. The comparison itself is pure and offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.fetch_prizepicks as fp  # noqa: E402
from mlblib import props, store  # noqa: E402
from mlblib.theme import render_footer, render_nav, render_page_chrome  # noqa: E402
from mlblib.util import today_iso  # noqa: E402

# One-click grabber: runs in the user's OWN logged-in browser (where PrizePicks
# is not Cloudflare-blocked), fetches every MLB projection, and copies a clean
# "Name | Stat | Line" list to the clipboard ready to paste below.
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
    "Paste them into DiamondValue Props.')}catch(e){alert('Could not fetch "
    "PrizePicks ('+e+'). Open prizepicks.com in this tab first, then click "
    "the bookmarklet.')}})();"
)

st.set_page_config(page_title="Props · DiamondValue", page_icon="static/favicon.svg",
                   layout="wide")
render_page_chrome()
render_nav("Props")

st.markdown('<div class="dv-brand">Model <span class="accent">vs</span> the board</div>'
            '<div class="dv-tagline">PrizePicks lines vs our projections</div>',
            unsafe_allow_html=True)

date_iso = st.query_params.get("date") or today_iso()
preds = store.load_predictions(date_iso)
if preds is None or preds.empty:
    st.info(f"No projections generated for {date_iso} yet.")
    render_footer()
    st.stop()

st.caption(
    "Each row shows our expected value next to a posted line and the gap "
    "between them, biggest disagreements first. These are model means, not "
    "predictions of outcomes, and nothing here is a wager recommendation.")


def _show(lines):
    props.save_lines(date_iso, lines)   # persist so the Game pages can read it
    table, meta = props.compare(lines, preds)
    if table.empty:
        st.warning("No lines matched a projected player on today's slate. "
                   "Check the date and that names line up.")
        return
    st.markdown(store.html_df(table, label_cols=3, hero=("Edge",)),
                unsafe_allow_html=True)
    bits = [f"{meta['matched']} matched"]
    if meta["unmatched"]:
        bits.append(f"{meta['unmatched']} player(s) not on this slate")
    if meta["unmapped"]:
        bits.append(f"{meta['unmapped']} stat type(s) we don't project")
    bits.append("saved for the game pages")
    st.caption(" · ".join(bits))


# ── Live "Update now" (best effort) ─────────────────────────────────────────
c1, c2 = st.columns([1, 3])
with c1:
    update = st.button("Update now", type="primary",
                       help="Try a live PrizePicks pull for this date")
if update:
    with st.spinner("Pulling PrizePicks lines..."):
        payload = fp.fetch(date_iso)
    if payload:
        st.session_state["pp_payload"] = payload
        st.success("Pulled live lines.")
    else:
        st.warning("PrizePicks blocked the automated request (this is normal). "
                   "Paste the lines below instead.")

# Reuse a previously pulled or pasted payload across reruns.
payload = st.session_state.get("pp_payload") or fp.load_raw(date_iso)

if payload:
    _show(props.parse_prizepicks_json(payload))

# ── One-click grabber (bookmarklet) ─────────────────────────────────────────
with st.expander("One-click grab (bookmarklet)", expanded=False):
    st.markdown(
        "Install this once, then it grabs every MLB line for you. "
        "**Drag the button below to your bookmarks bar** (or make a new "
        "bookmark and paste the code as its URL). Then open **prizepicks.com** "
        "in a tab and click the bookmark: it copies all the lines to your "
        "clipboard. Come back here, open **Paste lines**, and paste.")
    # A real draggable javascript: link has to live in an iframe -- Streamlit's
    # markdown sanitizer strips javascript: hrefs.
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

# ── Paste fallback (the reliable path) ──────────────────────────────────────
with st.expander("Paste lines (recommended)", expanded=payload is None):
    st.markdown(
        "Open PrizePicks in your own browser, then either copy the JSON from "
        "`api.prizepicks.com/projections?league_id=2&per_page=250` and paste "
        "it below, **or** type a simple `Name, Stat, Line` list (one per row).")
    txt = st.text_area("PrizePicks JSON or a Name, Stat, Line list", height=200,
                       key="pp_paste", placeholder=(
                           "Ketel Marte, Total Bases, 1.5\n"
                           "Corbin Carroll, Hits, 0.5\n"
                           "Zac Gallen, Pitcher Strikeouts, 6.5"))
    if st.button("Compare pasted lines"):
        stripped = (txt or "").strip()
        lines = None
        if stripped.startswith("{"):
            try:
                lines = props.parse_prizepicks_json(stripped)
            except Exception:  # noqa: BLE001
                st.error("That did not parse as PrizePicks JSON.")
        if lines is None:
            lines = props.parse_line_list(stripped)
        if lines is None or lines.empty:
            st.warning("Nothing to compare yet.")
        else:
            _show(lines)

render_footer()
