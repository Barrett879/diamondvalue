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
import streamlit.components.v1 as components

import scripts.fetch_prizepicks as fp
from mlblib import props, store
from mlblib.util import today_iso


def _fmt_date(date_iso: str) -> str:
    """'2026-07-17' -> 'Thu Jul 17' (falls back to the raw string)."""
    try:
        return _dt.date.fromisoformat(date_iso).strftime("%a %b %-d")
    except ValueError:
        return date_iso

# One-click grabber: runs in the user's OWN logged-in browser (where PrizePicks
# is not Cloudflare-blocked). It resolves MLB's league id from /leagues (the
# numeric id is NOT stable -- a hardcoded league_id returned 0 lines), auto-pages
# through every projection (Name | Stat | Line | odds_type | game date, so the
# Demon/Goblin distinction survives and tonight's grab of TOMORROW's board lands
# on tomorrow's slate), then pops up a small box with the lines pre-selected.
# The user presses Cmd+C (or clicks Copy) and pastes. IMPORTANT: it does NOT
# auto-copy after the fetch -- awaiting the network consumes the click's user
# activation, so navigator.clipboard.writeText then throws NotAllowedError in
# Safari (and increasingly Chrome); the pre-selected textarea + Copy button copy
# under a FRESH gesture, which every browser allows. If it still finds nothing it
# alerts a short diagnostic (resolved league, data/included counts, item types).
BOOKMARKLET = (
    "javascript:(async()=>{try{let H={headers:{Accept:'application/json'},"
    "credentials:'include'};let lid='2';try{let Lr=await fetch('https://"
    "api.prizepicks.com/leagues?per_page=250',H);if(Lr.ok){let Lj=await "
    "Lr.json();let mm=(Lj.data||[]).find(l=>{let a=l.attributes||{};return("
    "(a.name||a.display_name||'')+'').toUpperCase()==='MLB'});if(mm)lid="
    "mm.id}}catch(e){}let o=[],P=1,dbg='';for(let p=1;p<=P&&p<=8;p++){let r="
    "await fetch('https://api.prizepicks.com/projections?league_id='+lid+"
    "'&per_page=250&page='+p,H);if(!r.ok){alert('PrizePicks returned '+"
    "r.status+'. Open prizepicks.com (signed in), then click again.');return}"
    "let j=await r.json();P=(j.meta&&j.meta.total_pages)||1;let n={};"
    "(j.included||[]).forEach(i=>{if(i.type=='new_player'||i.type=='player'){"
    "let a=i.attributes||{};n[i.id]=a.name||a.display_name}});if(p==1)dbg="
    "'lg '+lid+' data '+((j.data||[]).length)+' incl '+((j.included||[])."
    "length)+' dt '+((j.data&&j.data[0]&&j.data[0].type)||'?')+' it '+"
    "((j.included&&j.included[0]&&j.included[0].type)||'?');(j.data||[])."
    "forEach(d=>{let a=d.attributes||{},rel=d.relationships||{},xd=((rel."
    "new_player||rel.player||{}).data)||{},m=n[xd.id];if(m&&a.line_score!="
    "null&&a.stat_type)o.push(m+' | '+a.stat_type+' | '+a.line_score+' | '+"
    "(a.odds_type||'standard')+' | '+(a.start_time||''))})}"
    "if(!o.length){alert('Grabbed 0 lines. "
    "Debug: '+dbg+'. Copy this text and send it to DiamondValue.');return}"
    "let s=o.join(String.fromCharCode(10));let ov=document."
    "createElement('div');ov.style.cssText='position:fixed;inset:0;z-index:"
    "2147483647;background:rgba(0,0,0,.72);display:flex;align-items:center;"
    "justify-content:center;font-family:-apple-system,sans-serif';let bx="
    "document.createElement('div');bx.style.cssText='background:#fff;color:"
    "#111;padding:18px;border-radius:12px;width:min(560px,92vw);box-shadow:"
    "0 12px 44px rgba(0,0,0,.45)';let ms=document.createElement('div');"
    "ms.style.cssText='font:600 15px sans-serif;margin-bottom:10px';"
    "ms.textContent='Got '+o.length+' PrizePicks lines. Press Cmd+C (already "
    "selected) or click Copy, then paste into DiamondValue.';let ta=document."
    "createElement('textarea');ta.readOnly=true;ta.value=s;ta.style.cssText="
    "'width:100%;height:150px;font:12px monospace;padding:8px;border:1px "
    "solid #ccc;border-radius:8px;box-sizing:border-box';let cp=document."
    "createElement('button');cp.textContent='Copy';cp.style.cssText="
    "'margin-top:10px;padding:8px 18px;border:0;border-radius:20px;"
    "background:#0fae9d;color:#fff;font:600 14px sans-serif;cursor:pointer';"
    "let cl=document.createElement('button');cl.textContent='Close';"
    "cl.style.cssText='margin:10px 0 0 8px;padding:8px 18px;border:1px solid "
    "#ccc;border-radius:20px;background:#fff;color:#333;font:600 14px "
    "sans-serif;cursor:pointer';bx.appendChild(ms);bx.appendChild(ta);"
    "bx.appendChild(cp);bx.appendChild(cl);ov.appendChild(bx);document.body."
    "appendChild(ov);ta.focus();ta.select();cp.onclick=function(){ta.focus();"
    "ta.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}"
    "if(!ok){try{navigator.clipboard.writeText(s)}catch(e){}}cp.textContent="
    "'Copied!'};cl.onclick=function(){ov.remove()};}catch(e){alert('Could "
    "not reach PrizePicks ('+e+'). Open prizepicks.com in THIS tab (signed "
    "in), then click the bookmark.')}})();"
)


def resolve_and_persist(date_iso: str):
    """Current lines from any source -- a live pull or the pasted text (JSON,
    board, or list) -- MERGED into the previously-saved set and persisted, so
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
            if got is not None and got.empty and got.attrs.get("skipped_noname"):
                note = (f"That JSON has {got.attrs['skipped_noname']} props but "
                        "no player names (its player list is missing), so "
                        "nothing could be saved. Use the grabber above -- it "
                        "resolves the names -- or re-open the feed link and "
                        "copy the whole page.")
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
    st.session_state["pp_payload"] = None


# No single_stat filter: combo props (H+R+RBI) are projected too, and the
# ladder of Demon/Goblin alt lines is collapsed to one line per prop at merge.
FEED_URL = ("https://api.prizepicks.com/projections?"
            "league_id=2&per_page=250")


def render_input(date_iso: str) -> None:
    """The line-input controls. The PRIMARY path is pasting the feed page:
    open one link, copy the whole wall of code, paste once -- nothing to
    install, and date routing puts every line on its game's slate. The
    one-click grabber bookmark and a best-effort live pull sit below it.
    Persistence/comparison is handled by resolve_and_persist + render_board;
    this only draws widgets."""
    # ── Fastest: open the feed link, copy the wall of text, paste. A bordered
    #    container, NOT an expander -- render_input is already inside the
    #    "Add / update" expander and Streamlit forbids nesting expanders. ──
    with st.container(border=True):
        st.markdown("**Fastest: copy the feed page and paste it here** "
                    "(recommended)")
        st.markdown(
            f'1. Open the <a href="{FEED_URL}" target="_blank" rel="noopener">'
            "<b>PrizePicks feed &#8599;</b></a> in a new tab. It looks like a "
            "huge wall of code -- that IS the whole board, and exactly what "
            "you want.<br>"
            "2. Select all (&#8984;A) and copy (&#8984;C).<br>"
            "3. Come back here, paste in the box below, and click **Add these "
            "lines**. Every line lands on the date of its own game, so "
            "tonight's copy of tomorrow's board goes to tomorrow's slate.",
            unsafe_allow_html=True)
    st.caption("A stat tab copied from the PrizePicks board page, or a simple "
               "`Name, Stat, Line` list, pastes fine too.")
    st.text_area("Paste the grabber output, feed JSON, board text, or a "
                 "Name, Stat, Line list", height=150, key="pp_paste",
                 placeholder=(
                     "Paste the grabber/feed output here, or type a simple list:\n"
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

    # ── Alternative: the one-click grabber bookmark. Runs in the user's own
    #    logged-in browser and copies the whole board without opening the feed
    #    page. A bordered container, NOT an expander (no nesting). ──
    with st.container(border=True):
        st.markdown("**Alternative: a one-click grabber bookmark**")
        st.markdown(
            "A bookmark that copies every line in one click, no feed page "
            "involved.<br><br>"
            "**Install once — Chrome (easiest):**<br>"
            "1. Show the bookmarks bar (&#8984;&#8679;B).<br>"
            "2. Drag the teal button below onto the **bookmarks bar** (not the "
            "address bar).<br>"
            "**Safari:** dragging turns into a search, so instead **click the "
            "button to copy it**, press &#8984;D to bookmark any page, open "
            "**Bookmarks &rsaquo; Edit Bookmarks**, and paste it into that "
            "bookmark's **Address** field.<br><br>"
            "**Each day:** open **prizepicks.com** (signed in), click the "
            "bookmark, press &#8984;C when the box pops up (already selected), "
            "then paste into the box above.",
            unsafe_allow_html=True)
        # The <a href> holds the bookmarklet so Chrome can DRAG it to the
        # bookmarks bar; clicking instead COPIES it (Safari can't drag a
        # javascript: link, so click-to-copy is its path). The component iframe
        # carries clipboard-write, so navigator.clipboard works on the real
        # click; execCommand is the fallback, and the st.code block below is the
        # last resort if a browser blocks both.
        grab = BOOKMARKLET.replace('"', "&quot;")
        html = (
            '<a id="pp-grab" href="' + grab + '" '
            'style="display:inline-block;padding:9px 18px;border-radius:20px;'
            'background:#0fae9d;color:#fff;font:600 14px Manrope,sans-serif;'
            'text-decoration:none;cursor:pointer">Grab PrizePicks lines</a>'
            '<div id="pp-hint" style="font:13px Manrope,sans-serif;color:#71757f;'
            'margin-top:8px;line-height:1.45">Chrome: drag me to the bookmarks '
            "bar. Safari: click me to copy, then paste into a new "
            "bookmark&rsquo;s Address.</div>"
            "<script>(function(){"
            'var a=document.getElementById("pp-grab"),'
            'h=document.getElementById("pp-hint");if(!a)return;'
            'function done(ok){h.style.color=ok?"#0b8f80":"#c0392b";'
            'h.textContent=ok?"Copied. Now press Cmd+D to bookmark any page, open '
            'Bookmarks > Edit Bookmarks, and paste this as its Address.":'
            '"Could not copy here - use the code box below instead.";}'
            'function fallback(code){try{var t=document.createElement("textarea");'
            't.value=code;t.style.position="fixed";t.style.top="-1000px";'
            "document.body.appendChild(t);t.focus();t.select();"
            'var ok=document.execCommand("copy");document.body.removeChild(t);'
            "done(ok);}catch(err){done(false);}}"
            'a.addEventListener("click",function(e){e.preventDefault();'
            'var code=a.getAttribute("href");'
            "if(navigator.clipboard&&navigator.clipboard.writeText){"
            "navigator.clipboard.writeText(code).then(function(){done(true);},"
            "function(){fallback(code);});}else{fallback(code);}});})();"
            "</script>")
        components.html(html, height=115)
        st.caption("Or copy this code manually and paste it as the bookmark's "
                   "Address:")
        st.code(BOOKMARKLET, language="javascript")

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
                       "grabber or copy-and-paste above instead.")
