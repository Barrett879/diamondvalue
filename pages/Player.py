"""Player page: one player's projection for the selected date, the headline
model inputs behind it, and his recent predicted-vs-actual history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, store  # noqa: E402
from mlblib.theme import SENTINEL, render_footer, render_nav, render_page_chrome  # noqa: E402
from mlblib.util import parse_iso_date, today_iso  # noqa: E402

st.set_page_config(page_title="Player · DiamondValue", page_icon="static/favicon.svg",
                   layout="wide")
render_page_chrome()
render_nav("Player")

st.markdown('<div class="dv-brand">Player</div>', unsafe_allow_html=True)


def _mirror_date():
    d = st.session_state.get("player_date")
    if d:
        st.query_params["date"] = d.isoformat()


if "player_date" not in st.session_state:
    st.session_state["player_date"] = (
        parse_iso_date(st.query_params.get("date")) or parse_iso_date(today_iso()))

st.date_input("Game date", key="player_date", on_change=_mirror_date)
_mirror_date()
date_iso = st.session_state["player_date"].isoformat()

preds = store.load_predictions(date_iso)
if preds is None or preds.empty or not preds.get("PA", pd.Series(dtype=float)).notna().any():
    st.info(f"No predictions have been generated for {date_iso} yet. Pick a date "
            "with a built slate, or generate predictions first.")
    render_footer()
    st.stop()

# Stable, sorted option list (spec: never reorder a keyed selectbox's options).
names = sorted(preds["fullName"].dropna().unique().tolist())
seed = st.query_params.get("player")
if "player_pick" not in st.session_state:
    st.session_state["player_pick"] = seed if seed in names else None


def _mirror_player():
    p = st.session_state.get("player_pick")
    if p:
        st.query_params["player"] = p


pick = st.selectbox("Player", options=names, index=None, key="player_pick",
                    on_change=_mirror_player, placeholder="Search a player on this slate")

if not pick:
    st.caption("Select a player to see his projection and the inputs behind it.")
    render_footer()
    st.stop()

rows = preds[preds["fullName"] == pick]
row = rows.iloc[0]
role = row["role"]

st.subheader(pick)
if role == "bat":
    st.caption(f"Projected line for {date_iso}"
               + ("  ·  bench (per game if he starts)" if row.get("is_bench") else ""))
    st.markdown(store.html_batter_table(rows), unsafe_allow_html=True)
    st.markdown("**What the model saw**")
    inp = {
        "Expected PA (Marcel prior)": row.get("marcel_PApg"),
        "Lineup slot": row.get("slot"),
        "Opposing starter hand": {0.0: "L", 1.0: "R", 0.5: "S"}.get(row.get("opp_sp_hand")),
    }
    st.write({k: (round(float(v), 2) if isinstance(v, (int, float)) and v == v else (v or SENTINEL))
              for k, v in inp.items()})
else:
    st.caption(f"Projected start for {date_iso}")
    st.markdown(store.html_pitcher_table(rows), unsafe_allow_html=True)

# Recent predicted-vs-actual, from the accuracy tracker (if built).
acc_path = cache.dc_path("accuracy_history_v1.parquet")
acc = cache.read_parquet_or_none(acc_path)
if acc is not None and "personId" in acc.columns:
    hist = acc[acc["personId"] == row["personId"]]
    if not hist.empty:
        st.markdown("**Recent predicted vs actual**")
        cols = [c for c in ("date", "stat", "pred", "actual", "abs_err_model")
                if c in hist.columns]
        st.markdown(store.html_df(hist.sort_values("date").tail(20)[cols],
                                  label_cols=2), unsafe_allow_html=True)

render_footer()
