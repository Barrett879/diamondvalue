"""Accuracy page: how the model has done against the baselines over time.

Reads only cache/accuracy_history_v1.parquet, which the accuracy tracker builds
by scoring past predictions against actual box scores. The projection stays
pure; scoring happens against it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, store  # noqa: E402
from mlblib.theme import render_footer, render_nav, render_page_chrome  # noqa: E402

st.set_page_config(page_title="Accuracy · DiamondValue", page_icon="static/favicon.svg",
                   layout="wide")
render_page_chrome()
render_nav("Accuracy")

st.markdown('<div class="dv-brand">Accuracy</div>', unsafe_allow_html=True)
st.caption("Model predictions scored against actual results, next to the "
           "season-average baseline. Lower mean absolute error is better. "
           "Every number is an expected value, so the honest test is whether "
           "the model beats a simple season average, not whether it calls games.")

acc = cache.read_parquet_or_none(cache.dc_path("accuracy_history_v1.parquet"))
if acc is None or acc.empty:
    st.info("The accuracy tracker has not been built yet. Once predictions have "
            "been generated for past dates and scored against results, this page "
            "will show model-versus-baseline error per stat over time.")
    render_footer()
    st.stop()

# Per-stat summary: model MAE vs season-average baseline MAE.
if {"stat", "abs_err_model", "abs_err_b2"}.issubset(acc.columns):
    summary = (acc.groupby("stat")
               .agg(n=("abs_err_model", "size"),
                    model_MAE=("abs_err_model", "mean"),
                    seasonavg_MAE=("abs_err_b2", "mean"))
               .reset_index())
    summary["edge"] = (summary["seasonavg_MAE"] - summary["model_MAE"]).round(3)
    # Percentage edge is scale-free, so counts (Pitches ~9 MAE) and rate stats
    # (~0.2 MAE) sit on one comparable axis instead of one bar dwarfing the rest.
    summary["edge_pct"] = (100 * summary["edge"]
                           / summary["seasonavg_MAE"].replace(0, float("nan")))
    summary = summary.round(3)

    import plotly.graph_objects as go

    from mlblib.theme import theme_fig

    chart = summary.sort_values("edge_pct", ascending=True)
    _dark = st.session_state.get("theme_dark", False)
    pos = "#16d4c1" if _dark else "#0fae9d"
    neg = "#e74c3c" if _dark else "#dc3a2c"
    colors = [pos if v >= 0 else neg for v in chart["edge_pct"]]
    fig = go.Figure()
    fig.add_bar(orientation="h", y=chart["stat"], x=chart["edge_pct"],
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in chart["edge_pct"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:+.1f}% better than season avg<extra></extra>")
    fig.update_layout(height=430, margin=dict(l=10, r=40, t=10, b=10),
                      showlegend=False,
                      xaxis_title="% better than a season average (higher is better)")
    st.plotly_chart(theme_fig(fig), use_container_width=True)

    tbl = summary.drop(columns=["edge_pct"]).sort_values("edge", ascending=False)
    st.markdown(store.html_df(
        tbl.rename(columns={"stat": "Stat", "n": "N", "model_MAE": "Model MAE",
                            "seasonavg_MAE": "Season-avg MAE", "edge": "Edge"}),
        label_cols=1, hero=("Edge",)), unsafe_allow_html=True)

# ── Model vs the board: the PrizePicks scoreboard, kept for fun ──────────────
# Reads the committed AGGREGATE (cache/props_summary_v1.json). The per-line
# history is deliberately NOT published: its line values come from a
# third-party mirror of the PrizePicks API, so only derived counts and rates
# are shared. Renders nothing until there is something to show.
_summary = None
_sp = cache.dc_path("props_summary_v1.json")
if _sp.exists():
    try:
        _summary = cache.json_load(_sp)
    except Exception:  # noqa: BLE001
        _summary = None

if _summary and _summary.get("graded"):
    st.markdown('<div class="dv-bar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dv-eyebrow">Model vs the board &middot; '
                'PrizePicks lines</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Lines graded", f"{_summary['graded']:,}")
    c2.metric("Model called it right",
              f"{_summary['right']:,} ({_summary['hit_pct']}%)")
    c3.metric("Slates covered", f"{_summary['dates']}")
    st.caption(
        f"{_summary['date_min']} to {_summary['date_max']}. A PrizePicks entry "
        "needs roughly 54 to 58 percent per leg just to break even, so 50 "
        "percent is not the bar and this is a curiosity rather than a measured "
        "edge over the market. "
        f"{_summary['exact']} line(s) landed exactly on the number, which "
        "lowers the payout tier rather than pushing, so they are excluded.")

    if _summary.get("by_confidence"):
        st.markdown("**By how confident the model was.** This is the honest "
                    "test: if the model knows something, the hit rate should "
                    "climb with its own confidence. It does, but only after "
                    "the predictive distribution was fitted per stat. Under a "
                    "plain Poisson the top confidence bucket was the *worst* "
                    "one.")
        _bc = pd.DataFrame(_summary["by_confidence"])
        _bc["hit_pct"] = _bc["hit_pct"].map(lambda v: f"{v:.1f}%")
        st.markdown(store.html_df(
            _bc.rename(columns={"bucket": "Model confidence", "n": "N",
                                "right": "Right", "hit_pct": "Hit %"}),
            label_cols=1, hero=("Hit %",)), unsafe_allow_html=True)
    if _summary.get("by_top_slice"):
        st.markdown("**How good are the strongest calls?** Ranked by the "
                    "model's own confidence. The dashed reference is the rough "
                    "54 percent per leg a PrizePicks entry needs to break even.")
        _ts = pd.DataFrame(_summary["by_top_slice"])
        _ts["hit"] = _ts["hit_pct"].map(lambda v: f"{v:.1f}%")
        _ts["ci"] = [f"{lo:.1f} to {hi:.1f}" for lo, hi in
                     zip(_ts["ci_lo"], _ts["ci_hi"])]
        st.markdown(store.html_df(
            _ts[["slice", "n", "hit", "ci"]].rename(columns={
                "slice": "Slice", "n": "N", "hit": "Hit %",
                "ci": "95% interval"}),
            label_cols=1, hero=("Hit %",)), unsafe_allow_html=True)
        st.caption("Intervals are widened for the fact that props on the same "
                   "slate are not independent. They reach break-even but none "
                   "of them clear it, so this is not a demonstrated edge. "
                   + ("Every line here is a standard line: no Demons or "
                      "Goblins are in this data at all."
                      if _summary.get("all_standard") else ""))

    if _summary.get("by_direction"):
        _au = _summary.get("always_under_pct")
        st.markdown("**Over vs Under, against the right benchmark.** This "
                    "board's outcomes are not a 50/50 coin flip: unders hit "
                    f"{_au}% of the time on their own. So the fair test for "
                    "each direction is not 50 percent, it is what betting "
                    "that side on everything would have done.")
        _bd = pd.DataFrame(_summary["by_direction"])
        _bd["hit"] = _bd["hit_pct"].map(lambda v: f"{v:.1f}%")
        _bd["bench"] = _bd["benchmark_pct"].map(lambda v: f"{v:.1f}%")
        _bd["skill"] = _bd["skill_pts"].map(lambda v: f"{v:+.1f} pts")
        st.markdown(store.html_df(
            _bd[["lean", "n", "hit", "bench", "skill"]].rename(columns={
                "lean": "Lean", "n": "N", "hit": "Hit %",
                "bench": "Always this side", "skill": "Difference"}),
            label_cols=1, hero=("Difference",)), unsafe_allow_html=True)
        st.caption("Read the top-slice table above through this. Most of the "
                   "model's confident leans are Unders, and unders already win "
                   f"{_au}% of the time here, so the honest measure of the "
                   "model is the gap between its slice and simply betting "
                   "Under on everything.")

    if _summary.get("by_stat"):
        st.markdown("**By stat.**")
        _bs = pd.DataFrame(_summary["by_stat"])
        _bs["hit_pct"] = _bs["hit_pct"].map(lambda v: f"{v:.1f}%")
        st.markdown(store.html_df(
            _bs.rename(columns={"stat": "Stat", "n": "N", "hit_pct": "Hit %"}),
            label_cols=1, hero=("Hit %",)), unsafe_allow_html=True)

    # Per-line detail only when the private history is present locally.
    _ph = cache.read_parquet_or_none(cache.dc_path("props_history_v1.parquet"))
    if _ph is not None and not _ph.empty:
        with st.expander(f"Every graded line ({len(_ph):,}) · local only"):
            st.markdown(store.html_df(_ph.sort_values("date", ascending=False)
                                      .head(500), label_cols=3),
                        unsafe_allow_html=True)

st.markdown("**Scored predictions**")
st.markdown(store.html_df(acc.sort_values("date").tail(200), label_cols=2),
            unsafe_allow_html=True)

render_footer()
