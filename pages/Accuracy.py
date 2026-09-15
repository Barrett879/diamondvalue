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
# Only appears once some pasted lines have been graded. Framed as a running
# tally, not a claim: the sample is whatever boards got pasted, so it is a
# curiosity rather than evidence of a market edge.
ph = cache.read_parquet_or_none(cache.dc_path("props_history_v1.parquet"))
if ph is not None and not ph.empty:
    st.markdown('<div class="dv-bar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dv-eyebrow">Model vs the board &middot; '
                'PrizePicks lines</div>', unsafe_allow_html=True)
    g = ph[ph["graded"]]
    n_g, n_right = len(g), int(g["model_right"].sum())
    pct = (100.0 * n_right / n_g) if n_g else 0.0
    n_exact = int((ph["result"] == "exact").sum())
    n_days = ph["date"].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Lines graded", f"{n_g}")
    c2.metric("Model called it right", f"{n_right} ({pct:.0f}%)")
    c3.metric("Slates with lines", f"{n_days}")
    if n_g:
        # Bucket by CONFIDENCE, not by the raw mean-vs-line gap: the lean comes
        # from P(Over), and a big gap on a low line can still be a coin flip.
        # Falls back to the gap for rows graded before p_over was recorded.
        conf = ((g["p_over"] - 0.5).abs() if "p_over" in g.columns
                else pd.Series(float("nan"), index=g.index))
        if conf.notna().any():
            b = g.assign(bucket=pd.cut(conf, [0, 0.05, 0.10, 0.20, 0.50],
                                       labels=["coin flip (50-55%)", "55-60%",
                                               "60-70%", "70%+"],
                                       include_lowest=True))
            gap_label = "Model confidence"
        else:
            b = g.assign(bucket=pd.cut(g["edge"].abs(),
                                       [0, 0.25, 0.5, 1.0, float("inf")],
                                       labels=["under 0.25", "0.25 to 0.5",
                                               "0.5 to 1.0", "over 1.0"]))
            gap_label = "Model-vs-line gap"
        by = (b.groupby("bucket", observed=True)
              .agg(N=("model_right", "size"), right=("model_right", "sum"))
              .reset_index())
        # Formatted as a string: html_df renders raw floats to 3 decimals,
        # which turns a hit rate into "25.000".
        by["hit_rate"] = [f"{100 * r / n:.0f}%" for r, n in zip(by["right"], by["N"])]
        st.markdown(store.html_df(
            by.rename(columns={"bucket": gap_label, "hit_rate": "Hit %"}),
            label_cols=1, hero=("Hit %",)), unsafe_allow_html=True)
    st.caption(
        f"{n_exact} line(s) landed exactly on the number, which PrizePicks "
        "treats as a lower payout tier rather than a push, so they are shown "
        "but not counted either way. Lines are only graded when the side the "
        "model leans is one the board actually offered. This tally covers "
        "whichever boards happened to get pasted, so treat it as a curiosity, "
        "not a measured edge over the market. For scale, a PrizePicks entry "
        "needs roughly 54 to 58 percent per leg just to break even, so 50 "
        "percent is not the bar and a few hundred props cannot separate a real "
        "edge from luck.")
    with st.expander(f"Every graded line ({len(ph)})"):
        st.markdown(store.html_df(ph.sort_values("date", ascending=False),
                                  label_cols=3), unsafe_allow_html=True)

st.markdown("**Scored predictions**")
st.markdown(store.html_df(acc.sort_values("date").tail(200), label_cols=2),
            unsafe_allow_html=True)

render_footer()
