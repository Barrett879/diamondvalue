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

st.markdown("**Scored predictions**")
st.markdown(store.html_df(acc.sort_values("date").tail(200), label_cols=2),
            unsafe_allow_html=True)

render_footer()
