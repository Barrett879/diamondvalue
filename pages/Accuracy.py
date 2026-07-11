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

from mlblib import cache  # noqa: E402
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
    summary = summary.round(3)

    import plotly.graph_objects as go

    from mlblib.theme import theme_fig

    chart = summary.sort_values("edge", ascending=False)
    fig = go.Figure()
    fig.add_bar(name="Model", x=chart["stat"], y=chart["model_MAE"],
                marker_color="#17b890")
    fig.add_bar(name="Season average", x=chart["stat"], y=chart["seasonavg_MAE"],
                marker_color="#8a97a0")
    fig.update_layout(barmode="group", height=360,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12),
                      yaxis_title="Mean absolute error (lower is better)")
    st.plotly_chart(theme_fig(fig), use_container_width=True)

    st.dataframe(summary, hide_index=True, use_container_width=True)

st.markdown("**Scored predictions**")
st.dataframe(acc.sort_values("date").tail(200), hide_index=True, use_container_width=True)

render_footer()
