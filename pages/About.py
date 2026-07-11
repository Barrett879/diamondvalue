"""About / methodology page."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib.theme import render_footer, render_nav, render_page_chrome  # noqa: E402

st.set_page_config(page_title="About · DiamondValue", page_icon="static/favicon.svg",
                   layout="wide")
render_page_chrome()
render_nav("About")

st.markdown('<div class="dv-brand">About Diamond<span class="accent">Value</span></div>',
            unsafe_allow_html=True)

st.markdown(
    """
DiamondValue predicts per-game statistics for every available player on an MLB
slate. Pick a date, open a game, and see what the models expect from each hitter
and starting pitcher.

### Read every number as an expected value

Single-game baseball is dominated by variance. Every number here is the mean of
a distribution, not a call on what will happen. "1.1 expected hits" means the
model's distribution centers near one hit, not that the player will get exactly
one. A great model looks good through calibration and small improvements over a
season average, not by calling individual games.

Pitcher strikeouts are the most predictable per-game stat, so treat them as the
headline number. Batter single-game counting stats carry much less signal;
that is the nature of the sport, not a flaw in the model.

### How it is built

Each stat has its own gradient-boosted model (scikit-learn
HistGradientBoostingRegressor, Poisson loss). Rather than predict a counting
stat directly, the models predict opportunities (plate appearances for hitters,
batters faced for pitchers) and per-opportunity rates, then multiply. Composite
numbers like hits and total bases are derived from the predicted singles,
doubles, triples, and home runs so the table always stays internally
consistent.

Features are strictly point-in-time: everything a model sees was knowable before
first pitch. They include regressed multi-season talent priors (Marcel style),
season-to-date and recent-form rates, the opposing starting pitcher's quality
and handedness, the batter's platoon matchup, the ballpark, the month of the
season, the lineup slot, rest, and team context.

### What v1 deliberately does not do

- It does not model single-game OPS, OBP, or SLG directly. They are undefined on
  zero-plate-appearance games and are nearly pure noise per game, so they are
  derived from predicted components instead.
- It does not project relief-pitcher lines. Whether a reliever appears is a
  bullpen-management decision, not a stats problem. Relievers are noted, not
  projected.
- It does not use batter-versus-pitcher history, hot-streak flags, or
  un-regressed small-sample splits. The best public systems treat those as
  noise, and so does this one.
- Bench hitters are shown with a clearly labeled "if he starts" projection at a
  neutral lineup slot, so every rostered player is visible without pretending he
  is in the lineup.

### Lineups

Probable starters are usually known a day ahead. Batting orders post one to four
hours before first pitch, and scratches happen up to game time. When a lineup is
not posted yet, the site shows a projected lineup built from the team's recent
games and labels it as projected. Once the real lineup posts, the numbers update
and the badge flips to posted.
    """
)

st.divider()
st.caption(
    "Statistics data via MLB Stats API. DiamondValue is an independent, "
    "non-commercial project and is not affiliated with or endorsed by Major "
    "League Baseball."
)

render_footer()
