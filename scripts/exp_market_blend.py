"""REJECTED ROUND, kept for reference (scripts/ convention: exp_* stays even
when nothing ships).

QUESTION
--------
After the negative-binomial fit, the residual miscalibration looked like a
LEVEL effect: in the low-confidence bins the model said ~28% Over where reality
was ~45%. Hypothesis was a "propped-population offset": conditional on
PrizePicks posting a prop, maybe the player outperforms his unconditional
projection, so add a level correction for propped players.

FINDING 1: THERE IS NO POPULATION OFFSET TO ADD
-----------------------------------------------
E[actual - model] over 11,792 settled props = +0.0107 (sd 1.456). The model is
essentially UNBIASED on propped players. Per-stat residuals are all within
about +/-0.14. So the premise was wrong.

FINDING 2: THE LINE CARRIES INFORMATION, MONOTONICALLY
------------------------------------------------------
E[actual - model] as a function of (model - line):
    model 0.38 BELOW the line -> model undershoots by 0.391
    model level              -> 0.035
    model 0.37 above         -> overshoots by 0.104
    model 0.73 above         -> overshoots by 0.283
corr(model - line, actual - model) = -0.170, and
OLS actual ~ 1 + model + line gives model 0.410 / LINE 0.594, i.e. the market
deserves ~60% of the weight for predicting the actual value. Per stat the market
dominates even harder: Pitcher K 0.92, Total Bases 0.85, Runs 0.84, Hits 0.62,
Singles 0.19.

FINDING 3: AND YET BLENDING CANNOT HELP THIS BOARD
--------------------------------------------------
Fitted on props before 2026-08-15, evaluated on 5,629 held-out props:
    no blend (shipped NB)   hit 51.95%  calib 0.077   <- still the best
    global blend, no c      hit 51.73%  calib 0.090
    global blend + c        hit 51.73%  calib 0.090
    per-stat blend          hit 51.77%  calib 0.098

The reason is structural, not statistical. A blend shrinks the estimate TOWARD
the line, which is the very threshold the lean is computed against, so it is a
monotone transform that preserves the SIGN of (estimate - line) unless the
correction exceeds the gap. It never does: the overshoot is at most 0.49 of the
gap in every bucket. Measured directly, at the fitted 0.60 weight the lean flips
on 0.00% of props. Blending improves the POINT estimate and cannot change a
single directional call.

CONCLUSION
----------
Nothing shipped. The honest reading is that the market is the stronger
forecaster (it earns ~60% of the weight), but that fact is not convertible into
a better model-vs-market lean, because any shrink toward the line leaves the
side of the line unchanged. Improving direction would require information about
WHEN the model is wrong, not a uniform pull toward the market.

Usage: python scripts/exp_market_blend.py [cutoff_date]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache  # noqa: E402

CUT = "2026-08-15"


def fit_weight(df: pd.DataFrame) -> tuple[float, float]:
    """(weight on the line, intercept) from OLS actual ~ 1 + model + line."""
    X = np.column_stack([np.ones(len(df)), df["model"], df["line"]])
    c, b_model, b_line = np.linalg.lstsq(X, df["actual"], rcond=None)[0]
    total = b_model + b_line
    return (b_line / total if total else 0.0), float(c)


def main(argv):
    cut = argv[0] if argv else CUT
    h = cache.read_parquet_or_none(cache.dc_path("props_history_v1.parquet"))
    if h is None or h.empty:
        raise SystemExit("no props_history_v1.parquet; run import_mirror_lines first")
    g = h[h["result"] != "exact"].dropna(subset=["actual", "model", "line"]).copy()
    g["gap"] = g["model"] - g["line"]
    g["resid"] = g["actual"] - g["model"]

    print(f"n={len(g):,} settled props")
    print(f"FINDING 1  E[actual - model] = {g['resid'].mean():+.4f}  "
          f"(no population offset to add)")
    w, c = fit_weight(g[g["date"] < cut])
    print(f"FINDING 2  weight on the line (fit < {cut}) = {w:.3f}, intercept {c:+.4f}")
    # The SYSTEMATIC overshoot is a bucket-level quantity. Per-row ratios are
    # meaningless here (a single game's residual dwarfs a 0.05 gap, so the
    # per-row median exceeds 1 while the systematic correction is far smaller).
    over = -np.sign(g["gap"]) * g["resid"]
    buckets = pd.qcut(g["gap"].abs(), 6, duplicates="drop")
    worst = 0.0
    for b, sub in g.assign(_b=buckets, _o=over).groupby("_b", observed=True):
        mg = sub["gap"].abs().mean()
        if mg:
            worst = max(worst, sub["_o"].mean() / mg)
    flips = (np.sign((1 - w) * g["model"] + w * g["line"] - g["line"])
             != np.sign(g["gap"])).mean()
    print(f"FINDING 3  largest bucket-mean overshoot/gap = {worst:.2f} "
          f"(< 1 everywhere), so the lean flips on {100 * flips:.2f}% of props")
    print("CONCLUSION nothing to ship: a shrink toward the line preserves the "
          "side of the line.")


if __name__ == "__main__":
    main(sys.argv[1:])
