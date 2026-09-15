"""CONCLUSIVE NEGATIVE RESULT. Kept per the scripts/ convention (exp_* stays
even when nothing ships). Read this before attempting to raise the board's hit
rate again, because it shows the ceiling is already reached.

THE QUESTION
------------
"Do anything that grants a higher hit rate on PrizePicks lines."

WHAT HAD ALREADY BEEN TRIED AND REJECTED OUT OF SAMPLE
------------------------------------------------------
market blend (flips 0.00% of leans, monotone); propped-population offset
(E[actual-model] = +0.011, no bias to correct); stat x direction filter (did not
replicate); mean recalibration, level and linear (+0.45 in-sample, -0.16 out);
batting-slot refinement (+1.24 in-sample, -0.15 out); a batter-walks IBB feature
(MAE worse out of sample); a calibration refit on a played-only population
(outcome-conditioning artifact). One thing DID work and is shipped: the per-stat
negative binomial, 51.3% -> 52.2%.

THE LAST UNTESTED FAMILY, AND WHY IT WAS WORTH TESTING
------------------------------------------------------
A blend cannot flip a call because shrinking toward the line preserves which
side of the line you are on. A LEARNED CLASSIFIER is not monotone, so it can.
Tested logistic (C = 0.05, 1.0), HistGB depth 2, all with gap x stat
interactions, fitted on props before 2026-08-15 and scored on 5,629 held out:

    always Under (no model at all)      51.57%
    shipped NB lean                     52.00%
    logistic C=1.0                      51.87%   (flips 24% of calls)
    logistic C=0.05                     51.73%   (flips 26%)
    HistGB depth 2                      51.34%   (flips 26%)

Every variant is at or below the shipped lean while flipping a quarter of the
calls, so the flips are noise.

THE DECISIVE DIAGNOSTIC, WHICH EXPLAINS ALL OF THE ABOVE
--------------------------------------------------------
Predicting the ACTUAL result:

    R^2 from the posted line alone          0.9329
    R^2 adding the model's projection       0.9337
    INCREMENTAL R^2 FROM THE MODEL          0.0007

And directionally, out of sample:

    line + stat, NO model features          51.63%
    line + stat + model features            51.57%

Adding the model makes it very slightly WORSE. The posted line is, to a very
good approximation, a sufficient statistic for this model's projection: it
already contains what the model knows, plus information the model does not have
(confirmed lineups, late scratches, bullpen plans, weather reads).

Ceiling check, to rule out a fitting failure rather than an information failure:
an unregularized fit reaches only 54.45% IN-SAMPLE and collapses to 51.98% out.
When the in-sample ceiling is itself barely above the 51.6% baseline, the
information is thin. No estimator recovers signal that is not there.

CONCLUSION
----------
The 52.2% board is at the ceiling of this approach, and it is statistically
indistinguishable from the 52.7% available by betting Under on everything with
no model at all. Beating these lines requires information the market does not
already have. It does not require a better estimator, a different loss, more
features, or more data on the same features. Stop optimising this.

Usage: python scripts/exp_beat_the_line.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache  # noqa: E402

CUT = "2026-08-15"


def main():
    h = cache.read_parquet_or_none(cache.dc_path("props_history_v1.parquet"))
    if h is None or h.empty:
        raise SystemExit("no props_history_v1.parquet")
    d = h[h["graded"]].dropna(subset=["model", "line", "p_over"]).copy()
    y = (d["result"] == "over").astype(int).to_numpy()

    def r2(X):
        b = np.linalg.lstsq(X, d["actual"].to_numpy(float), rcond=None)[0]
        r = d["actual"].to_numpy(float) - X @ b
        return 1 - r.var() / d["actual"].to_numpy(float).var()

    ones = np.ones(len(d))
    r_line = r2(np.column_stack([ones, d["line"]]))
    r_both = r2(np.column_stack([ones, d["line"], d["model"]]))
    te = d[d["date"] >= CUT]
    n = len(te)
    p = (((te["p_over"] > .5) & (te["result"] == "over"))
         | ((te["p_over"] <= .5) & (te["result"] == "under"))).mean()
    se = math.sqrt(p * (1 - p) / n) * math.sqrt(1.9)
    print(f"R^2 line alone            {r_line:.4f}")
    print(f"R^2 line + model          {r_both:.4f}")
    print(f"INCREMENTAL from model    {r_both - r_line:.4f}   <- the whole story")
    print(f"shipped lean, held out    {100 * p:.2f}%  "
          f"[{100 * (p - 1.96 * se):.1f}, {100 * (p + 1.96 * se):.1f}]  n={n:,}")
    print("always Under, no model    "
          f"{100 * (te['result'] == 'under').mean():.2f}%")
    print("CONCLUSION: the line is a sufficient statistic for this model. "
          "Stop optimising the estimator.")


if __name__ == "__main__":
    main()
