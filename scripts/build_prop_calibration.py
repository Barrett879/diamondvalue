"""Fit the predictive distribution used for P(Over) on prop lines.

WHY THIS EXISTS
---------------
compare() originally called Over whenever the projected mean beat the line,
which is a mean-vs-median error. That was fixed by computing P(Over) from a
Poisson. Grading 11,792 historical props then showed Poisson is ALSO wrong, in
both directions at once:

    measured var((actual - pred)) / mean(pred), 254,782 pred/actual pairs
      OVERdispersed : TB 2.08, Pitches 2.05, ER 1.51, RBI 1.47, SB 1.10
      ~Poisson      : BB 0.98, K 0.96
      UNDERdispersed: R .94 H .87 1B .86 SO .84 HR .83 3B .82 2B .80
                      PA 0.39, IP 0.28

A negative binomial fixes the overdispersed half (and Total Bases is the single
worst prop on the board, 46.8% hit rate while leaning Over 90% of the time).
It CANNOT fix the underdispersed half, because NB only ever widens a Poisson.
Underdispersion is expected: a count bounded by opportunities is binomial-like,
and nobody hits five home runs in four at-bats.

So two candidates are fitted per stat and the choice is made per stat:
  nb        - negative binomial, variance matched to the measured ratio
  empirical - the conditional distribution of actuals, binned by projected
              mean, read straight off the data. Handles either direction and
              needs no distributional assumption.

Output: cache/prop_calibration_v1.json (small, committed). Keyed by
"{role}:{prediction column}" so compare() can look it up without guessing.

Usage: python scripts/build_prop_calibration.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache  # noqa: E402
from mlblib.cache import logger  # noqa: E402

# accuracy_history display stat -> the prediction column compare() sums.
BAT_KEYS = {"PA": "PA", "H": "H", "HR": "HR", "1B": "b1", "2B": "b2",
            "3B": "b3", "SO": "SO", "BB": "BB", "TB": "TB", "R": "R",
            "RBI": "RBI", "SB": "SB"}
PIT_KEYS = {"K": "K", "BB": "BB", "H": "H", "ER": "ER", "IP": "IP",
            "Pitches": "Pitches"}
MIN_ROWS_PER_BIN = 400      # enough for a stable tail estimate
MAX_BINS = 12
MAX_K = 40                  # survival stored for X > 0 .. X > MAX_K


def _survival(actuals: np.ndarray, kmax: int) -> list[float]:
    """[P(X>0), P(X>1), ... P(X>kmax)] read off the empirical distribution."""
    n = len(actuals)
    return [float((actuals > k).sum()) / n for k in range(kmax + 1)]


def fit_stat(mu: np.ndarray, y: np.ndarray, count_mult: float) -> dict | None:
    """Variance ratio, NB dispersion, and a binned empirical survival table."""
    ok = np.isfinite(mu) & np.isfinite(y)
    mu, y = mu[ok] * count_mult, y[ok] * count_mult
    if len(mu) < MIN_ROWS_PER_BIN * 2:
        return None
    mean_mu = float(mu.mean())
    var_resid = float(np.mean((y - mu) ** 2))
    ratio = var_resid / mean_mu if mean_mu > 0 else 1.0
    # NB2: var = mu * ratio = mu + mu^2 / r  ->  r = mu / (ratio - 1)
    nb_r = (mean_mu / (ratio - 1.0)) if ratio > 1.01 else None

    # Empirical: quantile bins on the projected mean, each with enough rows.
    n_bins = max(1, min(MAX_BINS, len(mu) // MIN_ROWS_PER_BIN))
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(mu, qs))
    if len(edges) < 2:
        return None
    idx = np.clip(np.searchsorted(edges, mu, side="right") - 1, 0, len(edges) - 2)
    kmax = int(min(MAX_K, max(3, np.percentile(y, 99.9) + 3)))
    bins = []
    for b in range(len(edges) - 1):
        sel = y[idx == b]
        if len(sel) < 50:
            sel = y                      # degenerate bin: fall back to pooled
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": int((idx == b).sum()),
                     "mean_mu": float(mu[idx == b].mean()) if (idx == b).any() else mean_mu,
                     "surv": _survival(sel, kmax)})
    return {"n": int(len(mu)), "mean_mu": round(mean_mu, 4),
            "var_ratio": round(ratio, 4),
            "nb_r": (round(nb_r, 4) if nb_r else None),
            "kmax": kmax, "bins": bins}


def main():
    acc = cache.read_parquet_or_none(cache.dc_path("accuracy_history_v1.parquet"))
    if acc is None or acc.empty:
        raise SystemExit("no accuracy_history_v1.parquet to fit on")
    out = {}
    for role, keymap in (("bat", BAT_KEYS), ("pit", PIT_KEYS)):
        sub = acc[acc["role"] == role]
        for disp, col in keymap.items():
            s = sub[sub["stat"] == disp]
            if s.empty:
                continue
            # IP is the one non-count unit: its real count is OUTS.
            mult = 3.0 if (role == "pit" and col == "IP") else 1.0
            fit = fit_stat(s["pred"].to_numpy(float), s["actual"].to_numpy(float), mult)
            if fit is None:
                continue
            fit["count_mult"] = mult
            out[f"{role}:{col}"] = fit
            logger.warning("%s:%-8s n=%6d ratio=%.3f nb_r=%s bins=%d",
                           role, col, fit["n"], fit["var_ratio"],
                           fit["nb_r"], len(fit["bins"]))
    cache.json_save(cache.dc_path("prop_calibration_v1.json"),
                    {"schema": 1, "min_rows_per_bin": MIN_ROWS_PER_BIN,
                     "stats": out})
    logger.warning("wrote prop_calibration_v1.json (%d stats)", len(out))


if __name__ == "__main__":
    main()
