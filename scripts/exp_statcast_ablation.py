"""Per-target Statcast ablation on the 2024 VALIDATION year.

For each target: fit twice on 2021-2023 (with and without the Statcast columns)
and evaluate on 2024. The per-target keep/drop decision is made HERE, on the
validation year, so the 2025 test year stays an honest final check rather than
a feature-selection playground.

Usage: python scripts/exp_statcast_ablation.py bat|pit
Writes cache/exp_statcast_{role}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, features as F, model as M, train as T  # noqa: E402
from mlblib.cache import logger  # noqa: E402

PRIOR = [2019, 2020]
TRAIN = [2021, 2022, 2023]
VAL = 2024
CTX_YEARS = list(range(2018, 2027))

SC_BAT = ["sc_xwoba", "sc_xslg", "sc_avg_ev", "sc_brl_pct", "sc_hardhit",
          "sc_sprint", "opp_sc_xwoba", "opp_sc_brl_pct"]
SC_PIT = ["sc_xwoba_ag", "sc_avg_ev_ag", "sc_brl_pct_ag"]


def main(role: str) -> None:
    sc_cols = SC_BAT if role == "bat" else SC_PIT
    targets = M.BAT_TARGETS if role == "bat" else M.PIT_TARGETS

    hist = F.load_gamelogs(PRIOR + TRAIN + [VAL])
    hist = F.attach_catchers(hist)
    frames = [cache.read_parquet_or_none(cache.dc_path(f"player_universe_{y}_v1.parquet"))
              for y in CTX_YEARS]
    uni = pd.concat([f for f in frames if f is not None],
                    ignore_index=True).drop_duplicates(subset="personId", keep="last")
    hist["pitchHand"] = hist["personId"].map(dict(zip(uni["personId"], uni["pitchHand"])))
    ctx = F.Context(CTX_YEARS)

    feat = T.build_features(hist, role, ctx, uni)
    counts = T._counts_frame(hist, role)
    fcols_full = M.feature_columns(feat)
    fcols_base = [c for c in fcols_full if c not in sc_cols]

    feat_train = feat[feat["season"].isin(TRAIN)].reset_index(drop=True)
    feat_val = feat[feat["season"] == VAL].reset_index(drop=True)
    actual = feat_val[["personId", "gamePk"]].merge(counts, on=["personId", "gamePk"],
                                                    how="left")

    results = {}
    for target, spec in targets.items():
        row = {}
        for label, cols in (("base", fcols_base), ("statcast", fcols_full)):
            art = T.fit_target(feat_train, counts, target, spec, cols)
            pred = T.predict_target(art, feat_val)
            if spec["kind"] == "rate":
                denom_col = spec["denom"]
                y = np.where(actual[denom_col].fillna(0) > 0,
                             actual[spec["count"]].fillna(0) / actual[denom_col].replace(0, np.nan),
                             np.nan)
                mask = ~np.isnan(y)
                w = actual[denom_col].fillna(0).values[mask]
                yv, pv = np.asarray(y)[mask], pred[mask]
                row[label] = {
                    "mae": float(np.average(np.abs(yv - pv), weights=w)),
                    "dev": T.poisson_deviance(yv * w, pv * w),
                }
            else:
                y = actual[spec["count"]].astype(float).values
                mask = ~np.isnan(y)
                row[label] = {"mae": T.mae(y[mask], pred[mask]),
                              "dev": T.poisson_deviance(y[mask], pred[mask])}
        b, s = row["base"], row["statcast"]
        row["dev_chg_pct"] = 100 * (s["dev"] - b["dev"]) / b["dev"]
        row["mae_chg_pct"] = 100 * (s["mae"] - b["mae"]) / b["mae"]
        # Keep rule: deviance must improve, MAE must not degrade more than 0.1%.
        row["keep_statcast"] = bool(row["dev_chg_pct"] < 0 and row["mae_chg_pct"] <= 0.1)
        results[target] = row
        logger.warning("[%s] %s: dev %+0.2f%% mae %+0.2f%% -> %s", role, target,
                       row["dev_chg_pct"], row["mae_chg_pct"],
                       "KEEP" if row["keep_statcast"] else "drop")

    out = Path("cache") / f"exp_statcast_{role}.json"
    out.write_text(json.dumps(results, indent=1))
    logger.warning("wrote %s", out)


if __name__ == "__main__":
    main(sys.argv[1])
