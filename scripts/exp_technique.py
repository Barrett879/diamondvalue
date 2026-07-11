"""Per-target TRAINING-TECHNIQUE ablation (tier 3), on the 2024 validation
year by default; pass a third arg "2025" for the confirmation run.

Unlike feature-block ablation, both arms use the SAME features; what changes
is how the model trains:
  recency550 / recency300  exponential sample-weight day-decay (half-life days)
  mono                     known-direction monotonic constraints

Usage: python scripts/exp_technique.py <bat|pit> <variant> [test_year]
Writes cache/exp_tech_{role}_{variant}[_2025].json
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
CTX_YEARS = list(range(2018, 2027))

VARIANTS = {
    "recency550": {"recency_half_life": 550.0},
    "recency300": {"recency_half_life": 300.0},
    "mono": {"monotonic": True},
}


def main(role: str, variant: str, test_year: int = 2024) -> None:
    vk = VARIANTS[variant]
    targets = M.BAT_TARGETS if role == "bat" else M.PIT_TARGETS

    # Load the SAME history the real validator uses: 2024 must be present
    # even when testing 2025, or 2025 rows get thinner Marcel/as-of features
    # than production and the verdicts do not transfer (this bug shipped two
    # false confirmations before the full-history audit caught them).
    hist = F.load_gamelogs(PRIOR + TRAIN + sorted({2024, test_year}))
    hist = F.attach_catchers(hist)
    frames = [cache.read_parquet_or_none(cache.dc_path(f"player_universe_{y}_v1.parquet"))
              for y in CTX_YEARS]
    uni = pd.concat([f for f in frames if f is not None],
                    ignore_index=True).drop_duplicates(subset="personId", keep="last")
    hist["pitchHand"] = hist["personId"].map(dict(zip(uni["personId"], uni["pitchHand"])))
    ctx = F.Context(CTX_YEARS)

    feat = T.build_features(hist, role, ctx, uni)
    counts = T._counts_frame(hist, role)
    fcols_all = M.feature_columns(feat)
    feat_train = feat[feat["season"].isin(TRAIN)].reset_index(drop=True)
    feat_val = feat[feat["season"] == test_year].reset_index(drop=True)
    actual = feat_val[["personId", "gamePk"]].merge(counts, on=["personId", "gamePk"],
                                                    how="left")

    results = {}
    for target, spec in targets.items():
        cols = M.target_feature_cols(target, fcols_all)
        kwargs = {}
        if "recency_half_life" in vk:
            kwargs["recency_half_life"] = vk["recency_half_life"]
        if vk.get("monotonic"):
            mono = M.MONOTONIC_MAPS.get(target, {})
            if not mono:
                continue  # no map for this target; nothing to test
            kwargs["monotonic"] = mono
        row = {}
        for label, kw in (("base", {}), ("tech", kwargs)):
            art = T.fit_target(feat_train, counts, target, spec, cols, **kw)
            pred = T.predict_target(art, feat_val)
            if spec["kind"] == "rate":
                denom_col = spec["denom"]
                y = np.where(actual[denom_col].fillna(0) > 0,
                             actual[spec["count"]].fillna(0) / actual[denom_col].replace(0, np.nan),
                             np.nan)
                mask = ~np.isnan(y)
                w = actual[denom_col].fillna(0).values[mask]
                yv, pv = np.asarray(y)[mask], pred[mask]
                row[label] = {"mae": float(np.average(np.abs(yv - pv), weights=w)),
                              "dev": T.poisson_deviance(yv * w, pv * w)}
            else:
                y = actual[spec["count"]].astype(float).values
                mask = ~np.isnan(y)
                row[label] = {"mae": T.mae(y[mask], pred[mask]),
                              "dev": T.poisson_deviance(y[mask], pred[mask])}
        b, t = row["base"], row["tech"]
        row["dev_chg_pct"] = 100 * (t["dev"] - b["dev"]) / b["dev"]
        row["mae_chg_pct"] = 100 * (t["mae"] - b["mae"]) / b["mae"]
        row["keep"] = bool(row["dev_chg_pct"] < 0 and row["mae_chg_pct"] <= 0.1)
        results[target] = row
        logger.warning("[%s/%s/%s] %s: dev %+0.2f%% mae %+0.2f%% -> %s",
                       role, variant, test_year, target,
                       row["dev_chg_pct"], row["mae_chg_pct"],
                       "KEEP" if row["keep"] else "drop")

    suffix = "" if test_year == 2024 else f"_{test_year}"
    out = Path("cache") / f"exp_tech_{role}_{variant}{suffix}.json"
    out.write_text(json.dumps(results, indent=1))
    logger.warning("wrote %s", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2024)
