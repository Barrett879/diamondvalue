"""Per-target feature-block ablation on the 2024 VALIDATION year.

Generalizes exp_statcast_ablation.py: for a role and a named block from
mlblib.model.FEATURE_BLOCKS, fit every target twice on 2021-2023 (current
policy vs policy + block) and evaluate on 2024. Keep decisions are made HERE so
2025 stays an honest final check.

Usage: python scripts/exp_feature_blocks.py <bat|pit> <block>
Writes cache/exp_block_{role}_{block}.json
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


def main(role: str, block_name: str) -> None:
    block = M.FEATURE_BLOCKS[block_name]
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
    fcols_all = M.feature_columns(feat)
    block_cols = [c for c in block["cols"] if c in fcols_all]
    if not block_cols:
        raise SystemExit(f"block {block_name} has no columns in the {role} frame")

    feat_train = feat[feat["season"].isin(TRAIN)].reset_index(drop=True)
    feat_val = feat[feat["season"] == VAL].reset_index(drop=True)
    actual = feat_val[["personId", "gamePk"]].merge(counts, on=["personId", "gamePk"],
                                                    how="left")

    results = {}
    for target, spec in targets.items():
        base_cols = M.target_feature_cols(target, fcols_all)
        test_cols = base_cols + [c for c in block_cols if c not in base_cols]
        row = {}
        for label, cols in (("base", base_cols), ("block", test_cols)):
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
                row[label] = {"mae": float(np.average(np.abs(yv - pv), weights=w)),
                              "dev": T.poisson_deviance(yv * w, pv * w)}
            else:
                y = actual[spec["count"]].astype(float).values
                mask = ~np.isnan(y)
                row[label] = {"mae": T.mae(y[mask], pred[mask]),
                              "dev": T.poisson_deviance(y[mask], pred[mask])}
        b, s = row["base"], row["block"]
        row["dev_chg_pct"] = 100 * (s["dev"] - b["dev"]) / b["dev"]
        row["mae_chg_pct"] = 100 * (s["mae"] - b["mae"]) / b["mae"]
        row["keep"] = bool(row["dev_chg_pct"] < 0 and row["mae_chg_pct"] <= 0.1)
        results[target] = row
        logger.warning("[%s/%s] %s: dev %+0.2f%% mae %+0.2f%% -> %s", role, block_name,
                       target, row["dev_chg_pct"], row["mae_chg_pct"],
                       "KEEP" if row["keep"] else "drop")

    out = Path("cache") / f"exp_block_{role}_{block_name}.json"
    out.write_text(json.dumps(results, indent=1))
    logger.warning("wrote %s", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
