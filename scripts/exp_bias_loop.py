"""Backtest the self-correcting bias loop (round 5, item 2).

Simulates, chronologically over a test season, the correction the daily job
would apply: for each date, a per-stat multiplicative ratio fitted on the
TRAILING 30 days of that season's own predictions vs actuals (never the
current day), clipped to [0.9, 1.1], applied only once >= MIN_DAYS prior
dates exist. Reports per-stat MAE/deviance with and without the loop.

Usage: python scripts/exp_bias_loop.py <bat|pit> [test_year]
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
from scripts.validate_models import BAT_DISPLAY, PIT_DISPLAY, _actual_count  # noqa: E402

PRIOR = [2019, 2020]
TRAIN = [2021, 2022, 2023]
CTX_YEARS = list(range(2018, 2027))
WINDOW_DAYS = 30
MIN_DAYS = 8
CLIP = (0.9, 1.1)


def main(role: str, test_year: int = 2025, drift: bool = False) -> None:
    global TRAIN
    if drift:
        # Drift simulation: train on the 2021-22 regime only (sticky-stuff
        # crackdown + dead ball), test on 2025 -- a real environment gap, the
        # closest available analog to old-regime models facing 2026 ABS.
        TRAIN = [2021, 2022]
    display = BAT_DISPLAY if role == "bat" else PIT_DISPLAY
    targets = M.BAT_TARGETS if role == "bat" else M.PIT_TARGETS

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
    fcols = M.feature_columns(feat)
    feat_train = feat[feat["season"].isin(TRAIN)].reset_index(drop=True)
    feat_test = feat[feat["season"] == test_year].reset_index(drop=True)
    actual = feat_test[["personId", "gamePk"]].merge(counts, on=["personId", "gamePk"],
                                                     how="left")
    dates = pd.to_datetime(actual["officialDate"], errors="coerce")

    artifacts = {t: T.fit_target(feat_train, counts, t, spec,
                                 M.target_feature_cols(t, fcols),
                                 **M.train_kwargs(t))
                 for t, spec in targets.items()}
    opp_key = "PA" if role == "bat" else "p_BF"
    opp_pred = T.predict_target(artifacts[opp_key], feat_test)

    results = {}
    for stat, spec in display.items():
        y = _actual_count(actual, spec)
        kind = spec[0]
        if kind in ("opp", "direct"):
            pred = T.predict_target(artifacts[spec[1]], feat_test)
        else:
            pred = np.zeros(len(feat_test))
            for comp, mult in spec[1].items():
                pred = pred + mult * T.predict_target(artifacts[comp], feat_test)
            pred = pred * opp_pred
        mask = ~np.isnan(y) & ~dates.isna().values
        df = pd.DataFrame({"d": dates.values[mask], "y": y[mask], "p": pred[mask]})
        daily = df.groupby("d").agg(ys=("y", "sum"), ps=("p", "sum")).sort_index()
        # Trailing-window ratio per date, strictly past.
        csum = daily.rolling(f"{WINDOW_DAYS}D").sum()
        ndays = daily["ys"].rolling(f"{WINDOW_DAYS}D").count()
        ratio = (csum["ys"] / csum["ps"]).shift(1)
        enough = ndays.shift(1) >= MIN_DAYS
        ratio = ratio.where(enough, 1.0).clip(*CLIP).fillna(1.0)
        df["corr"] = df["p"] * df["d"].map(ratio)
        base = {"mae": T.mae(df["y"], df["p"]), "dev": T.poisson_deviance(df["y"], df["p"])}
        loop = {"mae": T.mae(df["y"], df["corr"]),
                "dev": T.poisson_deviance(df["y"], df["corr"])}
        dd = 100 * (loop["dev"] - base["dev"]) / base["dev"]
        dm = 100 * (loop["mae"] - base["mae"]) / base["mae"]
        keep = bool(dd < 0 and dm <= 0.1)
        results[stat] = {"base": base, "loop": loop, "dev_chg_pct": dd,
                         "mae_chg_pct": dm, "keep": keep}
        logger.warning("[bias/%s/%s] %s: dev %+0.2f%% mae %+0.2f%% -> %s",
                       role, test_year, stat, dd, dm, "KEEP" if keep else "drop")

    tag = "_drift" if drift else ""
    out = Path("cache") / f"exp_bias_{role}_{test_year}{tag}.json"
    out.write_text(json.dumps(results, indent=1))
    logger.warning("wrote %s", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2025,
         drift="drift" in sys.argv[3:])
