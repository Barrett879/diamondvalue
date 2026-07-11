"""Train the shipped production models on all available data (2021-2025, with
2019-2020 loaded only to feed Marcel priors). One joblib artifact per target in
models/. For the honest out-of-sample report, see validate_models.py.

Usage:
  python scripts/train_models.py                 # all targets, seasons 2021-2025
  python scripts/train_models.py --bat           # batter targets only
  python scripts/train_models.py --pit           # pitcher targets only
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, features as F, model as M, train as T  # noqa: E402
from mlblib.cache import logger  # noqa: E402

PRIOR_SEASONS = [2019, 2020]
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
CTX_YEARS = list(range(2018, 2027))


def _load_history():
    seasons = PRIOR_SEASONS + TRAIN_SEASONS
    hist = F.load_gamelogs(seasons)
    if hist.empty:
        raise SystemExit("no gamelogs found; run build_training_backfill.py first")
    hist = F.attach_catchers(hist)
    # Attach each player's own pitchHand (for the pitcher 'hand' feature).
    uni = _universe(CTX_YEARS)
    ph = dict(zip(uni["personId"], uni["pitchHand"]))
    hist["pitchHand"] = hist["personId"].map(ph)
    return hist, uni


def _universe(years):
    import pandas as pd
    frames = []
    for y in years:
        df = cache.read_parquet_or_none(cache.dc_path(f"player_universe_{y}_v1.parquet"))
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["personId", "birthDate", "pitchHand", "batSide"])
    u = pd.concat(frames, ignore_index=True)
    return u.drop_duplicates(subset=["personId"], keep="last")


def _train_role(role: str, hist, ctx, uni):
    targets = M.BAT_TARGETS if role == "bat" else M.PIT_TARGETS
    logger.warning("[%s] building features ...", role)
    feat = T.build_features(hist, role, ctx, uni)
    feat = feat[feat["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    counts = T._counts_frame(hist, role)
    fcols = M.feature_columns(feat)
    logger.warning("[%s] %d feature rows, %d feature cols", role, len(feat), len(fcols))
    for target, spec in targets.items():
        art = T.fit_target(feat, counts, target, spec,
                           M.target_feature_cols(target, fcols))
        out = Path(__file__).resolve().parent.parent / "models" / f"{target}_histgb_{M.MODEL_VERSION}.joblib"
        joblib.dump(art, out)
        logger.warning("  saved %s (%d rows)", out.name, art["n_rows"])


def main(argv):
    do_bat = "--pit" not in argv
    do_pit = "--bat" not in argv
    hist, uni = _load_history()
    ctx = F.Context(CTX_YEARS)
    logger.warning("history rows: %d (seasons %s)", len(hist),
                   sorted(hist["season"].unique()))
    if do_bat:
        _train_role("bat", hist, ctx, uni)
    if do_pit:
        _train_role("pit", hist, ctx, uni)
    logger.warning("done.")


if __name__ == "__main__":
    main(sys.argv[1:])
