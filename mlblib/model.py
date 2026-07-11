"""Model artifacts: target catalog, the feature-vector contract, prediction,
and the derived-stat math (H, TB, AB, AVG, OBP, SLG, OPS from components).

The feature-order contract: an artifact stores the exact `feature_cols` list it
was trained on. Both the trainer (scripts/train_models.py) and the app select
features by reindexing to that stored list, so column ORDER can never drift
between training and inference. Do not reorder or rename feature columns in
mlblib/features.py without retraining. FEATURE ORDER MUST MATCH.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cache

MODEL_VERSION = "m1"

# Identity columns that are never model inputs.
ID_COLS = ["personId", "gamePk", "season"]

# Batter targets. kind: 'opp' opportunity (raw count), 'rate' per-PA rate
# (trained count/PA weighted by PA), 'direct' raw per-game count.
BAT_TARGETS = {
    "PA": {"kind": "opp", "count": "PA"},
    "b1": {"kind": "rate", "count": "b1", "denom": "PA"},
    "b2": {"kind": "rate", "count": "b2", "denom": "PA"},
    "b3": {"kind": "rate", "count": "b3", "denom": "PA"},
    "HR": {"kind": "rate", "count": "HR", "denom": "PA"},
    "BB": {"kind": "rate", "count": "BB", "denom": "PA"},
    "HBP": {"kind": "rate", "count": "HBP", "denom": "PA"},
    "SO": {"kind": "rate", "count": "SO", "denom": "PA"},
    "R": {"kind": "direct", "count": "R"},
    "RBI": {"kind": "direct", "count": "RBI"},
    "SB": {"kind": "direct", "count": "SB"},
}

# Pitcher (starter) targets.
PIT_TARGETS = {
    "p_outs": {"kind": "opp", "count": "p_outs"},
    "p_BF": {"kind": "opp", "count": "p_BF"},
    "p_K": {"kind": "rate", "count": "p_K", "denom": "p_BF"},
    "p_BB": {"kind": "rate", "count": "p_BB", "denom": "p_BF"},
    "p_H": {"kind": "rate", "count": "p_H", "denom": "p_BF"},
    "p_HR": {"kind": "rate", "count": "p_HR", "denom": "p_BF"},
    "p_ER": {"kind": "direct", "count": "p_ER"},
}


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """The model-input columns of a feature frame (everything but identity)."""
    return [c for c in feat.columns if c not in ID_COLS]


def _artifact_file(target: str):
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "models" / f"{target}_histgb_{MODEL_VERSION}.joblib"


def load_artifacts(targets: list[str]) -> dict:
    """Load joblib artifacts for the given targets. Missing files return None
    for that target (the app degrades gracefully instead of crashing)."""
    import joblib

    out = {}
    for t in targets:
        p = _artifact_file(t)
        if p.exists():
            try:
                out[t] = joblib.load(p)
            except Exception as e:  # noqa: BLE001
                cache.logger.warning("artifact load failed for %s: %s", t, e)
                out[t] = None
        else:
            out[t] = None
    return out


def _predict_one(artifact: dict, feat: pd.DataFrame) -> np.ndarray:
    """Predict with one artifact, honoring its stored feature_cols order."""
    cols = artifact["feature_cols"]
    X = feat.reindex(columns=cols)
    return artifact["model"].predict(X)


def predict_batters(feat: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Assemble the batter prediction table from component models.

    Returns one row per input feature row with columns:
      personId, gamePk, PA, b1,b2,b3,HR,BB,HBP,SO, H, TB, AB, R, RBI, SB,
      AVG, OBP, SLG, OPS. Rate components are multiplied by predicted PA.
    """
    out = pd.DataFrame({"personId": feat["personId"].values,
                        "gamePk": feat["gamePk"].values})
    pa = _safe_predict("PA", artifacts, feat)
    out["PA"] = pa
    for c in ["b1", "b2", "b3", "HR", "BB", "HBP", "SO"]:
        rate = _safe_predict(c, artifacts, feat)
        out[c] = rate * pa if rate is not None else np.nan
    for c in ["R", "RBI", "SB"]:
        out[c] = _safe_predict(c, artifacts, feat)
    # Derived stats (canonical table in the spec).
    out["H"] = out[["b1", "b2", "b3", "HR"]].sum(axis=1)
    out["TB"] = out["b1"] + 2 * out["b2"] + 3 * out["b3"] + 4 * out["HR"]
    out["AB"] = (out["PA"] - out["BB"] - out["HBP"] - 0.008 * out["PA"]).clip(lower=0.0)
    out["AVG"] = out["H"] / out["AB"].replace(0, np.nan)
    out["OBP"] = (out["H"] + out["BB"] + out["HBP"]) / out["PA"].replace(0, np.nan)
    out["SLG"] = out["TB"] / out["AB"].replace(0, np.nan)
    out["OPS"] = out["OBP"] + out["SLG"]
    return out


def predict_pitchers(feat: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Assemble the starter prediction table. Returns personId, gamePk, IP, BF,
    K, BB, H, HR, ER.
    """
    out = pd.DataFrame({"personId": feat["personId"].values,
                        "gamePk": feat["gamePk"].values})
    outs = _safe_predict("p_outs", artifacts, feat)
    bf = _safe_predict("p_BF", artifacts, feat)
    out["IP"] = outs / 3.0 if outs is not None else np.nan
    out["BF"] = bf
    for c in ["p_K", "p_BB", "p_H", "p_HR"]:
        rate = _safe_predict(c, artifacts, feat)
        out[c.replace("p_", "")] = rate * bf if (rate is not None and bf is not None) else np.nan
    out["ER"] = _safe_predict("p_ER", artifacts, feat)
    return out


def _safe_predict(target, artifacts, feat):
    art = artifacts.get(target)
    if art is None:
        return np.full(len(feat), np.nan)
    try:
        return _predict_one(art, feat)
    except Exception as e:  # noqa: BLE001
        cache.logger.warning("predict failed for %s: %s", target, e)
        return np.full(len(feat), np.nan)
