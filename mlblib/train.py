"""Shared training + evaluation helpers, imported by both
scripts/train_models.py (ships production artifacts trained on all data) and
scripts/validate_models.py (walk-forward OOS report). Keeping the fit logic in
one place guarantees the two use identical feature assembly and target math.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features as F
from . import model as M

# Proven-range hyperparameters (spec 4.2). Poisson loss for every target.
HGB_PARAMS = dict(
    loss="poisson",
    max_iter=600,
    max_depth=4,
    learning_rate=0.03,
    min_samples_leaf=40,
    l2_regularization=0.1,
    early_stopping=False,
    random_state=0,
)


def build_features(history: pd.DataFrame, role: str, ctx, universe) -> pd.DataFrame:
    if role == "bat":
        return F.compute_batter_features(history, ctx=ctx, universe=universe)
    return F.compute_pitcher_features(history, ctx=ctx, universe=universe)


def _counts_frame(history: pd.DataFrame, role: str) -> pd.DataFrame:
    """One row per (personId, gamePk) with the raw counting stats, for aligning
    targets to the feature frame.
    """
    if role == "bat":
        m = history[history["played"] & history["is_batter"]]
        cols = ["personId", "gamePk", "officialDate"] + F.BAT_COUNTS
    else:
        m = history[history["played"] & history["is_pitcher"] & history["is_sp"]]
        cols = ["personId", "gamePk", "officialDate"] + F.PIT_COUNTS
    return m[cols].drop_duplicates(subset=["personId", "gamePk"])


def target_yw(counts: pd.DataFrame, target: str, spec: dict) -> pd.DataFrame:
    """Return (personId, gamePk, y, w) for a target given the counts frame."""
    df = counts.copy()
    kind = spec["kind"]
    if kind == "opp" or kind == "direct":
        df["y"] = df[spec["count"]].astype(float)
        df["w"] = 1.0
    elif kind == "rate":
        denom = df[spec["denom"]].astype(float)
        df["y"] = np.where(denom > 0, df[spec["count"]].astype(float) / denom, 0.0)
        df["w"] = denom
        df = df[denom > 0]
    return df[["personId", "gamePk", "y", "w"]]


def fit_target(feat_train: pd.DataFrame, counts: pd.DataFrame, target: str,
               spec: dict, feature_cols: list[str],
               recency_half_life: float | None = None,
               monotonic: dict | None = None) -> dict:
    """Fit one HistGBR for a target and return a serializable artifact dict.

    recency_half_life: optional exponential day-decay of sample weights,
      w *= 0.5^(days_before_train_end / half_life). Composes with the Poisson
      exposure weights.
    monotonic: optional {feature_name: +1|-1} map passed to HistGB's
      monotonic_cst (variance reduction on known-direction features).
    """
    yw = target_yw(counts, target, spec)
    data = feat_train.merge(yw, on=["personId", "gamePk"], how="inner")
    X = data.reindex(columns=feature_cols)
    y = data["y"].values
    w = data["w"].values.astype(float)
    if recency_half_life:
        dates = counts[["personId", "gamePk", "officialDate"]]
        data2 = data[["personId", "gamePk"]].merge(dates, on=["personId", "gamePk"],
                                                   how="left")
        d = pd.to_datetime(data2["officialDate"], errors="coerce")
        days_ago = (d.max() - d).dt.days.fillna(0).values
        w = w * np.power(0.5, days_ago / float(recency_half_life))
    params = dict(HGB_PARAMS)
    if monotonic:
        cst = {c: v for c, v in monotonic.items() if c in feature_cols}
        if cst:
            params["monotonic_cst"] = cst
    est = HistGradientBoostingRegressor(**params)
    est.fit(X, y, sample_weight=w)
    return {
        "model": est,
        "feature_cols": list(feature_cols),
        "model_class": "HistGradientBoostingRegressor",
        "target": target,
        "kind": spec["kind"],
        "loss": "poisson",
        "n_rows": int(len(data)),
        "recency_half_life": recency_half_life,
        "monotonic": monotonic or {},
        "MODEL_VERSION": M.MODEL_VERSION,
    }


def predict_target(artifact: dict, feat: pd.DataFrame) -> np.ndarray:
    X = feat.reindex(columns=artifact["feature_cols"])
    return artifact["model"].predict(X)


# ── Baselines + metrics (validation) ─────────────────────────────────────────
def poisson_deviance(y, yhat) -> float:
    y = np.asarray(y, float)
    yhat = np.clip(np.asarray(yhat, float), 1e-9, None)
    # np.where evaluates both branches, so the y=0 branch computes log(0) and
    # warns even though it is discarded; compute the log only where y>0.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y > 0, y / yhat, 1.0)
        term = np.where(y > 0, y * np.log(ratio), 0.0)
    return float(2.0 * np.mean(term - (y - yhat)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def decile_calibration(y, yhat, n=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, float), "yhat": np.asarray(yhat, float)})
    df = df.sort_values("yhat").reset_index(drop=True)
    df["bucket"] = (np.arange(len(df)) * n // len(df)).clip(0, n - 1)
    return df.groupby("bucket").agg(pred_mean=("yhat", "mean"),
                                    real_mean=("y", "mean"),
                                    n=("y", "size")).reset_index()
