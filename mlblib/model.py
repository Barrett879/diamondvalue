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
    "p_pitches": {"kind": "opp", "count": "p_pitches"},
    "p_K": {"kind": "rate", "count": "p_K", "denom": "p_BF"},
    "p_BB": {"kind": "rate", "count": "p_BB", "denom": "p_BF"},
    "p_H": {"kind": "rate", "count": "p_H", "denom": "p_BF"},
    "p_HR": {"kind": "rate", "count": "p_HR", "denom": "p_BF"},
    "p_ER": {"kind": "direct", "count": "p_ER"},
}


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """The model-input columns of a feature frame (everything but identity)."""
    return [c for c in feat.columns if c not in ID_COLS]


# Per-target Statcast policy, decided on the 2024 validation year (see
# scripts/exp_statcast_ablation.py and docs/decisions.md): the quality-of-
# contact block did not reliably beat the existing priors anywhere EXCEPT
# stolen bases, where sprint speed improved Poisson deviance by ~3.8% on 2024
# and ~3.0% on 2025 (two independent years). Every other target trains without
# the Statcast columns.
STATCAST_COLS = [
    "sc_xwoba", "sc_xslg", "sc_avg_ev", "sc_brl_pct", "sc_hardhit",
    "sc_sprint", "opp_sc_xwoba", "opp_sc_brl_pct",
    "sc_xwoba_ag", "sc_avg_ev_ag", "sc_brl_pct_ag",
]

# Gated feature blocks: each block's columns train ONLY the targets in its set.
# Membership is decided by per-target ablation on the 2024 validation year
# (scripts/exp_feature_blocks.py); an empty set means the block is computed but
# ships nowhere (kept for future rounds).
FEATURE_BLOCKS = {
    "statcast": {"cols": STATCAST_COLS, "targets": {"SB"}},
    # Round-4 ablation on 2024 (exp_feature_blocks.py): opposing bullpen and
    # own-team form showed nothing above the noise floor anywhere (rejected).
    # The opposing lineup's last-30 form helped the starter's K on BOTH 2024
    # (dev -0.41%) and 2025 (dev -0.19%) -> shipped for p_K. Its 2024 ER gain
    # (-0.64%) FAILED to replicate on 2025 (slightly worse) -> rejected.
    "bullpen": {"cols": ["opp_bp_k", "opp_bp_bb", "opp_bp_hr", "opp_bp_er"],
                "targets": set()},
    "teamform": {"cols": ["own_form_r_pa", "own_form_obp"], "targets": set()},
    "oppform": {"cols": ["opp_form_k", "opp_form_obp"], "targets": {"p_K"}},
    # Tier-1 blocks (2026-07-11). 2024 ablation: batter-level arsenal matchup
    # (mu_xwhiff) showed nothing above the noise floor for ANY batter target --
    # rejected -- while the LINEUP-AGGREGATED version was the round's strongest
    # block for pitchers (averaging 9 batters cancels single-batter noise).
    # Pitcher discipline helped workload targets but HURT p_K (+0.36%,
    # double-counts the Marcel K prior). Sets below are the 2024 survivors;
    # the 2025 confirmation prunes non-replicators.
    # 2025 confirmation pruned hard: p_K (-0.32% dev, stacking with oppform)
    # and p_H (-0.14%) replicated and ship; p_ER produced its SECOND mirage in
    # two rounds, p_HR/p_outs/p_pitches/HBP all failed to replicate. Batter-
    # level matchup and both discipline blocks ship nowhere (columns still
    # computed for future rounds).
    "arsenal_bat": {"cols": ["mu_xwhiff"], "targets": set()},
    "disc_bat": {"cols": ["bat_whiff", "bat_chase", "bat_izcon",
                          "sp_whiff", "sp_fstrike"], "targets": set()},
    "arsenal_pit": {"cols": ["opp_lineup_xwhiff"], "targets": {"p_K", "p_H"}},
    "disc_pit": {"cols": ["p_whiff", "p_fstrike", "p_chase"], "targets": set()},
    # Tier-2 candidate blocks (2026-07-11): game environment + HP umpire.
    "weather_bat": {"cols": ["env_temp", "env_wind_out", "env_roof"],
                    "targets": set()},
    # 2025 confirmation: p_HR (-0.92% dev, biggest effect since sprint->SB),
    # p_H, p_BB, p_K all replicated; p_ER failed for the THIRD time and is
    # struck (its noise swallows even physics-backed signals).
    "weather_pit": {"cols": ["env_temp", "env_wind_out", "env_roof"],
                    "targets": {"p_H", "p_HR"}},
    "ump_bat": {"cols": ["ump_k_delta", "ump_bb_delta"], "targets": set()},
    # Tier-3 candidate: structural expected-PA identity feature.
    "pa_struct": {"cols": ["pa_struct"], "targets": set()},
    "ump_pit": {"cols": ["ump_k_delta", "ump_bb_delta"],
                "targets": {"p_BB", "p_K"}},
}


# ── Tier-3 training techniques (per-target, gauntlet-decided) ────────────────
# Conservative known-direction monotonic maps: only signs that are a priori
# certain. Applied per target when TRAIN_POLICY grants "monotonic". Columns
# absent from a target's feature list are filtered at fit time.
MONOTONIC_MAPS = {
    "PA": {"slot": -1, "marcel_PApg": 1},
    "b1": {"rate_b1": 1}, "b2": {"rate_b2": 1}, "b3": {"rate_b3": 1},
    "HR": {"rate_HR": 1, "marcel_HR": 1, "pf_hr": 1},
    "BB": {"rate_BB": 1, "marcel_BB": 1, "opp_sp_bb_rate": 1},
    "HBP": {"rate_HBP": 1},
    "SO": {"rate_SO": 1, "marcel_SO": 1, "opp_sp_k_rate": 1},
    "R": {"own_team_r_pa": 1}, "RBI": {"own_team_r_pa": 1},
    "SB": {"sc_sprint": 1},
    "p_outs": {"marcel_BFpg": 1}, "p_BF": {"marcel_BFpg": 1},
    "p_pitches": {"marcel_BFpg": 1},
    "p_K": {"rate_p_K": 1, "opp_team_k": 1, "opp_lineup_xwhiff": 1,
            "ump_k_delta": 1, "opp_form_k": 1},
    "p_BB": {"rate_p_BB": 1, "ump_bb_delta": 1},
    "p_H": {"rate_p_H": 1, "opp_team_obp": 1},
    "p_HR": {"rate_p_HR": 1, "pf_hr": 1, "env_temp": 1, "env_wind_out": 1},
}

# Per-target training-technique grants (recency half-life in days, monotonic
# on/off), decided by the tier-3 gauntlet (2024 ablation + 2025 confirmation):
# PA recency-300 replicated at -1.37%/-1.10% dev (role-driven stat, short
# memory wins; PA feeds every batter counting stat); monotonic constraints
# confirmed for p_outs (-0.47%, its first-ever win), p_K (fifth stacked gain),
# p_BB, and b3. Caught: p_BF recency was a full mirage (-0.42% -> +1.82%);
# p_ER is now 0-for-4; batter-PA monotonic was catastrophic (+4.4%).
# FULL-HISTORY AUDIT (see decisions.md): the technique harness loaded a
# thinner 2025 history than the real validator, and its p_outs/p_K monotonic
# "confirmations" reversed under the true configuration (+0.10%/+0.06%) --
# reverted. PA recency (-1.09%), b3 mono (-0.28%), p_BB mono (dev and MAE
# both better in every configuration) survived the audit.
TRAIN_POLICY: dict = {
    "PA": {"recency_half_life": 300.0},
    "b3": {"monotonic": True},
    "p_BB": {"monotonic": True},
}


def train_kwargs(target: str) -> dict:
    """fit_target kwargs for one target under the shipped technique policy."""
    pol = TRAIN_POLICY.get(target, {})
    out = {}
    if pol.get("recency_half_life"):
        out["recency_half_life"] = pol["recency_half_life"]
    if pol.get("monotonic"):
        out["monotonic"] = MONOTONIC_MAPS.get(target, {})
    return out


def target_feature_cols(target: str, all_cols: list[str]) -> list[str]:
    """The feature list one target's model trains on (the block policy).

    A gated column is kept iff AT LEAST ONE block containing it grants this
    target. (An earlier any-block-can-veto version silently stripped columns
    shared between per-role blocks -- weather_bat's empty set vetoed
    weather_pit's grant -- which made a confirmation run train byte-identical
    models. Grant-wins semantics fixes that.)
    """
    gated: dict[str, bool] = {}
    for block in FEATURE_BLOCKS.values():
        granted = target in block["targets"]
        for c in block["cols"]:
            gated[c] = gated.get(c, False) or granted
    return [c for c in all_cols if gated.get(c, True)]


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
    out["Pitches"] = _safe_predict("p_pitches", artifacts, feat)
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
