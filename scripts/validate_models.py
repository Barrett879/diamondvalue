"""Walk-forward validation and the model report (Phase 3 STOP deliverable).

Trains on 2021-2023, tests on 2025 (2024 reserved for tuning; 2019-2020 feed
priors). Evaluates every DISPLAYED per-game stat at the count level the user
sees: model prediction vs three point-in-time baselines
  B1 league-average constant
  B2 player season-to-date mean excluding the target game
  B3 Marcel-style weighted rate x expected opportunities
on MAE and Poisson deviance, plus decile calibration. Writes docs/model_report.md
opening with a plain-English per-target recommendation.

Usage: python scripts/validate_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, features as F, model as M, train as T  # noqa: E402
from mlblib.cache import logger  # noqa: E402

PRIOR = [2019, 2020]
TRAIN = [2021, 2022, 2023]
TEST = 2025
CTX_YEARS = list(range(2018, 2027))

# Displayed stats -> how the model builds the per-game count.
#   ('opp', col)          model predicts the count directly
#   ('direct', col)       model predicts the count directly
#   ('rate', {comp:mult}) count = sum(mult * rate_comp) * opp_pred
BAT_DISPLAY = {
    "PA": ("opp", "PA"),
    "H": ("rate", {"b1": 1, "b2": 1, "b3": 1, "HR": 1}),
    "TB": ("rate", {"b1": 1, "b2": 2, "b3": 3, "HR": 4}),
    "HR": ("rate", {"HR": 1}),
    "BB": ("rate", {"BB": 1}),
    "SO": ("rate", {"SO": 1}),
    "b2": ("rate", {"b2": 1}),
    "b3": ("rate", {"b3": 1}),
    "R": ("direct", "R"),
    "RBI": ("direct", "RBI"),
    "SB": ("direct", "SB"),
}
PIT_DISPLAY = {
    "p_outs": ("opp", "p_outs"),
    "p_BF": ("opp", "p_BF"),
    "p_pitches": ("opp", "p_pitches"),
    "p_K": ("rate", {"p_K": 1}),
    "p_BB": ("rate", {"p_BB": 1}),
    "p_H": ("rate", {"p_H": 1}),
    "p_HR": ("rate", {"p_HR": 1}),
    "p_ER": ("direct", "p_ER"),
}


def _universe():
    frames = [cache.read_parquet_or_none(cache.dc_path(f"player_universe_{y}_v1.parquet"))
              for y in CTX_YEARS]
    frames = [f for f in frames if f is not None]
    u = pd.concat(frames, ignore_index=True)
    return u.drop_duplicates(subset=["personId"], keep="last")


def _asof_baseline_frame(history, role):
    """Per (personId, gamePk): std_{c}/std_G (B2) and marcel_{c}, opp-per-game
    (B3), for baseline construction. Point-in-time by the same shift/prior rules
    as features.
    """
    if role == "bat":
        counts, denom, weights, ballast = F.BAT_COUNTS, "PA", F.MARCEL_HIT_WEIGHTS, F.HITTER_BALLAST_PA
        sub = history[history["played"] & history["is_batter"]].copy()
        prefix = "b_"
    else:
        counts, denom, weights, ballast = F.PIT_COUNTS, "p_BF", F.MARCEL_PIT_WEIGHTS, F.PITCHER_BALLAST_BF
        sub = history[history["played"] & history["is_pitcher"] & history["is_sp"]].copy()
        prefix = "p_"
    league = F._league_rates(sub, counts, denom)
    sub = sub.sort_values(["personId", "gameDate", "gameNumber"]).reset_index(drop=True)
    sub = F._add_asof_aggregates(sub, counts, prefix=prefix)
    marcel = F._marcel_prior_rates(sub, counts, denom, weights, ballast, league)
    sub = sub.merge(marcel, on=["personId", "season"], how="left")
    keep = ["personId", "gamePk", "season", f"{prefix}std_G"] + \
           [f"{prefix}std_{c}" for c in counts] + \
           [f"marcel_{c}" for c in counts] + [f"marcel_{denom}pg"]
    return sub[keep], prefix, denom, league


def _model_count(role, feat, artifacts, display_spec, opp_pred):
    kind = display_spec[0]
    if kind in ("opp", "direct"):
        return T.predict_target(artifacts[display_spec[1]], feat)
    total = np.zeros(len(feat))
    for comp, mult in display_spec[1].items():
        total = total + mult * T.predict_target(artifacts[comp], feat)
    return total * opp_pred


def _b2(base, prefix, display_spec):
    g = base[f"{prefix}std_G"].replace(0, np.nan)
    kind = display_spec[0]
    if kind in ("opp", "direct"):
        return (base[f"{prefix}std_{display_spec[1]}"] / g).values
    tot = np.zeros(len(base))
    for comp, mult in display_spec[1].items():
        tot = tot + mult * base[f"{prefix}std_{comp}"].fillna(0).values
    return (tot / g.values)


def _b3(base, denom, display_spec):
    pg = base[f"marcel_{denom}pg"]
    kind = display_spec[0]
    if kind in ("opp", "direct"):
        col = display_spec[1]
        if col == denom:  # PA / BF opportunity itself
            return pg.values
        return (base[f"marcel_{col}"] * pg).values
    tot = np.zeros(len(base))
    for comp, mult in display_spec[1].items():
        tot = tot + mult * base[f"marcel_{comp}"].fillna(0).values
    return (tot * pg.values)


def evaluate_role(role, history, ctx, uni) -> tuple[pd.DataFrame, dict]:
    display = BAT_DISPLAY if role == "bat" else PIT_DISPLAY
    targets = M.BAT_TARGETS if role == "bat" else M.PIT_TARGETS
    opp_key = "PA" if role == "bat" else "p_BF"

    feat = T.build_features(history, role, ctx, uni)
    counts = T._counts_frame(history, role)
    fcols = M.feature_columns(feat)

    feat_train = feat[feat["season"].isin(TRAIN)].reset_index(drop=True)
    feat_test = feat[feat["season"] == TEST].reset_index(drop=True)
    logger.warning("[%s] train rows %d, test rows %d", role, len(feat_train), len(feat_test))

    artifacts = {t: T.fit_target(feat_train, counts, t, spec,
                                 M.target_feature_cols(t, fcols))
                 for t, spec in targets.items()}

    base, prefix, denom, league = _asof_baseline_frame(history, role)
    base_test = feat_test[["personId", "gamePk"]].merge(base, on=["personId", "gamePk"], how="left")
    actual = feat_test[["personId", "gamePk"]].merge(counts, on=["personId", "gamePk"], how="left")

    opp_pred = T.predict_target(artifacts[opp_key], feat_test)

    # League constant B1 per stat = train mean of the per-game count.
    train_counts = counts.merge(feat_train[["personId", "gamePk"]], on=["personId", "gamePk"])
    rows = []
    calib = {}
    for stat, spec in display.items():
        y = _actual_count(actual, spec)
        model_pred = _model_count(role, feat_test, artifacts, spec, opp_pred)
        b1 = np.full(len(feat_test), _train_mean_count(train_counts, spec))
        b2 = _b2(base_test, prefix, spec)
        b3 = _b3(base_test, denom, spec)
        mask = ~np.isnan(y)
        y, mp, b1, b2, b3 = y[mask], model_pred[mask], b1[mask], b2[mask], b3[mask]
        b2 = np.where(np.isnan(b2), b1, b2)
        b3 = np.where(np.isnan(b3), b1, b3)
        rows.append({
            "stat": stat, "n": int(mask.sum()),
            "model_MAE": T.mae(y, mp), "b1_MAE": T.mae(y, b1),
            "b2_MAE": T.mae(y, b2), "b3_MAE": T.mae(y, b3),
            "model_dev": T.poisson_deviance(y, mp), "b2_dev": T.poisson_deviance(y, b2),
            "b3_dev": T.poisson_deviance(y, b3),
        })
        calib[stat] = T.decile_calibration(y, mp)
    return pd.DataFrame(rows), calib


def _actual_count(actual, spec):
    kind = spec[0]
    if kind in ("opp", "direct"):
        return actual[spec[1]].astype(float).values
    tot = np.zeros(len(actual))
    for comp, mult in spec[1].items():
        tot = tot + mult * actual[comp].fillna(0).values
    return tot


def _train_mean_count(train_counts, spec):
    kind = spec[0]
    if kind in ("opp", "direct"):
        return float(train_counts[spec[1]].mean())
    tot = np.zeros(len(train_counts))
    for comp, mult in spec[1].items():
        tot = tot + mult * train_counts[comp].fillna(0).values
    return float(np.mean(tot))


def _verdict(r):
    beats_b2 = r["model_MAE"] <= r["b2_MAE"] and r["model_dev"] <= r["b2_dev"]
    beats_b3 = r["model_MAE"] <= r["b3_MAE"] and r["model_dev"] <= r["b3_dev"]
    if beats_b2 and beats_b3:
        return "PASS", "beats season-average and Marcel baselines"
    if r["model_dev"] <= min(r["b2_dev"], r["b3_dev"]):
        return "MARGINAL", "better calibrated (deviance) but MAE near baseline"
    return "WEAK", "no better than a season average; show with caveat or drop"


def write_report(bat_tbl, pit_tbl):
    lines = ["# Model report (v1, walk-forward)", ""]
    lines.append(f"Train seasons {TRAIN}, test season {TEST}. Every stat is "
                 "evaluated at the per-game count the site displays, against "
                 "three point-in-time baselines: league average (B1), the "
                 "player's season-to-date average excluding the game (B2), and "
                 "a Marcel-style rate times expected opportunities (B3).")
    lines += ["", "## Plain-English summary", ""]
    for role_name, tbl in [("Batting", bat_tbl), ("Pitching", pit_tbl)]:
        lines.append(f"**{role_name}**")
        for _, r in tbl.iterrows():
            v, why = _verdict(r)
            lines.append(f"- `{r['stat']}` [{v}]: {why} "
                         f"(model MAE {r['model_MAE']:.3f} vs season-avg {r['b2_MAE']:.3f}, "
                         f"Marcel {r['b3_MAE']:.3f}).")
        lines.append("")
    for role_name, tbl in [("Batting", bat_tbl), ("Pitching", pit_tbl)]:
        lines += [f"## {role_name} detail", "", tbl.round(4).to_markdown(index=False), ""]
    out = Path(__file__).resolve().parent.parent / "docs" / "model_report.md"
    out.write_text("\n".join(lines))
    logger.warning("wrote %s", out)


def main():
    hist = F.load_gamelogs(PRIOR + TRAIN + [2024, TEST])
    if hist.empty:
        raise SystemExit("no gamelogs; run build_training_backfill.py first")
    hist = F.attach_catchers(hist)
    uni = _universe()
    hist["pitchHand"] = hist["personId"].map(dict(zip(uni["personId"], uni["pitchHand"])))
    ctx = F.Context(CTX_YEARS)
    bat_tbl, _ = evaluate_role("bat", hist, ctx, uni)
    pit_tbl, _ = evaluate_role("pit", hist, ctx, uni)
    write_report(bat_tbl, pit_tbl)
    print(bat_tbl.to_string(index=False))
    print(pit_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
