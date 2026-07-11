"""Score past predictions against actual results and append to one committed
file: cache/accuracy_history_v1.parquet. The projection stays pure; this reads
the predictions the daily pipeline wrote and the actual box-score counts, and
records model-vs-actual (and a season-average baseline) per displayed stat.

Usage:
  python scripts/build_accuracy_tracker.py 2025-07-05 2025-07-06 ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, features as F, store  # noqa: E402
from mlblib.cache import logger  # noqa: E402

# (display stat, prediction column, actual gamelog column, actual scale)
# actual scale converts the gamelog column onto the prediction's units, e.g. the
# IP prediction is in innings while the gamelog stores outs (IP = outs / 3).
BAT_STATS = [("PA", "PA", "PA", 1.0), ("H", "H", "H", 1.0), ("HR", "HR", "HR", 1.0),
             ("SO", "SO", "SO", 1.0), ("BB", "BB", "BB", 1.0), ("TB", "TB", "TB", 1.0),
             ("R", "R", "R", 1.0), ("RBI", "RBI", "RBI", 1.0), ("SB", "SB", "SB", 1.0)]
PIT_STATS = [("K", "K", "p_K", 1.0), ("BB", "BB", "p_BB", 1.0), ("H", "H", "p_H", 1.0),
             ("ER", "ER", "p_ER", 1.0), ("IP", "IP", "p_outs", 1.0 / 3.0)]


def _season_todate_mean(logs, role):
    """Point-in-time season-to-date per-game mean per (personId, gamePk)."""
    counts = F.BAT_COUNTS if role == "bat" else F.PIT_COUNTS
    if role == "bat":
        sub = logs[logs["played"] & logs["is_batter"]].copy()
        prefix = "b_"
    else:
        sub = logs[logs["played"] & logs["is_pitcher"] & logs["is_sp"]].copy()
        prefix = "p_"
    sub = sub.sort_values(["personId", "gameDate", "gameNumber"]).reset_index(drop=True)
    sub = F._add_asof_aggregates(sub, counts, prefix=prefix)
    g = sub[f"{prefix}std_G"].replace(0, np.nan)
    out = sub[["personId", "gamePk"]].copy()
    for c in counts:
        out[f"b2_{c}"] = sub[f"{prefix}std_{c}"] / g
    return out


def score_date(date: str) -> pd.DataFrame:
    preds = store.load_predictions(date)
    if preds is None or preds.empty:
        logger.warning("%s: no predictions file", date)
        return pd.DataFrame()
    season = int(date[:4])
    logs = F.load_gamelogs([season])
    if logs.empty:
        logger.warning("%s: no gamelogs for season %s", date, season)
        return pd.DataFrame()
    actual = logs[["personId", "gamePk"] + F.BAT_COUNTS + F.PIT_COUNTS].drop_duplicates(
        subset=["personId", "gamePk"])

    rows = []
    for role, statmap in (("bat", BAT_STATS), ("pit", PIT_STATS)):
        sub = preds[preds["role"] == role]
        if sub.empty:
            continue
        base = _season_todate_mean(logs, role)
        m = sub.merge(actual, on=["personId", "gamePk"], how="inner", suffixes=("", "_act"))
        m = m.merge(base, on=["personId", "gamePk"], how="left")
        for disp, pred_col, act_col, scale in statmap:
            if pred_col not in m.columns:
                continue
            pred = pd.to_numeric(m[pred_col], errors="coerce")
            act = pd.to_numeric(m[f"{act_col}_act"] if f"{act_col}_act" in m else m[act_col],
                                errors="coerce") * scale
            b2 = m.get(f"b2_{act_col}")
            if b2 is not None:
                b2 = b2 * scale
            for i in range(len(m)):
                a = act.iloc[i]
                p = pred.iloc[i]
                if pd.isna(a) or pd.isna(p):
                    continue
                rows.append({
                    "date": date, "personId": int(m["personId"].iloc[i]),
                    "fullName": m["fullName"].iloc[i], "role": role, "stat": disp,
                    "pred": float(p), "actual": float(a),
                    "abs_err_model": abs(float(p) - float(a)),
                    "abs_err_b2": (abs(float(b2.iloc[i]) - float(a))
                                   if b2 is not None and not pd.isna(b2.iloc[i]) else np.nan),
                })
    return pd.DataFrame(rows)


def main(argv):
    if not argv:
        raise SystemExit("pass one or more dates, e.g. 2025-07-05 2025-07-06")
    path = cache.dc_path("accuracy_history_v1.parquet")
    existing = cache.read_parquet_or_none(path)
    frames = [existing] if existing is not None else []
    for date in argv:
        scored = score_date(date)
        if not scored.empty:
            frames.append(scored)
            logger.warning("%s: scored %d rows", date, len(scored))
    if not frames:
        logger.warning("nothing scored")
        return
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["date", "personId", "stat"], keep="last")
    cache.atomic_to_parquet(combined, path)
    logger.warning("wrote %s (%d rows total)", path.name, len(combined))


if __name__ == "__main__":
    main(sys.argv[1:])
