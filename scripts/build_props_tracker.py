"""Grade previously-pasted PrizePicks lines against what actually happened and
append to one committed file: cache/props_history_v1.parquet.

This is the "model vs the board" scoreboard, kept for fun. The working line
files (pp_lines_{date}_v1.json) are a transient editing buffer that "Clear all"
wipes, so the graded record lives in its own append-only history that nothing
in the UI deletes. Run the day AFTER a slate, once the box scores are in.

A line can only be graded if the lines for that date were still saved when this
ran, so the record grows at the pace of the pasting habit, not automatically.

Grading notes (PrizePicks specifics, not generic over/under):
  - An exact landing on the line is NOT a push. It lowers the payout tier, so
    it gets its own "exact" bucket rather than being dropped or counted a win.
  - Only lines whose model lean is a side the board actually offers count toward
    the hit rate (props.compare's Playable flag); the rest are recorded with
    graded=False so the tally is not inflated by picks nobody could make.

Usage:
  python scripts/build_props_tracker.py 2026-07-17 2026-07-18 ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, props, store  # noqa: E402
from mlblib.cache import logger  # noqa: E402


def _outcome(actual: float, line: float) -> str:
    if actual > line:
        return "over"
    if actual < line:
        return "under"
    return "exact"


def score_date(date: str) -> pd.DataFrame:
    """One row per posted line that has a settled actual, for `date`."""
    lines = props.load_lines(date)
    if lines is None or lines.empty:
        logger.warning("%s: no saved PrizePicks lines", date)
        return pd.DataFrame()
    preds = store.load_predictions(date)
    if preds is None or preds.empty:
        logger.warning("%s: no predictions file", date)
        return pd.DataFrame()
    actuals = store.load_actuals(date)
    if actuals is None or actuals.empty:
        logger.warning("%s: no actuals yet (game not scored)", date)
        return pd.DataFrame()

    table, _meta = props.compare(lines, preds, actuals=actuals)
    if table.empty:
        logger.warning("%s: no lines matched a projected stat", date)
        return pd.DataFrame()

    rows = []
    for _, r in table.iterrows():
        act = r.get("Actual")
        if act is None or act != act:      # None or NaN -> did not play / unscored
            continue
        line = float(r["Line"])
        actual = float(act)
        result = _outcome(actual, line)
        lean = str(r.get("Lean") or "")
        playable = bool(r.get("Playable", True))
        # A lean only counts when it named a side AND that side was offered.
        graded = playable and lean in ("Over", "Under") and result != "exact"
        rows.append({
            "date": date,
            "player": r["Player"],
            "stat": r["Stat"],
            "model": float(r["Model"]),
            "line": line,
            "edge": float(r["Edge"]),
            "lean": lean,
            "odds_type": str(r.get("OddsType") or ""),
            "direction": str(r.get("Direction") or ""),
            "playable": playable,
            "actual": actual,
            "result": result,
            "graded": graded,
            "model_right": bool(graded and lean.lower() == result),
        })
    return pd.DataFrame(rows)


def main(argv):
    if not argv:
        raise SystemExit("pass one or more dates, e.g. 2026-07-17")
    path = cache.dc_path("props_history_v1.parquet")
    existing = cache.read_parquet_or_none(path)
    frames = [existing] if existing is not None else []
    for date in argv:
        scored = score_date(date)
        if not scored.empty:
            frames.append(scored)
            n_ok = int(scored["graded"].sum())
            logger.warning("%s: graded %d of %d posted line(s)", date, n_ok, len(scored))
    if not frames:
        logger.warning("nothing graded")
        return
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["date", "player", "stat"], keep="last")
    cache.atomic_to_parquet(combined, path)
    logger.warning("wrote %s (%d rows total)", path.name, len(combined))


if __name__ == "__main__":
    main(sys.argv[1:])
