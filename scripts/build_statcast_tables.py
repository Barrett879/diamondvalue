"""Build season-level Statcast quality-of-contact tables from Baseball Savant.

Five committed parquets per year (small, one row per player):
  cache/statcast_xstats_bat_{year}_v1.parquet  (est_ba/est_slg/est_woba)
  cache/statcast_xstats_pit_{year}_v1.parquet  (same, allowed by the pitcher)
  cache/statcast_ev_bat_{year}_v1.parquet      (avg EV, barrel %, hard-hit %)
  cache/statcast_ev_pit_{year}_v1.parquet      (same, against the pitcher)
  cache/sprint_speed_{year}_v1.parquet         (sprint speed ft/s)

Years 2018-2026. Training rows for season Y use the Y-1 file (features.py's
prior-year rule); the current year's file only matters for NEXT season, but is
fetched for completeness. Year filtering VERIFIED per leaderboard (Judge's
avg_hit_speed 97.6 in 2023 vs 96.2 in 2024; Ohtani sprint 27.8 vs 28.1) -- the
catcher leaderboards' silent year-ignore does not apply to these.

Usage:
  python scripts/build_statcast_tables.py            # all years
  python scripts/build_statcast_tables.py 2024 2025  # specific years
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, fetch  # noqa: E402
from mlblib.cache import logger  # noqa: E402

YEARS = list(range(2018, 2027))

XSTATS_KEEP = ["player_id", "pa", "est_ba", "est_slg", "est_woba"]
EV_KEEP = ["player_id", "attempts", "avg_hit_speed", "brl_percent", "ev95percent"]
SPRINT_KEEP = ["player_id", "sprint_speed", "hp_to_1b"]


def _fetch_csv(path: str, params: dict) -> pd.DataFrame | None:
    text = fetch._http_text(f"{fetch.SAVANT_BASE}/leaderboard/{path}", params,
                            have_stale=False, session=fetch._savant)
    if not text or "," not in text[:200]:
        return None
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def build_year(year: int) -> None:
    jobs = [
        ("expected_statistics", {"type": "batter", "year": year, "position": "",
                                 "team": "", "min": "25", "csv": "true"},
         XSTATS_KEEP, f"statcast_xstats_bat_{year}_v1.parquet"),
        ("expected_statistics", {"type": "pitcher", "year": year, "position": "",
                                 "team": "", "min": "25", "csv": "true"},
         XSTATS_KEEP, f"statcast_xstats_pit_{year}_v1.parquet"),
        ("statcast", {"type": "batter", "year": year, "min": "25", "csv": "true"},
         EV_KEEP, f"statcast_ev_bat_{year}_v1.parquet"),
        ("statcast", {"type": "pitcher", "year": year, "min": "25", "csv": "true"},
         EV_KEEP, f"statcast_ev_pit_{year}_v1.parquet"),
        ("sprint_speed", {"year": year, "min": "10", "csv": "true"},
         SPRINT_KEEP, f"sprint_speed_{year}_v1.parquet"),
    ]
    for path, params, keep, outname in jobs:
        df = _fetch_csv(path, params)
        time.sleep(0.5)
        if df is None or df.empty:
            logger.warning("%s %s: no data", outname, year)
            continue
        cols = [c for c in keep if c in df.columns]
        out = df[cols].copy()
        out["season"] = year  # the CSVs carry no reliable year column
        cache.atomic_to_parquet(out, cache.dc_path(outname))
        logger.warning("wrote %s (%d players)", outname, len(out))


def main(argv: list[str]) -> None:
    years = [int(a) for a in argv] if argv else YEARS
    for yr in years:
        build_year(yr)


if __name__ == "__main__":
    main(sys.argv[1:])
