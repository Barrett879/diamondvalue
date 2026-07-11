"""Build pitch-arsenal and plate-discipline tables from Baseball Savant.

Four committed parquets per year:
  cache/arsenal_pit_{year}_v1.parquet  one row per (pitcher, pitch_type):
      pitch_usage, whiff_percent, k_percent, put_away, pitches, pa
  cache/arsenal_bat_{year}_v1.parquet  one row per (batter, pitch_type):
      the batter's performance AGAINST that pitch type (whiff_percent, ...)
  cache/disc_pit_{year}_v1.parquet     per-pitcher plate discipline
  cache/disc_bat_{year}_v1.parquet     per-batter plate discipline

Years 2018-2026, Y-1 rule at feature time. Year filtering VERIFIED live for
both endpoints (Gausman's FF whiff differs by year; discipline keys all
populate, zero nulls). Neither CSV carries a year column: stamp it.

Usage: python scripts/build_arsenal_tables.py [years...]
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

ARSENAL_KEEP = ["player_id", "pitch_type", "pitch_usage", "pitches", "pa",
                "whiff_percent", "k_percent", "put_away"]
DISC_SEL = ("k_percent,bb_percent,whiff_percent,swing_percent,"
            "oz_swing_percent,oz_contact_percent,iz_contact_percent,"
            "f_strike_percent")
DISC_KEEP = ["player_id", "k_percent", "bb_percent", "whiff_percent",
             "swing_percent", "oz_swing_percent", "oz_contact_percent",
             "iz_contact_percent", "f_strike_percent"]


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
        ("pitch-arsenal-stats",
         {"type": "pitcher", "pitchType": "", "year": year, "team": "",
          "min": "10", "csv": "true"},
         ARSENAL_KEEP, f"arsenal_pit_{year}_v1.parquet"),
        ("pitch-arsenal-stats",
         {"type": "batter", "pitchType": "", "year": year, "team": "",
          "min": "10", "csv": "true"},
         ARSENAL_KEEP, f"arsenal_bat_{year}_v1.parquet"),
        ("custom",
         {"year": year, "type": "pitcher", "min": "40", "selections": DISC_SEL,
          "chart": "false", "x": "k_percent", "y": "k_percent",
          "sort": "1", "sortDir": "desc", "csv": "true"},
         DISC_KEEP, f"disc_pit_{year}_v1.parquet"),
        ("custom",
         {"year": year, "type": "batter", "min": "40", "selections": DISC_SEL,
          "chart": "false", "x": "k_percent", "y": "k_percent",
          "sort": "1", "sortDir": "desc", "csv": "true"},
         DISC_KEEP, f"disc_bat_{year}_v1.parquet"),
    ]
    for path, params, keep, outname in jobs:
        df = _fetch_csv(path, params)
        time.sleep(0.6)
        if df is None or df.empty:
            logger.warning("%s: no data", outname)
            continue
        cols = [c for c in keep if c in df.columns]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            logger.warning("%s: missing cols %s", outname, missing)
        out = df[cols].copy()
        out["season"] = year
        cache.atomic_to_parquet(out, cache.dc_path(outname))
        logger.warning("wrote %s (%d rows)", outname, len(out))


def main(argv: list[str]) -> None:
    years = [int(a) for a in argv] if argv else YEARS
    for yr in years:
        build_year(yr)


if __name__ == "__main__":
    main(sys.argv[1:])
