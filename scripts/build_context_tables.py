"""Build the season-level context tables from Baseball Savant.

Writes committed parquets (small, one row per park or catcher per year):
  cache/park_factors_{year}_v1.parquet     (30 parks x per-stat index, 100=neutral)
  cache/catcher_framing_{year}_v1.parquet  (framing run values per catcher)
  cache/catcher_throwing_{year}_v1.parquet (throwing metrics per catcher)

Years 2018-2026 (training rows for season Y read the Y-1 file, so history is
needed; see mlblib/features.py). Refresh weekly during the season for the
current year; past years are immutable.

Savant requires a browser User-Agent (handled in mlblib.fetch). Park factors
come from the leaderboard's embedded `var data` JSON (csv=true does NOT work
there); the two catcher leaderboards do support csv=true.

Usage:
  python scripts/build_context_tables.py            # all years 2018-2026
  python scripts/build_context_tables.py 2024 2025  # specific years
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
_PF_INDEX_COLS = [
    "index_runs", "index_hr", "index_1b", "index_2b", "index_3b",
    "index_bb", "index_so", "index_woba", "index_obp",
]


def build_park_factors(year: int) -> None:
    frames = []
    for bat_side, tag in [("", "all"), ("L", "L"), ("R", "R")]:
        arr = fetch.get_park_factors_raw(year, bat_side)
        time.sleep(0.5)
        if not arr:
            logger.warning("park factors %s (%s): no data", year, tag or "all")
            continue
        rows = []
        for r in arr:
            row = {
                "year": int(r.get("key_year", year)),
                "venue_id": int(r["venue_id"]),
                "team": r.get("name_display_club"),
                "bat_side": tag,
                "n_pa": _to_int(r.get("n_pa")),
            }
            for col in _PF_INDEX_COLS:
                row[col] = _to_num(r.get(col))
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    out = cache.dc_path(f"park_factors_{year}_v1.parquet")
    cache.atomic_to_parquet(df, out)
    logger.warning("wrote %s (%d rows, %d parks)", out.name, len(df),
                   df["venue_id"].nunique())


def build_catcher_csv(leaderboard: str, short: str, year: int) -> None:
    csv_text = fetch.get_savant_csv(leaderboard, year)
    time.sleep(0.5)
    if not csv_text or "," not in csv_text:
        logger.warning("%s %s: no CSV", short, year)
        return
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df["season"] = year  # the CSV carries no year column
    out = cache.dc_path(f"catcher_{short}_{year}_v1.parquet")
    cache.atomic_to_parquet(df, out)
    logger.warning("wrote %s (%d catchers)", out.name, len(df))


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main(argv: list[str]) -> None:
    years = [int(a) for a in argv] if argv else YEARS
    for yr in years:
        build_park_factors(yr)
        build_catcher_csv("catcher-framing", "framing", yr)
        build_catcher_csv("catcher-throwing", "throwing", yr)


if __name__ == "__main__":
    main(sys.argv[1:])
