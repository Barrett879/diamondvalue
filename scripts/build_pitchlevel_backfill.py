"""Pitch-level Statcast backfill, reduced to two small per-season tables.

Pulls statcast_search details CSVs in 3-day chunks (the endpoint silently
truncates at exactly 25,000 rows per request, so every chunk asserts it came
back under the cap) and reduces each season to:

  cache/tto_pit_{year}_v1.parquet   per (pitcher, times-through-order 1/2/3+):
      pa, so, whiffs, pitches  -> each starter's decay profile
  cache/fbvelo_{year}_v1.parquet    per (pitcher, game_pk, game_date):
      avg fastball velo (FF/SI) and pitch count -> rolling velo trend, the
      fatigue/injury canary

Raw pitch rows are never kept (a season is ~500 MB raw; the reductions are a
few hundred KB). Chunks cache their reduced form under cache/raw_statcast/ so
the pull is resumable and re-runs are free.

Usage: python scripts/build_pitchlevel_backfill.py 2020 2021 ...  (or 'all')
"""
from __future__ import annotations

import datetime as dt
import io
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, fetch  # noqa: E402
from mlblib.cache import logger  # noqa: E402

ALL_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
CHUNK_DAYS = 3
ROW_CAP = 25000
KEEP = ["game_pk", "game_date", "pitcher", "n_thruorder_pitcher",
        "pitch_type", "release_speed", "description", "events"]
_SWING_MISS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}


def _season_days(season: int):
    start = dt.date(season, 3, 15)
    end = dt.date(season, 11, 5)
    cur = start
    while cur <= end:
        yield cur, min(cur + dt.timedelta(days=CHUNK_DAYS - 1), end)
        cur += dt.timedelta(days=CHUNK_DAYS)


def _fetch_chunk(d0: dt.date, d1: dt.date) -> pd.DataFrame | None:
    params = {
        "all": "true", "player_type": "pitcher", "type": "details",
        "game_date_gt": d0.isoformat(), "game_date_lt": d1.isoformat(),
        "hfSea": f"{d0.year}|", "hfGT": "R|",
    }
    text = fetch._http_text(f"{fetch.SAVANT_BASE}/statcast_search/csv", params,
                            have_stale=False, session=fetch._savant)
    if not text or "," not in text[:200]:
        return None
    df = pd.read_csv(io.StringIO(text), usecols=lambda c: c in KEEP,
                     low_memory=False)
    assert len(df) < ROW_CAP, (
        f"chunk {d0}..{d1} hit the 25k silent-truncation cap ({len(df)} rows); "
        "shrink CHUNK_DAYS")
    return df


def _reduce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_pa_end"] = df["events"].notna() & (df["events"] != "")
    df["is_so"] = df["events"].isin(["strikeout", "strikeout_double_play"])
    df["is_whiff"] = df["description"].isin(_SWING_MISS)
    df["is_fb"] = df["pitch_type"].isin(["FF", "SI"])
    df["tto"] = df["n_thruorder_pitcher"].clip(upper=3)
    return df


def build_season(season: int) -> None:
    raw_dir = cache.CACHE_DIR / "raw_statcast"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for d0, d1 in _season_days(season):
        part_path = raw_dir / f"red_{d0.isoformat()}_{d1.isoformat()}.parquet"
        if part_path.exists():
            parts.append(pd.read_parquet(part_path))
            continue
        df = _fetch_chunk(d0, d1)
        time.sleep(2.0)
        if df is None or df.empty:
            continue
        red = _reduce(df)
        cache.atomic_to_parquet(red, part_path)
        parts.append(red)
        logger.warning("season %s chunk %s..%s: %d pitches", season, d0, d1, len(red))
    if not parts:
        logger.warning("season %s: nothing fetched", season)
        return
    allp = pd.concat(parts, ignore_index=True)

    # (a) TTO profile per pitcher.
    tto = (allp.groupby(["pitcher", "tto"])
           .agg(pitches=("pitcher", "size"),
                pa=("is_pa_end", "sum"), so=("is_so", "sum"),
                whiffs=("is_whiff", "sum"))
           .reset_index().rename(columns={"pitcher": "player_id"}))
    tto["season"] = season
    cache.atomic_to_parquet(tto, cache.dc_path(f"tto_pit_{season}_v1.parquet"))

    # (b) Per-start fastball velocity.
    fb = allp[allp["is_fb"] & allp["release_speed"].notna()]
    velo = (fb.groupby(["pitcher", "game_pk", "game_date"])
            .agg(ff_velo=("release_speed", "mean"), n_ff=("release_speed", "size"))
            .reset_index().rename(columns={"pitcher": "player_id"}))
    velo["season"] = season
    cache.atomic_to_parquet(velo, cache.dc_path(f"fbvelo_{season}_v1.parquet"))
    logger.warning("season %s: wrote tto_pit (%d rows) + fbvelo (%d starts)",
                   season, len(tto), len(velo))


def main(argv: list[str]) -> None:
    seasons = ALL_SEASONS if (not argv or argv[0] == "all") else [int(a) for a in argv]
    for s in seasons:
        build_season(s)


if __name__ == "__main__":
    main(sys.argv[1:])
