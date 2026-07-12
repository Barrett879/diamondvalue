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
# Round-5 re-pull: the columns the v1 pull discarded that the next feature
# round needs (catcher framing v2, batter-vs-velocity, true per-PA platoon,
# release-point/spin drift, process-based batter form).
KEEP2 = KEEP + ["batter", "fielder_2", "stand", "p_throws",
                "release_pos_x", "release_pos_z", "release_spin_rate",
                "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
                "estimated_woba_using_speedangle"]
_SWING_MISS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}


def _season_days(season: int):
    start = dt.date(season, 3, 15)
    end = dt.date(season, 11, 5)
    cur = start
    while cur <= end:
        yield cur, min(cur + dt.timedelta(days=CHUNK_DAYS - 1), end)
        cur += dt.timedelta(days=CHUNK_DAYS)


def _fetch_chunk(d0: dt.date, d1: dt.date, keep: list[str] | None = None) -> pd.DataFrame | None:
    keep = keep or KEEP
    params = {
        "all": "true", "player_type": "pitcher", "type": "details",
        "game_date_gt": d0.isoformat(), "game_date_lt": d1.isoformat(),
        "hfSea": f"{d0.year}|", "hfGT": "R|",
    }
    text = fetch._http_text(f"{fetch.SAVANT_BASE}/statcast_search/csv", params,
                            have_stale=False, session=fetch._savant)
    if not text or "," not in text[:200]:
        return None
    df = pd.read_csv(io.StringIO(text), usecols=lambda c: c in keep,
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


def _merge_pergame(new: pd.DataFrame, path, key: list[str]) -> int:
    """Merge freshly reduced per-game rows into a committed parquet,
    replacing overlapping rows (keep='last' so a partial in-progress-game row
    fetched at the early run is replaced by the complete one next run)."""
    old = cache.read_parquet_or_none(path)
    combined = (pd.concat([old, new], ignore_index=True)
                if old is not None else new)
    combined = combined.drop_duplicates(subset=key, keep="last")
    cache.atomic_to_parquet(combined, path)
    return len(combined)


def update_current_velo(season: int, days: int = 8) -> None:
    """Rolling in-season update of the three per-game pitch-level tables
    (fbvelo, relspin2, batproc2) from ONE KEEP2 fetch of the recent window.

    Self-healing window: starts at the earliest of (today - days) and each
    table's own coverage edge, so an Actions outage or a bootstrap handoff
    gap is backfilled instead of becoming a permanent hole (capped at 30
    days). Stale-beats-empty: on a total fetch failure NO parquet is touched.
    """
    import datetime as _dt

    from scripts.build_pitchlevel_v2_tables import (_derive, reduce_batproc,
                                                    reduce_relspin)

    end = _dt.date.today()
    start = end - _dt.timedelta(days=days)
    for stem in (f"fbvelo_{season}_v1", f"relspin2_{season}_v1",
                 f"batproc2_{season}_v1"):
        t = cache.read_parquet_or_none(cache.dc_path(f"{stem}.parquet"))
        if t is not None and len(t) and "game_date" in t.columns:
            edge = (pd.to_datetime(t["game_date"].astype(str).str[:10]).max()
                    .date() + _dt.timedelta(days=1))
            start = min(start, edge)
    start = max(start, end - _dt.timedelta(days=30), _dt.date(season, 3, 15))

    parts = []
    cur = start
    while cur <= end:
        d1 = min(cur + _dt.timedelta(days=CHUNK_DAYS - 1), end)
        df = _fetch_chunk(cur, d1, keep=KEEP2)
        time.sleep(2.0)
        if df is not None and not df.empty:
            parts.append(df)
        cur += _dt.timedelta(days=CHUNK_DAYS)
    if not parts:
        logger.warning("pitch-level update %s: nothing fetched (tables "
                       "left untouched)", season)
        return
    allp = pd.concat(parts, ignore_index=True)
    allp["game_date"] = allp["game_date"].astype(str).str[:10]

    red = _reduce(allp)
    fb = red[red["is_fb"] & red["release_speed"].notna()]
    velo = (fb.groupby(["pitcher", "game_pk", "game_date"])
            .agg(ff_velo=("release_speed", "mean"), n_ff=("release_speed", "size"))
            .reset_index().rename(columns={"pitcher": "player_id"}))
    velo["season"] = season
    n1 = _merge_pergame(velo, cache.dc_path(f"fbvelo_{season}_v1.parquet"),
                        ["player_id", "game_pk"])

    d2 = _derive(allp)
    n2 = _merge_pergame(reduce_relspin(d2, season),
                        cache.dc_path(f"relspin2_{season}_v1.parquet"),
                        ["player_id", "game_pk"])
    n3 = _merge_pergame(reduce_batproc(d2, season),
                        cache.dc_path(f"batproc2_{season}_v1.parquet"),
                        ["player_id", "game_pk"])
    logger.warning("pitch-level update %s from %s: fbvelo %d, relspin2 %d, "
                   "batproc2 %d rows", season, start, n1, n2, n3)


def fetch_v2_parts(season: int) -> None:
    """Round-5 re-pull: cache full-column (KEEP2) chunk parts as red2_*.parquet
    under cache/raw_statcast/. No reduction yet -- the v2 feature tables
    (framing, batter-vs-velo, platoon, release drift) are built from these in
    a later round. Resumable; skips parts already on disk.
    """
    # Only cache windows that are safely complete (Savant rows for very recent
    # days are partial/intraday): a clipped window cached under the full-window
    # filename would be a permanent silent hole. The rolling daily updater owns
    # the recent edge.
    safe_end = dt.date.today() - dt.timedelta(days=2)
    n = 0
    for d0, d1 in _season_days(season):
        if d1 > safe_end:
            break
        part_path = (cache.CACHE_DIR / "raw_statcast" /
                     f"red2_{d0.isoformat()}_{d1.isoformat()}.parquet")
        if part_path.exists():
            continue
        df = _fetch_chunk(d0, d1, keep=KEEP2)
        time.sleep(2.0)
        if df is None or df.empty:
            continue
        cache.atomic_to_parquet(df, part_path)
        n += 1
        if n % 10 == 0:
            logger.warning("v2 %s: through %s", season, d1)
    logger.warning("v2 %s: done (%d new parts)", season, n)


def main(argv: list[str]) -> None:
    if argv and argv[0] == "v2":
        rest = argv[1:]
        seasons = (ALL_SEASONS + [2026]) if (not rest or rest[0] == "all") \
            else [int(a) for a in rest]
        for s in seasons:
            fetch_v2_parts(s)
        return
    seasons = ALL_SEASONS if (not argv or argv[0] == "all") else [int(a) for a in argv]
    for s in seasons:
        build_season(s)


if __name__ == "__main__":
    main(sys.argv[1:])
