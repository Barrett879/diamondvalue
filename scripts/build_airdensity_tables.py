"""Build the per-game air-density carry table (round 7).

For every historical game: the batted-ball carry index at first pitch, from
the Open-Meteo ARCHIVE (temperature, relative humidity, station surface
pressure) at the venue's coordinates. Batched by (venue, season) -- one
archive call returns the whole season's hourly reanalysis, cached raw under
cache/raw_airdensity/ so re-deriving carry_index (e.g. a formula change) is
free. Games at venues without coordinates (~4%, spring-training / neutral
sites) get no row -> NaN feature -> HistGB routes natively.

Writes cache/game_airdensity_v1.parquet (per gamePk: carry_index + provenance
columns temp_f_arc / rh / press for debugging).

Usage: python scripts/build_airdensity_tables.py           # all seasons
       python scripts/build_airdensity_tables.py 2024 2025  # subset
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, features as F, fetch  # noqa: E402
from mlblib.cache import logger  # noqa: E402

ALL_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def _game_meta(season: int) -> pd.DataFrame:
    gl = cache.read_parquet_or_none(cache.dc_path(f"gamelogs_{season}_v1.parquet"))
    if gl is None:
        return pd.DataFrame()
    g = (gl.groupby("gamePk")
         .agg(officialDate=("officialDate", "first"),
              gameDate=("gameDate", "first"),
              venue_id=("venue_id", "first"))
         .reset_index())
    g["season"] = season
    return g


def _archive_raw(venue_id, lat, lon, season) -> dict | None:
    """Cached per venue-season hourly reanalysis (raw), so formula re-derives
    are free. Returns {hour_iso: (temp, rh, press)}."""
    path = cache.CACHE_DIR / "raw_airdensity" / f"arc_{int(venue_id)}_{season}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return {t: (a, b, c) for t, a, b, c in
                zip(df["time"], df["temp"], df["rh"], df["press"])}
    got = fetch.get_weather_archive(lat, lon, f"{season}-03-01", f"{season}-11-15")
    time.sleep(1.0)
    if not got:
        return None
    df = pd.DataFrame([(t, a, b, c) for t, (a, b, c) in got.items()],
                      columns=["time", "temp", "rh", "press"])
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.atomic_to_parquet(df, path)
    return got


def build(seasons: list[int]) -> None:
    venues = cache.read_parquet_or_none(cache.dc_path("venues_v1.parquet"))
    vmap = {int(r.venue_id): (r.lat, r.lon) for r in venues.itertuples()
            if pd.notna(r.lat) and pd.notna(r.lon)}
    elev = {int(r.venue_id): r.elevation for r in venues.itertuples()
            if pd.notna(r.elevation)}
    rooftype = {int(r.venue_id): (r.roofType or "Open") for r in venues.itertuples()}
    # Roof handling keys ONLY on roofType + the park's historical closed share,
    # NEVER the per-game actual roof state -- because inference cannot know the
    # roof pregame. This keeps the carry_index formula byte-identical between
    # training (archive weather) and inference (forecast weather):
    #   Open        -> outdoor
    #   Dome (fixed) -> indoor constant at park elevation
    #   Retractable  -> closed_share*indoor + (1-closed_share)*outdoor  (blend)
    gw = cache.read_parquet_or_none(cache.dc_path("game_weather_v1.parquet"))
    closed_share = {}
    if gw is not None:
        vpk = pd.concat([_game_meta(s)[["gamePk", "venue_id"]] for s in ALL_SEASONS
                         if not _game_meta(s).empty], ignore_index=True)
        vg = vpk.merge(gw[["gamePk", "roof_closed"]], on="gamePk", how="inner")
        closed_share = vg.groupby("venue_id")["roof_closed"].mean().to_dict()

    rows = []
    for season in seasons:
        meta = _game_meta(season)
        if meta.empty:
            continue
        # First-pitch UTC hour key (floor to the hour = the shipped inference
        # convention in _game_environment; archive is hourly).
        meta["hrkey"] = (pd.to_datetime(meta["gameDate"], utc=True, errors="coerce")
                         .dt.strftime("%Y-%m-%dT%H:00"))
        for vid, sub in meta.groupby("venue_id"):
            if pd.isna(vid) or int(vid) not in vmap:
                continue
            vid = int(vid)
            lat, lon = vmap[vid]
            arc = _archive_raw(vid, lat, lon, season)
            if not arc:
                logger.warning("no archive for venue %s season %s", vid, season)
                continue
            rt = rooftype.get(vid, "Open")
            indoor_ci = (float(F.indoor_carry_index(elev[vid])[0])
                         if vid in elev else np.nan)
            share = float(closed_share.get(vid, 0.0))
            for r in sub.itertuples():
                got = arc.get(r.hrkey)
                out_ci = (float(F.air_carry_index(got[0], got[1], got[2])[0])
                          if got is not None else np.nan)
                ci = F.blend_carry(rt, indoor_ci, out_ci, share)
                if ci != ci:  # NaN
                    continue
                rows.append({"gamePk": r.gamePk, "venue_id": vid,
                             "carry_index": ci})
        logger.warning("season %s: %d games priced so far", season, len(rows))
    df = pd.DataFrame(rows)
    cache.atomic_to_parquet(df, cache.dc_path("game_airdensity_v1.parquet"))
    ci = df["carry_index"].dropna()
    logger.warning("wrote game_airdensity_v1.parquet (%d games, carry_index "
                   "mean %.3f, p01 %.3f, p99 %.3f, max %.3f)",
                   len(df), ci.mean(), ci.quantile(.01), ci.quantile(.99), ci.max())


if __name__ == "__main__":
    args = sys.argv[1:]
    build([int(a) for a in args] if args else ALL_SEASONS)
