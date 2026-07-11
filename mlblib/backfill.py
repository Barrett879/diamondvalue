"""Reusable backfill core, shared by scripts/build_training_backfill.py (full
history) and scripts/build_daily_predictions.py (append the current season
through today). Keeps ONE implementation of "schedule -> Final games -> parsed
gamelog rows" so training and inference stay on identical schemas.
"""
from __future__ import annotations

import time

import pandas as pd

from . import cache, fetch
from .cache import logger
from .parse import parse_boxscore

THROTTLE_SEC = 0.34  # ~3 req/s on real network fetches


def season_window(season: int) -> tuple[str, str]:
    return f"{season}-03-01", f"{season}-11-15"


def handedness_maps(years=range(2019, 2027)) -> tuple[dict, dict]:
    pitch, bat = {}, {}
    for yr in years:
        pu = fetch.get_player_universe(yr)
        if pu.empty:
            continue
        for _, r in pu.iterrows():
            pid = r["personId"]
            if pd.notna(r.get("pitchHand")):
                pitch[pid] = r["pitchHand"]
            if pd.notna(r.get("batSide")):
                bat[pid] = r["batSide"]
    return pitch, bat


def final_games(season: int, through_date: str | None = None) -> list[dict]:
    """Deduped, Final-only game meta for a season (optionally only games whose
    officialDate <= through_date). Deduped by gamePk keeping the Final entry.
    """
    start, end = season_window(season)
    if through_date:
        end = min(end, through_date)
    raw = fetch.get_schedule_range(start, end, game_type="R")
    by_pk: dict[int, dict] = {}
    for g in raw:
        state = (g.get("status") or {}).get("codedGameState")
        if state != "F":
            continue
        pk = g.get("gamePk")
        prev = by_pk.get(pk)
        if prev is None or (g.get("officialDate") or "") > (prev.get("officialDate") or ""):
            venue = g.get("venue") or {}
            by_pk[pk] = {
                "gamePk": pk,
                "officialDate": g.get("officialDate"),
                "gameDate": g.get("gameDate"),
                "gameNumber": g.get("gameNumber", 1),
                "venue_id": venue.get("id"),
                "dayNight": g.get("dayNight"),
            }
    games = list(by_pk.values())
    games.sort(key=lambda x: (x["officialDate"] or "", x["gameNumber"]))
    return games


def parse_games(games: list[dict], pitch_hand: dict, bat_side: dict,
                throttle: bool = True) -> pd.DataFrame:
    raw_dir = cache.CACHE_DIR / "raw_boxscores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for meta in games:
        pk = meta["gamePk"]
        need = not (raw_dir / f"{pk}.json").exists()
        raw = fetch.get_boxscore_raw(pk)
        if need and throttle:
            time.sleep(THROTTLE_SEC)
        if raw is None:
            continue
        rows.extend(parse_boxscore(raw, meta))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["batSide"] = df["personId"].map(bat_side)
    df["oppStarterHand"] = df["oppStarterId"].map(pitch_hand)
    return df


def append_season_through(season: int, through_date: str,
                          pitch_hand: dict, bat_side: dict) -> pd.DataFrame:
    """Ensure cache/gamelogs_{season}_v1.parquet covers all Final games with
    officialDate <= through_date, fetching+appending any missing ones. Returns
    the full season frame. Idempotent (raw boxscores are cached).
    """
    path = cache.dc_path(f"gamelogs_{season}_v1.parquet")
    existing = cache.read_parquet_or_none(path)
    have_pks = set(existing["gamePk"].unique()) if existing is not None else set()
    games = final_games(season, through_date=through_date)
    missing = [g for g in games if g["gamePk"] not in have_pks]
    if not missing:
        return existing if existing is not None else pd.DataFrame()
    logger.warning("season %s: appending %d new games through %s",
                   season, len(missing), through_date)
    new = parse_games(missing, pitch_hand, bat_side)
    combined = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    combined = combined.drop_duplicates(subset=["gamePk", "personId", "side"])
    cache.atomic_to_parquet(combined, path)
    return combined
