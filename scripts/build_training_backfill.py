"""Build the historical per-game log table from the MLB Stats API.

One parquet per season: cache/gamelogs_{season}_v1.parquet, one row per
(gamePk, personId, side), using the SAME parser the daily pipeline uses.

Resumable: every raw boxscore JSON is cached under cache/raw_boxscores/, so a
re-run only hits games it has not seen. Only ever needs to run once per season.

TRAP handled here (verified on 2024): the schedule range contains
postponed/rescheduled DUPLICATE gamePks with the wrong date. We dedupe by
gamePk, keep only Final games (codedGameState == "F"), and take game_date from
that Final entry's officialDate. The script asserts zero duplicate gamePks and
zero non-Final rows before writing.

Usage:
  python scripts/build_training_backfill.py 2024            # one season
  python scripts/build_training_backfill.py 2019 2020 2021  # several
  python scripts/build_training_backfill.py all             # 2019..2025
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, fetch  # noqa: E402
from mlblib.cache import logger  # noqa: E402
from mlblib.parse import parse_boxscore  # noqa: E402

ALL_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
THROTTLE_SEC = 0.34  # ~3 requests/second on real network fetches (polite)


def _season_window(season: int) -> tuple[str, str]:
    # Wide window; gameType=R restricts to the regular season regardless.
    return f"{season}-03-01", f"{season}-11-15"


def _handedness_maps() -> tuple[dict, dict]:
    """personId -> pitchHand and personId -> batSide, unioned across the player
    universes 2019-2026. Handedness is static, so any season's universe works.
    """
    pitch, bat = {}, {}
    for yr in range(2019, 2027):
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


def _final_games(season: int) -> list[dict]:
    """Deduped, Final-only game entries for a season, each with the fields the
    parser's meta needs. Deduped by gamePk keeping the Final entry.
    """
    start, end = _season_window(season)
    raw = fetch.get_schedule_range(start, end, game_type="R")
    by_pk: dict[int, dict] = {}
    for g in raw:
        pk = g.get("gamePk")
        state = (g.get("status") or {}).get("codedGameState")
        if state != "F":
            continue  # drop postponed/scheduled/cancelled duplicates
        # If the same Final gamePk somehow appears twice, the later officialDate
        # is the one actually played; keep the max.
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


def build_season(season: int, pitch_hand: dict, bat_side: dict) -> Path:
    games = _final_games(season)
    logger.warning("season %s: %d final games", season, len(games))
    raw_dir = cache.CACHE_DIR / "raw_boxscores"

    all_rows: list[dict] = []
    fetched = 0
    for i, meta in enumerate(games):
        pk = meta["gamePk"]
        need_network = not (raw_dir / f"{pk}.json").exists()
        raw = fetch.get_boxscore_raw(pk)
        if need_network:
            fetched += 1
            time.sleep(THROTTLE_SEC)
        if raw is None:
            logger.warning("season %s: boxscore fetch failed for %s (skipped)", season, pk)
            continue
        all_rows.extend(parse_boxscore(raw, meta))
        if (i + 1) % 250 == 0:
            logger.warning("season %s: %d/%d games parsed (%d fetched)",
                           season, i + 1, len(games), fetched)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit(f"season {season}: no rows parsed")

    # Enrich with handedness (static player attributes; point-in-time safe).
    df["batSide"] = df["personId"].map(bat_side)
    df["oppStarterHand"] = df["oppStarterId"].map(pitch_hand)

    # Acceptance assertions.
    dup = df.duplicated(subset=["gamePk", "personId", "side"]).sum()
    assert dup == 0, f"season {season}: {dup} duplicate (gamePk,personId,side) rows"
    n_pk = df["gamePk"].nunique()
    assert n_pk == len(games), f"season {season}: {n_pk} gamePks in rows vs {len(games)} games"

    out = cache.dc_path(f"gamelogs_{season}_v1.parquet")
    cache.atomic_to_parquet(df, out)
    played = int(df["played"].sum())
    logger.warning(
        "season %s: wrote %s (%d rows, %d gamePks, %d played, %d fetched this run)",
        season, out.name, len(df), n_pk, played, fetched,
    )
    return out


def main(argv: list[str]) -> None:
    if not argv or argv[0] == "all":
        seasons = ALL_SEASONS
    else:
        seasons = [int(a) for a in argv]
    logger.warning("building handedness maps from player universes ...")
    pitch_hand, bat_side = _handedness_maps()
    logger.warning("handedness: %d pitchers, %d batters", len(pitch_hand), len(bat_side))
    for s in seasons:
        build_season(s, pitch_hand, bat_side)


if __name__ == "__main__":
    main(sys.argv[1:])
