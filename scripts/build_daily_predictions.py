"""Build the day's predictions offline.

For a target date: fetch the slate (probables + lineups), ensure the current
season's gamelogs are current, build point-in-time inference features for every
relevant player (posted or projected lineup, bench, starters), predict every
target, and write:
  cache/predictions_{YYYY_MM_DD}_m{VERSION}.parquet   (one row per player-game)
  cache/slate_pred_{YYYY_MM_DD}_v1.json               (game metadata + status)

The Streamlit pages read ONLY these files; the model never runs at request time.

Hardened like HoopsValue's build_player_hub.py: socket timeout + faulthandler
watchdog so a hung endpoint kills the process instead of stalling forever.

Usage:
  python scripts/build_daily_predictions.py                 # today
  python scripts/build_daily_predictions.py 2025-07-10      # a specific date
"""
from __future__ import annotations

import faulthandler
import hashlib
import socket
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

socket.setdefaulttimeout(30)
faulthandler.dump_traceback_later(1800, exit=True)  # 30-min watchdog

from mlblib import backfill, cache, features as F, fetch, model as M  # noqa: E402
from mlblib.cache import logger  # noqa: E402
from mlblib.util import today_iso  # noqa: E402

CTX_YEARS = list(range(2018, 2027))
BENCH_SLOT = 6  # neutral slot for conditional "if he starts" bench predictions


def _universe():
    frames = [cache.read_parquet_or_none(cache.dc_path(f"player_universe_{y}_v1.parquet"))
              for y in CTX_YEARS]
    frames = [f for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="personId", keep="last")


def _projected_lineup(history, team_id, before_date, n=10) -> list[tuple]:
    """Most common (personId, slot) over the team's last n games before the
    date, from starter slots. Returns up to 9 ordered (personId, slot) tuples.
    """
    sub = history[(history["teamId"] == team_id)
                  & (history["officialDate"] < before_date)
                  & history["is_batter"] & history["played"]
                  & history["is_starter_slot"] & history["slot"].notna()]
    if sub.empty:
        return []
    recent_pks = (sub[["gamePk", "gameDate"]].drop_duplicates()
                  .sort_values("gameDate").tail(n)["gamePk"])
    recent = sub[sub["gamePk"].isin(recent_pks)]
    used, lineup = set(), []
    for slot in range(1, 10):
        counts = Counter(recent[recent["slot"] == slot]["personId"])
        for pid, _ in counts.most_common():
            if pid not in used:
                lineup.append((pid, slot))
                used.add(pid)
                break
    return lineup


def _bat_target(pid, name, team_id, slot, is_home, g, opp, opp_hand, bat_side,
                status, is_bench):
    return {
        "personId": pid, "fullName": name, "teamId": team_id, "role": "bat",
        "gamePk": g["gamePk"], "gameDate": g["gameDate"], "gameNumber": g.get("gameNumber", 1),
        "season": int(g["officialDate"][:4]), "officialDate": g["officialDate"],
        "isHome": is_home, "slot": slot, "venue_id": g["venue_id"],
        "oppTeamId": opp["id"], "oppStarterId": (opp["probable"] or {}).get("id"),
        "oppStarterHand": opp_hand, "batSide": bat_side,
        "lineup_status": status, "is_bench": is_bench,
    }


def _pit_target(pid, name, team_id, is_home, g, opp, hand):
    return {
        "personId": pid, "fullName": name, "teamId": team_id, "role": "pit",
        "gamePk": g["gamePk"], "gameDate": g["gameDate"], "gameNumber": g.get("gameNumber", 1),
        "season": int(g["officialDate"][:4]), "officialDate": g["officialDate"],
        "isHome": is_home, "venue_id": g["venue_id"], "oppTeamId": opp["id"],
        "pitchHand": hand, "lineup_status": "confirmed" if pid else "projected",
    }


def build_targets(slate, history, uni):
    bat_map = dict(zip(uni["personId"], uni["batSide"]))
    hand_map = dict(zip(uni["personId"], uni["pitchHand"]))
    bat_rows, pit_rows, slate_meta = [], [], []
    for g in slate:
        gmeta = {"gamePk": g["gamePk"], "gameDate": g["gameDate"],
                 "officialDate": g["officialDate"], "gameNumber": g.get("gameNumber", 1),
                 "venue_id": g["venue_id"], "venue_name": g.get("venue_name"),
                 "away": g["away"]["abbr"] or g["away"]["name"],
                 "home": g["home"]["abbr"] or g["home"]["name"], "teams": {}}
        for side in ("home", "away"):
            team, opp = g[side], g["away" if side == "home" else "home"]
            is_home = side == "home"
            opp_hand = hand_map.get((opp["probable"] or {}).get("id"))
            posted = team.get("lineup") or []
            if posted:
                status = "confirmed"
                lineup = [(p["id"], i + 1, p["name"]) for i, p in enumerate(posted)]
            else:
                status = "projected"
                proj = _projected_lineup(history, team["id"], g["officialDate"])
                name_map = {r["id"]: r["name"]
                            for r in fetch.get_roster(team["id"], date=g["officialDate"])}
                lineup = [(pid, slot, name_map.get(pid, str(pid))) for pid, slot in proj]
            lineup_ids = {pid for pid, _, _ in lineup}
            for pid, slot, name in lineup:
                bat_rows.append(_bat_target(pid, name, team["id"], slot, is_home, g, opp,
                                            opp_hand, bat_map.get(pid), status, False))
            # Bench: active-roster position players not in the lineup.
            for r in fetch.get_roster(team["id"], date=g["officialDate"]):
                if r["id"] in lineup_ids or r["pos"] == "P":
                    continue
                bat_rows.append(_bat_target(r["id"], r["name"], team["id"], BENCH_SLOT,
                                            is_home, g, opp, opp_hand, bat_map.get(r["id"]),
                                            status, True))
            # Starting pitcher.
            prob = team.get("probable")
            if prob:
                pit_rows.append(_pit_target(prob["id"], prob["name"], team["id"], is_home,
                                            g, opp, hand_map.get(prob["id"])))
            gmeta["teams"][side] = {"abbr": team["abbr"] or team["name"],
                                    "lineup_status": status,
                                    "probable": (prob or {}).get("name")}
        slate_meta.append(gmeta)
    return pd.DataFrame(bat_rows), pd.DataFrame(pit_rows), slate_meta


def _model_stamp() -> str:
    h = hashlib.sha1()
    mdir = Path(__file__).resolve().parent.parent / "models"
    for p in sorted(mdir.glob(f"*_histgb_{M.MODEL_VERSION}.joblib")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def main(argv):
    date = argv[0] if argv else today_iso()
    season = int(date[:4])
    logger.warning("building predictions for %s (season %s)", date, season)

    slate = fetch.get_slate(date, today=today_iso())
    if not slate:
        logger.warning("no games on %s; nothing to do", date)
        return

    # Ensure the current season's gamelogs are current (append completed games
    # through the day before). Past complete seasons already have full parquets.
    load_seasons = list(range(season - 3, season + 1))
    if season >= int(today_iso()[:4]):
        pitch_hand, bat_side = backfill.handedness_maps()
        backfill.append_season_through(season, date, pitch_hand, bat_side)

    history = F.load_gamelogs([s for s in load_seasons if s >= 2019])
    if history.empty:
        raise SystemExit("no history; run build_training_backfill.py")
    # Strictly point-in-time: the whole slate is predicted pregame, so the
    # models may only see games BEFORE the target date. This also matters when
    # backfilling predictions for a past date whose games are already in the
    # loaded history: without this filter a target row would see its own real
    # game as a prior appearance (rest_days ~ 0, leaked rolling stats).
    history = history[history["officialDate"] < date].reset_index(drop=True)
    if history.empty:
        raise SystemExit("no history strictly before the target date")
    history = F.attach_catchers(history)
    uni = _universe()
    history["pitchHand"] = history["personId"].map(dict(zip(uni["personId"], uni["pitchHand"])))
    ctx = F.Context(CTX_YEARS)

    bat_t, pit_t, slate_meta = build_targets(slate, history, uni)
    logger.warning("targets: %d batters, %d pitchers", len(bat_t), len(pit_t))

    bat_art = M.load_artifacts(list(M.BAT_TARGETS))
    pit_art = M.load_artifacts(list(M.PIT_TARGETS))

    out_rows = []
    if not bat_t.empty:
        featb = F.compute_batter_features(history, targets=bat_t, ctx=ctx, universe=uni)
        predb = M.predict_batters(featb, bat_art)
        merged = bat_t.merge(predb, on=["personId", "gamePk"], how="left")
        # Carry a few headline inputs for the Player page transparency.
        merged = merged.merge(
            featb[["personId", "gamePk", "marcel_PApg", "rate_HR", "rate_SO", "opp_sp_hand"]],
            on=["personId", "gamePk"], how="left")
        out_rows.append(merged)
    if not pit_t.empty:
        featp = F.compute_pitcher_features(history, targets=pit_t, ctx=ctx, universe=uni)
        predp = M.predict_pitchers(featp, pit_art)
        merged = pit_t.merge(predp, on=["personId", "gamePk"], how="left")
        out_rows.append(merged)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    out["model_stamp"] = _model_stamp()
    out["built_for"] = date
    pred_path = cache.dc_path(f"predictions_{date.replace('-', '_')}_{M.MODEL_VERSION}.parquet")
    cache.atomic_to_parquet(out, pred_path)
    cache.json_save(cache.dc_path(f"slate_pred_{date.replace('-', '_')}_v1.json"), slate_meta)
    logger.warning("wrote %s (%d rows) and slate json", pred_path.name, len(out))


if __name__ == "__main__":
    main(sys.argv[1:])
