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

import datetime as dt
import faulthandler
import hashlib
import re
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


def _recent_catcher(history, team_id, before_date, roster) -> float | None:
    """Most frequent starting catcher over the team's last 10 games, filtered
    to the current active roster (an IL'd or traded catcher must not keep
    winning the projection)."""
    sub = history[(history["teamId"] == team_id)
                  & (history["officialDate"] < before_date)
                  & (history["position"] == "C") & history["is_starter_slot"]]
    if sub.empty:
        return None
    recent_pks = (sub[["gamePk", "gameDate"]].drop_duplicates()
                  .sort_values("gameDate").tail(10)["gamePk"])
    counts = Counter(sub[sub["gamePk"].isin(recent_pks)]["personId"])
    roster_ids = {r["id"] for r in roster}
    for pid, _ in counts.most_common():
        if pid in roster_ids:
            return pid
    return None


def _slate_catchers(g, history) -> dict:
    """Today's catcher per side: GUMBO pregame boxscore position first (the
    only source with per-game fielding positions), else the last-10-games
    most-frequent starter still on the roster."""
    cmap = fetch.get_gumbo_lineup_catchers(g["gamePk"])
    for side in ("home", "away"):
        if cmap.get(side) is None:
            roster = fetch.get_roster(g[side]["id"], date=g["officialDate"])
            cmap[side] = _recent_catcher(history, g[side]["id"],
                                         g["officialDate"], roster)
    return cmap


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
    bat_df, pit_df = pd.DataFrame(bat_rows), pd.DataFrame(pit_rows)
    # Round 6: catcher identities per (gamePk, teamId) for the framing
    # features -- batters face the OPPOSING catcher, starters throw to their
    # OWN catcher.
    cat_map = {}
    for g in slate:
        cmap = _slate_catchers(g, history)
        for side in ("home", "away"):
            cat_map[(g["gamePk"], g[side]["id"])] = cmap[side]
    if not bat_df.empty:
        bat_df["oppCatcherId"] = [cat_map.get((pk, tid))
                                  for pk, tid in zip(bat_df["gamePk"],
                                                     bat_df["oppTeamId"])]
    if not pit_df.empty:
        pit_df["ownCatcherId"] = [cat_map.get((pk, tid))
                                  for pk, tid in zip(pit_df["gamePk"],
                                                     pit_df["teamId"])]
    return bat_df, pit_df, slate_meta


_YDAY_SCHED: dict[str, list] = {}


def _crew_rotation_hp(g: dict) -> str | None:
    """Infer tonight's HP umpire from yesterday's crew when GUMBO hasn't
    posted officials yet (morning runs). Standard 4-man rotation: yesterday's
    1B umpire works the plate today — 97.2% accurate over 10,024
    consecutive-day series games, 2019-2025. Series openers return None.
    """
    try:
        prev = (dt.date.fromisoformat(g["officialDate"])
                - dt.timedelta(days=1)).isoformat()
    except (KeyError, ValueError, TypeError):
        return None
    if prev not in _YDAY_SCHED:
        data = fetch.get_schedule_raw(prev) or {}
        _YDAY_SCHED[prev] = [gm for d in data.get("dates", [])
                             for gm in d.get("games", [])]
    pair = {g["home"]["id"], g["away"]["id"]}
    cands = [gm for gm in _YDAY_SCHED[prev]
             if {gm["teams"]["home"]["team"].get("id"),
                 gm["teams"]["away"]["team"].get("id")} == pair
             and (gm.get("status") or {}).get("codedGameState") == "F"]
    if not cands:
        return None
    last = max(cands, key=lambda gm: gm.get("gameNumber", 1))
    box = fetch.get_boxscore_raw(last["gamePk"])
    if not box:
        return None
    ump = next((i.get("value") or "" for i in box.get("info", [])
                if i.get("label") == "Umpires"), "")
    m = re.search(r"1B:\s*([^.]+)\.", ump)
    return m.group(1).strip() if m else None


def _game_environment(slate: list[dict], history: pd.DataFrame) -> dict:
    """Per-gamePk pregame environment: forecast temp, park-relative wind-out
    component, roof state, and (when posted) the HP umpire.

    Dome: 72F, no wind, roof=1. Retractable roof: forecast weather with the
    wind dampened by the venue's historical closed share (roof state unknown
    pregame). Open: forecast as-is. Umpires populate via GUMBO only once a
    game hits Pre-Game status, so morning runs carry None (models route NaN).
    """
    venues = cache.read_parquet_or_none(cache.dc_path("venues_v1.parquet"))
    gw = cache.read_parquet_or_none(cache.dc_path("game_weather_v1.parquet"))
    if venues is None:
        return {}
    vmap = venues.set_index("venue_id").to_dict("index")
    closed_share = {}
    if gw is not None and not history.empty:
        vg = history[["gamePk", "venue_id"]].drop_duplicates().merge(
            gw[["gamePk", "roof_closed"]], on="gamePk", how="inner")
        closed_share = vg.groupby("venue_id")["roof_closed"].mean().to_dict()

    env = {}
    for g in slate:
        v = vmap.get(g.get("venue_id"))
        rec = {"temp_f": np.nan, "wind_out": np.nan, "roof_closed": np.nan,
               "hp_ump": None}
        if v is not None:
            roof = (v.get("roofType") or "Open")
            if roof == "Dome":
                rec.update(temp_f=72.0, wind_out=0.0, roof_closed=1.0)
            else:
                fc = fetch.get_forecast(v["lat"], v["lon"],
                                        g["officialDate"]) or {}
                hour_key = (g.get("gameDate") or "")[:13] + ":00"
                got = fc.get(hour_key)
                if got:
                    temp, ws, wd = got
                    az = v.get("azimuth")
                    if az is not None and ws is not None and wd is not None:
                        import math
                        blow_to = (wd + 180.0) % 360.0
                        rec["wind_out"] = ws * math.cos(
                            math.radians(blow_to - az))
                    rec["temp_f"] = temp
                if roof == "Retractable":
                    share = float(closed_share.get(g.get("venue_id"), 0.5))
                    rec["roof_closed"] = share
                    if rec["wind_out"] == rec["wind_out"]:  # not NaN
                        rec["wind_out"] *= (1.0 - share)
                else:
                    rec["roof_closed"] = 0.0
        rec["hp_ump"] = (fetch.get_gumbo_hp_umpire(g["gamePk"])
                         or _crew_rotation_hp(g))
        env[g["gamePk"]] = rec
    return env


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

    # Pregame environment (forecast weather, roof, HP ump when posted) onto
    # both target frames; features.py coalesces these with historical actuals.
    env = _game_environment(slate, history)
    for frame in (bat_t, pit_t):
        if frame.empty:
            continue
        for col in ("temp_f", "wind_out", "roof_closed", "hp_ump"):
            frame[col] = frame["gamePk"].map(
                lambda pk, c=col: (env.get(pk) or {}).get(c))

    bat_art = M.load_artifacts(list(M.BAT_TARGETS))
    pit_art = M.load_artifacts(list(M.PIT_TARGETS))

    out_rows = []
    featb = None
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
        # Lineup-aggregated arsenal matchup for today's starters: mean expected
        # whiff of the (posted or projected) lineup they will face, from the
        # batter feature frame computed above. Mirrors the training-side
        # aggregation over historical starting lineups in features.py.
        if featb is not None and "mu_xwhiff" in featb.columns:
            lineup_ids = bat_t[~bat_t["is_bench"]][["personId", "gamePk"]]
            mu = featb.merge(lineup_ids, on=["personId", "gamePk"])
            agg = mu.groupby("gamePk")["mu_xwhiff"].mean()
            pit_t = pit_t.assign(opp_lineup_xwhiff=pit_t["gamePk"].map(agg))
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
