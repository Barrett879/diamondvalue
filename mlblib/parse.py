"""Shared boxscore parser.

ONE implementation of "boxscore JSON -> per-game player rows", imported by both
scripts/build_training_backfill.py (historical) and
scripts/build_daily_predictions.py (current-season append), so training and
inference features come from identical schemas.

One output row per (gamePk, personId, side). A "side" is home or away; a
two-way player (Ohtani type) produces a batter row and, if he pitched, the same
row carries pitching stats too (is_batter and is_pitcher can both be True).

Batting-order slot comes from the PER-PLAYER battingOrder string, never the
team-level battingOrder array (which is the final, post-substitution order). In
that string the hundreds digit is the slot and the suffix marks starter ("00")
vs substitute ("01"+). A dressed player who did not appear has battingOrder
None and played=False; we keep him (training filters on played, the roster UI
uses the full set).
"""
from __future__ import annotations

import math


def _ip_to_outs(ip) -> int | None:
    """MLB inningsPitched string "6.1" -> outs (6*3 + 1 = 19). "6.2" -> 20."""
    if ip is None:
        return None
    try:
        whole, _, frac = str(ip).partition(".")
        outs = int(whole) * 3
        if frac:
            outs += int(frac[0])  # frac digit is 0, 1, or 2 thirds of an inning
        return outs
    except (ValueError, TypeError):
        return None


def _slot_from_batting_order(bo: str | None):
    """(slot 1-9 or None, is_starter_slot bool, played bool) from a per-player
    battingOrder string like "400" (slot 4 starter) or "401" (slot 4 sub).
    """
    if not bo:
        return None, False, False
    try:
        n = int(bo)
    except (ValueError, TypeError):
        return None, False, False
    slot = n // 100
    is_starter = (n % 100) == 0
    return (slot if 1 <= slot <= 9 else None), is_starter, True


def _starting_pitcher_id(team_block: dict) -> int | None:
    """personId of the pitcher who started for a team (gamesStarted == 1)."""
    for pid, p in team_block.get("players", {}).items():
        pit = (p.get("stats") or {}).get("pitching") or {}
        if pit.get("gamesStarted") in (1, "1"):
            return p["person"]["id"]
    # Fallback: first entry in the pitchers list.
    pitchers = team_block.get("pitchers") or []
    return pitchers[0] if pitchers else None


def parse_boxscore(raw: dict, meta: dict) -> list[dict]:
    """Parse one boxscore into per-player rows.

    meta must carry the fields the boxscore itself lacks:
      gamePk, officialDate, gameDate, gameNumber, venue_id, dayNight
    (dayNight is "day"/"night"; pass None if unknown).
    """
    if not raw or "teams" not in raw:
        return []
    teams = raw["teams"]
    home_id = teams["home"]["team"]["id"]
    away_id = teams["away"]["team"]["id"]
    sp = {
        "home": _starting_pitcher_id(teams["home"]),
        "away": _starting_pitcher_id(teams["away"]),
    }

    rows: list[dict] = []
    for side in ("home", "away"):
        block = teams[side]
        team_id = home_id if side == "home" else away_id
        opp_side = "away" if side == "home" else "home"
        opp_team_id = away_id if side == "home" else home_id
        opp_starter = sp[opp_side]

        for pid, p in block.get("players", {}).items():
            person = p.get("person", {})
            stats = p.get("stats") or {}
            bat = stats.get("batting") or {}
            pit = stats.get("pitching") or {}
            pos = (p.get("position") or {}).get("abbreviation")
            slot, is_starter_slot, appeared_in_order = _slot_from_batting_order(
                p.get("battingOrder")
            )

            has_bat = bool(bat) and bat.get("plateAppearances") is not None
            has_pit = bool(pit) and pit.get("battersFaced") is not None
            played = appeared_in_order or has_bat or has_pit

            doubles = bat.get("doubles", 0) or 0
            triples = bat.get("triples", 0) or 0
            hr = bat.get("homeRuns", 0) or 0
            hits = bat.get("hits", 0) or 0
            singles = max(hits - doubles - triples - hr, 0)

            outs = _ip_to_outs(pit.get("inningsPitched")) if has_pit else None
            gs = pit.get("gamesStarted")
            is_sp = has_pit and gs in (1, "1")

            rows.append(
                {
                    # identity
                    "gamePk": meta["gamePk"],
                    "personId": person.get("id"),
                    "fullName": person.get("fullName"),
                    "side": side,
                    "teamId": team_id,
                    "oppTeamId": opp_team_id,
                    # game meta (point-in-time keys)
                    "officialDate": meta.get("officialDate"),
                    "gameDate": meta.get("gameDate"),
                    "gameNumber": meta.get("gameNumber", 1),
                    "venue_id": meta.get("venue_id"),
                    "dayNight": meta.get("dayNight"),
                    "isHome": side == "home",
                    # role
                    "position": pos,
                    "slot": slot,
                    "is_starter_slot": is_starter_slot,
                    "played": played,
                    "is_batter": has_bat,
                    "is_pitcher": has_pit,
                    "is_sp": is_sp,
                    "oppStarterId": opp_starter,
                    # batting counting stats
                    "PA": bat.get("plateAppearances"),
                    "AB": bat.get("atBats"),
                    "H": hits,
                    "b1": singles,
                    "b2": doubles,
                    "b3": triples,
                    "HR": hr,
                    "BB": bat.get("baseOnBalls"),
                    "IBB": bat.get("intentionalWalks"),
                    "HBP": bat.get("hitByPitch"),
                    "SO": bat.get("strikeOuts"),
                    "R": bat.get("runs"),
                    "RBI": bat.get("rbi"),
                    "SB": bat.get("stolenBases"),
                    "CS": bat.get("caughtStealing"),
                    "SF": bat.get("sacFlies"),
                    "SH": bat.get("sacBunts"),
                    "TB": bat.get("totalBases"),
                    # pitching counting stats
                    "p_outs": outs,
                    "p_BF": pit.get("battersFaced"),
                    "p_pitches": pit.get("numberOfPitches") or pit.get("pitchesThrown"),
                    "p_K": pit.get("strikeOuts"),
                    "p_BB": pit.get("baseOnBalls"),
                    "p_H": pit.get("hits"),
                    "p_HR": pit.get("homeRuns"),
                    "p_ER": pit.get("earnedRuns"),
                    "p_R": pit.get("runs"),
                }
            )
    return rows
