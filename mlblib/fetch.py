"""Network fetch layer for DiamondValue.

Every function that touches the network follows HoopsValue's "stale beats empty"
template: read the possibly-stale disk cache first, return it when fresh, do a
BOUNDED retry loop (3 tries if a stale copy exists, 8 if not, exponential
backoff, per-request timeout=15) on a miss, and on total failure serve the
stale copy with a logged warning rather than blocking or raising. A hung
endpoint must never freeze a page behind a spinner.

Two data sources live here:
  - MLB Stats API (statsapi.mlb.com) — free, keyless, unofficial. All game,
    schedule, roster, boxscore, and player-universe data.
  - Baseball Savant (baseballsavant.mlb.com) — park factors + catcher metrics.

Hydrate failures are SILENT: a misspelled hydrate returns a normal response
with the hydration simply missing. After each hydrated fetch we assert the
expected keys and log a warning if absent.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

from . import cache
from .cache import logger

STATS_BASE = "https://statsapi.mlb.com/api/v1"
STATS_BASE_11 = "https://statsapi.mlb.com/api/v1.1"
SAVANT_BASE = "https://baseballsavant.mlb.com"

# Plain, standard headers. An unusual Accept header can trigger HTTP 406 from
# statsapi; requests' defaults plus an explicit JSON Accept are safe.
_HEADERS = {
    "User-Agent": "DiamondValue/1.0 (personal, non-commercial hobby project)",
    "Accept": "application/json",
}

_session = requests.Session()
_session.headers.update(_HEADERS)

# Baseball Savant 403s non-browser User-Agents, so its leaderboards get their
# own session with a browser UA. statsapi is happy with either.
_SAVANT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,text/csv,*/*",
}
_savant = requests.Session()
_savant.headers.update(_SAVANT_HEADERS)


# ── Low-level bounded fetch ──────────────────────────────────────────────────
def _http_json(url: str, params: dict | None, have_stale: bool) -> dict | None:
    """Bounded-retry GET returning parsed JSON, or None on total failure."""
    attempts = 3 if have_stale else 8
    delay = 1.0
    for _ in range(attempts):
        try:
            r = _session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            logger.debug("fetch retry %s: %s", url, e)
            time.sleep(delay)
            delay = min(delay * 2, 15)
    return None


def _http_text(url: str, params: dict | None, have_stale: bool,
               session: requests.Session | None = None) -> str | None:
    sess = session or _session
    attempts = 3 if have_stale else 8
    delay = 1.0
    for _ in range(attempts):
        try:
            r = sess.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001
            logger.debug("fetch retry %s: %s", url, e)
            time.sleep(delay)
            delay = min(delay * 2, 15)
    return None


# ── Schedule + slate ─────────────────────────────────────────────────────────
def get_schedule_raw(date: str) -> dict | None:
    """Raw schedule payload for one ISO date, hydrated with probables + lineups.

    Not disk-cached itself; get_slate() parses and caches the useful subset.
    """
    params = {
        "sportId": 1,
        "date": date,
        "hydrate": "probablePitcher(note),lineups,linescore,venue,team",
    }
    have_stale = False
    data = _http_json(f"{STATS_BASE}/schedule", params, have_stale)
    return data


def _parse_game(g: dict) -> dict:
    """Pull the slate-relevant fields out of one schedule game object."""
    home = g["teams"]["home"]
    away = g["teams"]["away"]

    def _probable(side: dict) -> dict | None:
        pp = side.get("probablePitcher")
        if not pp:
            return None
        return {"id": pp.get("id"), "name": pp.get("fullName"), "note": pp.get("note")}

    def _lineup(key: str) -> list[dict]:
        lu = g.get("lineups", {}) or {}
        players = lu.get(key, []) or []
        return [
            {
                "id": p.get("id"),
                "name": p.get("fullName"),
                "pos": (p.get("primaryPosition") or {}).get("abbreviation"),
            }
            for p in players
        ]

    return {
        "gamePk": g["gamePk"],
        "officialDate": g.get("officialDate"),
        "gameDate": g.get("gameDate"),  # UTC ISO; display in Eastern
        "gameNumber": g.get("gameNumber", 1),
        "doubleHeader": g.get("doubleHeader", "N"),
        "status": (g.get("status") or {}).get("codedGameState"),
        "detailedState": (g.get("status") or {}).get("detailedState"),
        "venue_id": (g.get("venue") or {}).get("id"),
        "venue_name": (g.get("venue") or {}).get("name"),
        "home": {
            "id": home["team"]["id"],
            "name": home["team"].get("name"),
            "abbr": (home["team"].get("abbreviation")),
            "probable": _probable(home),
            "lineup": _lineup("homePlayers"),
        },
        "away": {
            "id": away["team"]["id"],
            "name": away["team"].get("name"),
            "abbr": (away["team"].get("abbreviation")),
            "probable": _probable(away),
            "lineup": _lineup("awayPlayers"),
        },
    }


def get_slate(date: str, today: str | None = None) -> list[dict]:
    """Parsed games for one ISO date (probables + posted lineups), disk-cached.

    Stale-beats-empty: a fresh disk copy short-circuits the network; a failed
    refresh serves the stale copy; only a cold miss with no stale copy hits the
    network hard, and even that returns [] rather than raising.
    """
    path = cache.dc_path(f"slate_{date.replace('-', '_')}_v1.json")
    stale = None
    if path.exists():
        try:
            stale = cache.json_load(path)
        except Exception:  # noqa: BLE001
            stale = None
    if stale is not None and cache.dc_fresh(path, game_date=date, today=today):
        return stale

    params = {
        "sportId": 1,
        "date": date,
        "hydrate": "probablePitcher(note),lineups,linescore,venue,team",
    }
    data = _http_json(f"{STATS_BASE}/schedule", params, have_stale=stale is not None)
    if data is None:
        if stale is not None:
            logger.warning("slate refresh failed for %s — serving stale", date)
            return stale
        return []

    dates = data.get("dates", [])
    games = []
    for d in dates:
        for g in d.get("games", []):
            # Silent-hydrate guard: schedule always carries teams; probables and
            # lineups may legitimately be absent (not yet posted), so we only
            # warn when the core structure is missing.
            if "teams" not in g:
                logger.warning("schedule game %s missing teams key", g.get("gamePk"))
                continue
            games.append(_parse_game(g))
    games.sort(key=lambda x: (x.get("officialDate") or "", x.get("gameNumber") or 1,
                              x.get("gameDate") or ""))
    try:
        cache.json_save(path, games)
    except Exception:  # noqa: BLE001
        pass
    return games


# ── Boxscore (training backfill + daily append) ──────────────────────────────
def get_boxscore_raw(game_pk: int, force: bool = False) -> dict | None:
    """Raw boxscore JSON for one game, cached under cache/raw_boxscores/.

    Final games are immutable once past the correction window, so a cached copy
    is reused permanently. `force=True` re-fetches (used when a game just went
    final in the daily pipeline).
    """
    raw_dir = cache.CACHE_DIR / "raw_boxscores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{game_pk}.json"
    if path.exists() and not force:
        try:
            return cache.json_load(path)
        except Exception:  # noqa: BLE001
            pass
    data = _http_json(f"{STATS_BASE}/game/{game_pk}/boxscore", None,
                      have_stale=path.exists())
    if data is None:
        if path.exists():
            try:
                return cache.json_load(path)
            except Exception:  # noqa: BLE001
                return None
        return None
    if "teams" in data:
        try:
            cache.json_save(path, data)
        except Exception:  # noqa: BLE001
            pass
    return data


# ── Rosters ──────────────────────────────────────────────────────────────────
def get_roster(team_id: int, date: str | None = None, today: str | None = None) -> list[dict]:
    """Active roster for one team; date= gives a historical snapshot."""
    tag = f"_{date.replace('-', '_')}" if date else ""
    path = cache.dc_path(f"roster_{team_id}{tag}_v1.json")
    stale = None
    if path.exists():
        try:
            stale = cache.json_load(path)
        except Exception:  # noqa: BLE001
            stale = None
    # Rosters churn daily; give the current one a short TTL.
    if stale is not None and cache.dc_fresh(path, ttl=1800):
        return stale
    params = {"rosterType": "active"}
    if date:
        params["date"] = date
    data = _http_json(f"{STATS_BASE}/teams/{team_id}/roster", params,
                      have_stale=stale is not None)
    if data is None:
        if stale is not None:
            logger.warning("roster refresh failed for team %s — serving stale", team_id)
            return stale
        return []
    out = [
        {
            "id": p["person"]["id"],
            "name": p["person"]["fullName"],
            "pos": (p.get("position") or {}).get("abbreviation"),
            "status": (p.get("status") or {}).get("code"),
        }
        for p in data.get("roster", [])
    ]
    try:
        cache.json_save(path, out)
    except Exception:  # noqa: BLE001
        pass
    return out


# ── Player universe (batSide / pitchHand / team / birthDate) ─────────────────
def get_player_universe(season: int) -> pd.DataFrame:
    """Every MLB player active in a season with handedness, team, position, DOB.

    Cached as a committed parquet (player_universe_{season}_v1.parquet) so the
    site has handedness/age at inference without a network call.
    """
    path = cache.dc_path(f"player_universe_{season}_v1.parquet")
    stale = cache.read_parquet_or_none(path)
    # Current-year universe changes with call-ups; older years are immutable.
    fresh = cache.dc_fresh(path, ttl=86400) if stale is not None else False
    if stale is not None and fresh:
        return stale
    data = _http_json(f"{STATS_BASE}/sports/1/players", {"season": season},
                      have_stale=stale is not None)
    if data is None:
        if stale is not None:
            logger.warning("player universe refresh failed for %s — serving stale", season)
            return stale
        return pd.DataFrame()
    rows = []
    for p in data.get("people", []):
        rows.append(
            {
                "personId": p.get("id"),
                "fullName": p.get("fullName"),
                "teamId": (p.get("currentTeam") or {}).get("id"),
                "primaryPosition": (p.get("primaryPosition") or {}).get("abbreviation"),
                "batSide": (p.get("batSide") or {}).get("code"),
                "pitchHand": (p.get("pitchHand") or {}).get("code"),
                "birthDate": p.get("birthDate"),
                "mlbDebutDate": p.get("mlbDebutDate"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            cache.atomic_to_parquet(df, path)
        except Exception:  # noqa: BLE001
            pass
    return df


# ── Schedule range enumeration (backfill) ────────────────────────────────────
def get_schedule_range(start: str, end: str, game_type: str = "R") -> list[dict]:
    """Raw game entries across a date range. NOT deduped/filtered here — the
    backfill dedupes by gamePk and keeps only Final games (see build script).
    """
    params = {
        "sportId": 1,
        "startDate": start,
        "endDate": end,
        "gameType": game_type,
    }
    data = _http_json(f"{STATS_BASE}/schedule", params, have_stale=False)
    if data is None:
        return []
    out = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            out.append(g)
    return out


# ── Baseball Savant ──────────────────────────────────────────────────────────
def get_park_factors_raw(year: int, bat_side: str = "") -> list[dict] | None:
    """Park-factor rows for a season, parsed from the leaderboard's embedded
    `var data = [...]` JSON. csv=true does NOT work for this leaderboard.

    bat_side: "" (All), "L", or "R".
    """
    params = {"type": "year", "year": year, "batSide": bat_side}
    html = _http_text(f"{SAVANT_BASE}/leaderboard/statcast-park-factors", params,
                      have_stale=False, session=_savant)
    if not html:
        return None
    m = re.search(r"var\s+data\s*=\s*(\[.*?\]);", html, re.S)
    if not m:
        logger.warning("park factors: embedded 'var data' not found for %s", year)
        return None
    import json as _json

    try:
        return _json.loads(m.group(1))
    except Exception as e:  # noqa: BLE001
        logger.warning("park factors JSON parse failed for %s: %s", year, e)
        return None


def get_forecast(lat: float, lon: float, date_iso: str) -> dict | None:
    """Hourly forecast for one venue-day from Open-Meteo (keyless, free for
    non-commercial use). Returns {hour_iso_utc: (temp_f, wind_mph, wind_dir)}
    or None on failure (features degrade to NaN, never block).
    """
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "UTC",
        "start_date": date_iso, "end_date": date_iso,
    }
    data = _http_json("https://api.open-meteo.com/v1/forecast", params,
                      have_stale=True)  # bounded 3 tries; missing weather is OK
    if not data or "hourly" not in data:
        return None
    h = data["hourly"]
    out = {}
    for t, temp, ws, wd in zip(h.get("time", []), h.get("temperature_2m", []),
                               h.get("wind_speed_10m", []),
                               h.get("wind_direction_10m", [])):
        out[t] = (temp, ws, wd)
    return out


def get_gumbo_hp_umpire(game_pk: int) -> str | None:
    """Home-plate umpire from the GUMBO live feed. Populates when the game
    reaches Pre-Game status (~1-3h before first pitch); None before that.
    """
    data = _http_json(f"{STATS_BASE_11}/game/{game_pk}/feed/live", None,
                      have_stale=True)
    if not data:
        return None
    for o in (((data.get("liveData") or {}).get("boxscore") or {})
              .get("officials") or []):
        if (o.get("officialType") or "") == "Home Plate":
            return (o.get("official") or {}).get("fullName")
    return None


def get_gumbo_lineup_catchers(game_pk: int) -> dict:
    """Per side, today's CATCHER from the GUMBO live feed's pregame boxscore:
    the player whose fielding position is C and who carries a battingOrder
    (i.e. is in the posted lineup). Unlike the schedule lineups hydrate, this
    is the per-GAME fielding position, so a catcher DHing does not fool it.
    Returns {"home": personId|None, "away": personId|None}.
    """
    out = {"home": None, "away": None}
    data = _http_json(f"{STATS_BASE_11}/game/{game_pk}/feed/live", None,
                      have_stale=True)
    if not data:
        return out
    teams = ((data.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    for side in ("home", "away"):
        for pl in (teams.get(side, {}).get("players") or {}).values():
            pos = ((pl.get("position") or {}).get("abbreviation") or "")
            if pos == "C" and pl.get("battingOrder"):
                out[side] = (pl.get("person") or {}).get("id")
                break
    return out


def get_savant_csv(leaderboard: str, year: int) -> str | None:
    """Raw CSV text from a Savant leaderboard that supports csv=true
    (catcher-framing, catcher-throwing).
    """
    params = {"year": year, "csv": "true"}
    return _http_text(f"{SAVANT_BASE}/leaderboard/{leaderboard}", params,
                      have_stale=False, session=_savant)
