"""Build game-environment tables: venue geometry + per-game weather/umpires.

Two committed parquets:
  cache/venues_v1.parquet        per venue: lat, lon, elevation, azimuthAngle
                                 (home-plate-to-CF bearing), roofType
  cache/game_weather_v1.parquet  per gamePk (parsed from the ~17k boxscores
                                 already cached on disk, zero new fetches):
                                 temp_f, wind_mph, wind_out (park-relative
                                 out-toward-CF factor x mph), roof_closed,
                                 hp_ump (home-plate umpire name)

The boxscore 'Wind' string is already park-relative ("Out To CF", "In From
LF", "L To R"), so training needs no azimuth math; the azimuth matters only at
inference, when a meteorological forecast vector has to be projected into the
park frame.

Usage: python scripts/build_weather_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, fetch  # noqa: E402
from mlblib.cache import logger  # noqa: E402

# Park-relative out-component factor by wind-direction phrase.
_WIND_FACTOR = {
    "out to cf": 1.0, "out to lf": 0.7, "out to rf": 0.7,
    "in from cf": -1.0, "in from lf": -0.7, "in from rf": -0.7,
    "l to r": 0.0, "r to l": 0.0, "calm": 0.0, "none": 0.0, "varies": 0.0,
}


def build_venues() -> None:
    data = fetch._http_json(
        f"{fetch.STATS_BASE}/venues",
        {"sportId": 1, "hydrate": "location,fieldInfo"}, have_stale=False)
    rows = []
    for v in (data or {}).get("venues", []):
        loc = v.get("location", {})
        coords = loc.get("defaultCoordinates", {})
        fi = v.get("fieldInfo", {})
        rows.append({
            "venue_id": v.get("id"), "name": v.get("name"),
            "lat": coords.get("latitude"), "lon": coords.get("longitude"),
            "elevation": loc.get("elevation"),
            "azimuth": loc.get("azimuthAngle"),
            "roofType": fi.get("roofType"),
        })
    df = pd.DataFrame(rows)
    cache.atomic_to_parquet(df, cache.dc_path("venues_v1.parquet"))
    logger.warning("wrote venues_v1.parquet (%d venues)", len(df))


def _parse_info(info: list) -> dict:
    m = {i.get("label"): i.get("value") for i in info}
    out = {"temp_f": None, "wind_mph": None, "wind_out": None,
           "roof_closed": 0, "hp_ump": None}
    w = m.get("Weather") or ""
    t = re.match(r"\s*(\d+)\s*degrees", w)
    if t:
        out["temp_f"] = int(t.group(1))
    low = w.lower()
    if "roof closed" in low or "dome" in low:
        out["roof_closed"] = 1
    wind = (m.get("Wind") or "").lower().rstrip(". ")
    wm = re.match(r"\s*(\d+)\s*mph[, ]*(.*)", wind)
    if wm:
        mph = int(wm.group(1))
        out["wind_mph"] = mph
        phrase = wm.group(2).strip()
        factor = _WIND_FACTOR.get(phrase)
        out["wind_out"] = mph * factor if factor is not None else 0.0
    ump = m.get("Umpires") or ""
    um = re.search(r"HP:\s*([^.]+)\.", ump)
    if um:
        out["hp_ump"] = um.group(1).strip()
    return out


def build_game_weather() -> None:
    raw_dir = cache.CACHE_DIR / "raw_boxscores"
    rows = []
    files = sorted(raw_dir.glob("*.json"))
    for i, f in enumerate(files):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        rec = _parse_info(d.get("info", []))
        rec["gamePk"] = int(f.stem)
        rows.append(rec)
        if (i + 1) % 4000 == 0:
            logger.warning("parsed %d/%d boxscores", i + 1, len(files))
    df = pd.DataFrame(rows)
    cache.atomic_to_parquet(df, cache.dc_path("game_weather_v1.parquet"))
    got_t = df["temp_f"].notna().mean()
    got_u = df["hp_ump"].notna().mean()
    logger.warning("wrote game_weather_v1.parquet (%d games; temp %.0f%%, "
                   "ump %.0f%% coverage)", len(df), got_t * 100, got_u * 100)


if __name__ == "__main__":
    build_venues()
    build_game_weather()
