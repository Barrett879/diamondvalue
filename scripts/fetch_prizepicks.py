"""Best-effort live pull of PrizePicks MLB projection lines.

PrizePicks fronts its API with Cloudflare, so automated requests are usually
answered with 403 -- this will simply fail in that case and the Props page
falls back to a pasted payload. Where the client IS allowed through (some
networks/deployments), it pages the projections endpoint and writes the raw
JSON:API payload to cache/prizepicks_raw_{date}.json for props.compare to read.

Called on demand from the Props page's "Update now" button (never from the
daily cron: lines move intraday and a scheduled scrape would be stale, and
keeping the scrape off the automated path avoids hammering PrizePicks).

Usage: python scripts/fetch_prizepicks.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache  # noqa: E402
from mlblib.cache import logger  # noqa: E402
from mlblib.util import today_iso  # noqa: E402

MLB_LEAGUE_ID = 2
BASE = "https://api.prizepicks.com/projections"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 "
                   "Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
}


def raw_path(date: str):
    return cache.dc_path(f"prizepicks_raw_{date.replace('-', '_')}.json")


def fetch(date: str | None = None) -> dict | None:
    """Page the MLB projections endpoint into one merged JSON:API payload.
    Returns the payload dict on success, None on any failure (the caller shows
    the paste fallback). Writes the payload to cache on success."""
    import urllib.error
    import urllib.request

    date = date or today_iso()
    merged = {"data": [], "included": []}
    seen_inc = set()
    page, pages = 1, 1
    while page <= pages and page <= 12:
        url = (f"{BASE}?league_id={MLB_LEAGUE_ID}&per_page=250&page={page}"
               "&single_stat=true")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                payload = json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                ValueError) as e:
            logger.warning("prizepicks fetch failed (page %d): %s -- the site "
                           "likely blocked the request; use the paste box",
                           page, type(e).__name__)
            return None
        merged["data"].extend(payload.get("data", []))
        for inc in payload.get("included", []):
            k = (inc.get("type"), inc.get("id"))
            if k not in seen_inc:
                seen_inc.add(k)
                merged["included"].append(inc)
        pages = int(((payload.get("meta") or {}).get("total_pages")) or 1)
        page += 1
        time.sleep(1.0)
    cache.json_save(raw_path(date), merged)
    logger.warning("wrote %s (%d projections)", raw_path(date),
                   len(merged["data"]))
    return merged


def load_raw(date: str) -> dict | None:
    p = raw_path(date)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else today_iso()
    got = fetch(d)
    print("fetched" if got else "no data (blocked or empty) -- use the paste box")
