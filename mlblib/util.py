"""Small shared helpers used across pages and pipeline (no heavy deps)."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


def game_time_et(game_date_utc: str | None) -> str:
    """Format a UTC ISO gameDate as a US/Eastern clock time, e.g. '7:05 PM ET'.

    Returns '' when the input is missing or unparseable.
    """
    if not game_date_utc:
        return ""
    try:
        # gameDate looks like '2025-07-10T23:05:00Z'
        s = game_date_utc.replace("Z", "+00:00")
        utc = dt.datetime.fromisoformat(s)
        local = utc.astimezone(_ET)
        return local.strftime("%-I:%M %p ET")
    except (ValueError, TypeError):
        return ""


def today_iso() -> str:
    return dt.date.today().isoformat()


def game_url(date: str, game_pk, dark: bool) -> str:
    """Link to the game detail page, preserving the date and theme so a
    full-reload navigation (each Streamlit page is a fresh session) keeps them.
    """
    theme = "dark" if dark else "light"
    return f"/Game?date={date}&gamePk={game_pk}&theme={theme}"


def parse_iso_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except (ValueError, TypeError):
        return None
