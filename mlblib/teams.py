"""MLB team identity colors, keyed by the abbreviations the app renders.

Primary is the team's signature color (used for card accents); on is a text
color (#fff or #111) that reads on the primary. Aliases cover the abbreviation
variants the Stats API / schedule feed uses across seasons (OAK/ATH, ARI/AZ,
CWS/CHW, SF/SFG, TB/TBR, SD/SDP, KC/KCR, WSH/WSN). Colors only -- no logos.
"""
from __future__ import annotations

# canonical abbr -> (primary hex, on-primary text hex)
_PRIMARY = {
    "ARI": ("#A71930", "#fff"),
    "ATL": ("#13274F", "#fff"),
    "BAL": ("#DF4601", "#fff"),
    "BOS": ("#BD3039", "#fff"),
    "CHC": ("#0E3386", "#fff"),
    "CWS": ("#27251F", "#fff"),
    "CIN": ("#C6011F", "#fff"),
    "CLE": ("#0C2340", "#fff"),
    "COL": ("#333366", "#fff"),
    "DET": ("#0C2340", "#fff"),
    "HOU": ("#002D62", "#fff"),
    "KC":  ("#004687", "#fff"),
    "LAA": ("#BA0021", "#fff"),
    "LAD": ("#005A9C", "#fff"),
    "MIA": ("#00A3E0", "#fff"),
    "MIL": ("#12284B", "#fff"),
    "MIN": ("#002B5C", "#fff"),
    "NYM": ("#002D72", "#fff"),
    "NYY": ("#0C2340", "#fff"),
    "OAK": ("#003831", "#fff"),
    "PHI": ("#E81828", "#fff"),
    "PIT": ("#FDB827", "#111"),
    "SD":  ("#2F241D", "#fff"),
    "SEA": ("#0C2C56", "#fff"),
    "SF":  ("#FD5A1E", "#fff"),
    "STL": ("#C41E3A", "#fff"),
    "TB":  ("#092C5C", "#fff"),
    "TEX": ("#003278", "#fff"),
    "TOR": ("#134A8E", "#fff"),
    "WSH": ("#AB0003", "#fff"),
}
# alias -> canonical
_ALIAS = {
    "AZ": "ARI", "ATH": "OAK", "CHW": "CWS", "SFG": "SF", "TBR": "TB",
    "SDP": "SD", "KCR": "KC", "WSN": "WSH", "WAS": "WSH", "CHA": "CWS",
    "CHN": "CHC", "NYA": "NYY", "NYN": "NYM", "LAN": "LAD", "SLN": "STL",
    "SDN": "SD", "SFN": "SF", "TBA": "TB", "KCA": "KC", "WSA": "WSH",
    "ANA": "LAA", "FLA": "MIA",
}

_NEUTRAL = ("#8a8a93", "#fff")


def team_color(abbr: str | None) -> tuple[str, str]:
    """(primary, on-text) for a team abbreviation; a neutral gray for unknowns
    (never raises, so a new/odd abbr just renders gray)."""
    if not abbr:
        return _NEUTRAL
    key = str(abbr).strip().upper()
    key = _ALIAS.get(key, key)
    return _PRIMARY.get(key, _NEUTRAL)
