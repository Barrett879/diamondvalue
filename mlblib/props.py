"""PrizePicks line comparison: map posted prop lines to our per-game
projections and surface where the model disagrees with the market.

This is an INFORMATIONAL comparison of two numbers (our expected value vs the
posted line), not betting advice. It never recommends a wager or a stake.

Line input is decoupled from any scrape: PrizePicks' API is Cloudflare-
protected (automated requests get 403), so the reliable path is to paste the
projections JSON (or a simple "Name, Stat, Line" list) that the user copies
from their own browser. A best-effort live fetch is also provided; it works
only where PrizePicks does not block the client.
"""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd

# PrizePicks stat_type (lowercased) -> (columns to sum, scale). Resolved per
# ROLE: the same label ("Strikeouts", "Walks", "Hits") means different columns
# for a hitter vs a pitcher, so we pick the map by the matched player's role.
BAT_STAT_MAP = {
    "total bases": (["TB"], 1.0),
    "hits": (["H"], 1.0),
    "home runs": (["HR"], 1.0),
    "runs": (["R"], 1.0),
    "rbis": (["RBI"], 1.0),
    "runs+rbis": (["R", "RBI"], 1.0),
    "hits+runs+rbis": (["H", "R", "RBI"], 1.0),
    "walks": (["BB"], 1.0),
    "stolen bases": (["SB"], 1.0),
    "singles": (["b1"], 1.0),
    "doubles": (["b2"], 1.0),
    "triples": (["b3"], 1.0),
    "hitter strikeouts": (["SO"], 1.0),
    "batter strikeouts": (["SO"], 1.0),
    "strikeouts": (["SO"], 1.0),
    "plate appearances": (["PA"], 1.0),
}
PIT_STAT_MAP = {
    "pitcher strikeouts": (["K"], 1.0),
    "strikeouts": (["K"], 1.0),
    "hits allowed": (["H"], 1.0),
    "walks allowed": (["BB"], 1.0),
    "walks": (["BB"], 1.0),
    "pitches thrown": (["Pitches"], 1.0),
    "earned runs allowed": (["ER"], 1.0),
    "pitching outs": (["IP"], 3.0),   # outs = innings * 3
    "outs": (["IP"], 3.0),
    "innings pitched": (["IP"], 1.0),
}
# Human labels for the display, keyed by the lowercased stat_type.
STAT_LABEL = {
    "hits+runs+rbis": "H+R+RBI", "runs+rbis": "R+RBI",
    "pitcher strikeouts": "Pitcher K", "hitter strikeouts": "Batter K",
    "batter strikeouts": "Batter K", "hits allowed": "Hits Allowed",
    "walks allowed": "Walks Allowed", "pitches thrown": "Pitches",
    "earned runs allowed": "Earned Runs", "pitching outs": "Outs",
}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """Fold to a match key: strip accents, drop periods/suffixes, lowercase,
    collapse whitespace (the diacritics/period gotcha from the spreadsheet
    export)."""
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().replace(".", " ").replace("'", "").replace("-", " ")
    n = _SUFFIX.sub("", n)
    return re.sub(r"\s+", " ", n).strip()


def parse_prizepicks_json(raw) -> pd.DataFrame:
    """Parse the api.prizepicks.com/projections payload (dict or JSON string)
    into rows of (name, team, stat_type, line, start_time). Tolerant of the
    JSON:API shape (data[] projections + included[] new_player)."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    data = raw.get("data", []) if isinstance(raw, dict) else []
    included = raw.get("included", []) if isinstance(raw, dict) else []
    players = {}
    for inc in included:
        if inc.get("type") in ("new_player", "player"):
            a = inc.get("attributes", {})
            players[str(inc.get("id"))] = (a.get("name") or a.get("display_name"),
                                           a.get("team") or a.get("team_name"))
    rows = []
    for p in data:
        if p.get("type") != "projection":
            continue
        a = p.get("attributes", {})
        rel = (p.get("relationships", {}) or {}).get("new_player", {})
        pid = str(((rel.get("data") or {}) or {}).get("id"))
        name, team = players.get(pid, (a.get("description"), None))
        if not name or a.get("line_score") is None:
            continue
        rows.append({"name": name, "team": team,
                     "stat_type": a.get("stat_type") or a.get("stat_display_name"),
                     "line": float(a.get("line_score")),
                     "start_time": a.get("start_time")})
    return pd.DataFrame(rows)


def parse_line_list(text: str) -> pd.DataFrame:
    """Parse a pasted 'Name, Stat, Line' list (comma / pipe / tab separated),
    one prop per line. A permissive fallback when JSON is not to hand."""
    rows = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = re.split(r"\s*[|\t]\s*|\s*,\s*", ln)
        if len(parts) < 3:
            m = re.match(r"^(.*?)\s+([A-Za-z+ ]+?)\s+([\d.]+)$", ln)
            if not m:
                continue
            parts = [m.group(1), m.group(2), m.group(3)]
        try:
            line = float(parts[-1])
        except ValueError:
            continue
        rows.append({"name": parts[0].strip(), "team": None,
                     "stat_type": parts[1].strip(), "line": line,
                     "start_time": None})
    return pd.DataFrame(rows)


def compare(lines: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join posted lines to our projections and compute the model-vs-line gap.

    Returns (table, meta). table columns: Player, Team, Stat, Model, Line,
    Edge (model - line), Lean (Over/Under), _abs. meta reports matched /
    unmatched counts. Sorted by the size of the disagreement.
    """
    if lines is None or lines.empty or preds is None or preds.empty:
        return pd.DataFrame(), {"matched": 0, "unmatched": 0, "unmapped": 0}
    by_name: dict[str, list] = {}
    for _, r in preds.iterrows():
        by_name.setdefault(normalize_name(r["fullName"]), []).append(r)

    out, unmatched, unmapped = [], 0, 0
    for _, ln in lines.iterrows():
        key = normalize_name(ln["name"])
        cands = by_name.get(key)
        if not cands:
            unmatched += 1
            continue
        stat = str(ln.get("stat_type") or "").strip().lower()
        picked = None
        for row in cands:
            smap = BAT_STAT_MAP if row["role"] == "bat" else PIT_STAT_MAP
            if stat in smap:
                cols, scale = smap[stat]
                if all(c in row and row[c] == row[c] for c in cols):
                    picked = (row, cols, scale)
                    break
        if picked is None:
            unmapped += 1
            continue
        row, cols, scale = picked
        model = scale * float(sum(float(row[c]) for c in cols))
        line = float(ln["line"])
        edge = model - line
        out.append({
            "Player": row["fullName"],
            "Team": ln.get("team") or "",
            "Stat": STAT_LABEL.get(stat, str(ln.get("stat_type"))),
            "Model": round(model, 2),
            "Line": round(line, 2),
            "Edge": round(edge, 2),
            "Lean": "Over" if edge > 0 else ("Under" if edge < 0 else "Even"),
            "_abs": abs(edge),
        })
    table = pd.DataFrame(out)
    if not table.empty:
        table = table.sort_values("_abs", ascending=False).drop(columns="_abs")
    return table, {"matched": len(table), "unmatched": unmatched,
                   "unmapped": unmapped}
