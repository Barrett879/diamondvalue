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
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import cache

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover  # noqa: BLE001
    _ET = None

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


def _person_like(name) -> bool:
    """True when a string plausibly names a person (has a space and lowercase).
    The feed's attributes.description is the TEAM code ("BOS"), so it must
    never stand in for a missing player name -- a paste of a payload without
    its included[] player list would otherwise save thousands of lines keyed
    to team codes that can never match a player."""
    return (isinstance(name, str) and " " in name.strip()
            and any(c.islower() for c in name))


def parse_prizepicks_json(raw) -> pd.DataFrame:
    """Parse the api.prizepicks.com/projections payload (dict or JSON string)
    into rows of (name, team, stat_type, line, start_time). Tolerant of the
    JSON:API shape (data[] projections + included[] new_player). Props whose
    player name cannot be resolved are skipped and counted in
    df.attrs['skipped_noname'] so the UI can say WHY a paste yielded nothing."""
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
    skipped = 0
    for p in data:
        if p.get("type") != "projection":
            continue
        a = p.get("attributes", {})
        rel = (p.get("relationships", {}) or {}).get("new_player", {})
        pid = str(((rel.get("data") or {}) or {}).get("id"))
        name, team = players.get(pid, (None, None))
        if not name and _person_like(a.get("description")):
            name = a.get("description")
        if a.get("line_score") is None:
            continue
        if not name:
            skipped += 1
            continue
        # odds_type ("standard"/"demon"/"goblin") when the feed carries it. The
        # feed does not spell out More vs Less, so we can only say "both" for a
        # standard prop; a Demon/Goblin is one-sided but the side is unknown here
        # (the pasted board text is the reliable source for direction).
        odds = (a.get("odds_type") or "standard").lower()
        rows.append({"name": name, "team": team,
                     "stat_type": a.get("stat_type") or a.get("stat_display_name"),
                     "line": float(a.get("line_score")),
                     "start_time": a.get("start_time"),
                     "direction": "both" if odds == "standard" else "",
                     "odds_type": odds})
    df = pd.DataFrame(rows)
    df.attrs["skipped_noname"] = skipped
    return df


_ODDS_TYPES = {"standard", "demon", "goblin"}
# A bare game date or a full ISO stamp (the grabber emits the feed's whole
# start_time so bucket_by_date can do the ET conversion here, not in JS).
_LIST_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\S+)?$")


def parse_line_list(text: str) -> pd.DataFrame:
    """Parse a pasted 'Name, Stat, Line' list (comma / pipe / tab separated),
    one prop per line. Also reads the one-click grabber's rows
    'Name | Stat | Line | odds_type [| YYYY-MM-DD]' (odds_type keeps the
    Demon/Goblin distinction; the optional trailing date is the prop's GAME
    date, so lines grabbed tonight for tomorrow's board land on tomorrow's
    slate). A permissive fallback when JSON is not to hand."""
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
        # Peel an optional trailing game date/stamp first (grabber rows), so
        # the odds/line logic below sees the same shape either way.
        gdate = None
        if len(parts) >= 4 and _LIST_DATE.match(parts[-1].strip()):
            gdate = parts[-1].strip()
            parts = parts[:-1]
        # 4-column grabber rows carry odds_type after the line; otherwise the
        # line is the last field.
        odds = ""
        if len(parts) >= 4 and parts[3].strip().lower() in _ODDS_TYPES:
            name, stat, line_str = parts[0], parts[1], parts[2]
            odds = parts[3].strip().lower()
        else:
            name, stat, line_str = parts[0], parts[1], parts[-1]
        try:
            line = float(line_str)
        except ValueError:
            continue
        if not np.isfinite(line):   # reject nan/inf tokens at the source
            continue
        row = {"name": name.strip(), "team": None, "stat_type": stat.strip(),
               "line": line, "start_time": gdate}
        if odds:   # the feed does not spell out More vs Less; standard = both
            row["odds_type"] = odds
            row["direction"] = "both" if odds == "standard" else ""
        rows.append(row)
    return pd.DataFrame(rows)


# Parsing the copied PrizePicks BOARD (the visual page text, not the JSON API).
# Each prop is a block: [tag] [count] [name(+Demon/Goblin)] [TEAM - POS] [name]
# [matchup] [line#] [stat] [Less?] [More]. The "TEAM - POS" line is the reliable
# anchor: the next line is the clean player name, then a matchup, then the line
# number, then the stat label.
_BOARD_TEAMPOS = re.compile(r"^[A-Z]{2,4} - [A-Za-z0-9]{1,3}$")
_BOARD_NUM = re.compile(r"^\d+(\.\d+)?$")
_BOARD_SUFFIX = re.compile(r"(Demon|Goblin)+$")
# Board shorthand -> a stat_type string compare() understands.
_BOARD_ALIAS = {"tb": "Total Bases", "ks": "Strikeouts"}
# Composite fantasy scores we do not project -- skip so they don't inflate the
# "stat types we don't project" count.
_BOARD_SKIP = {"hitter fs", "pitcher fs", "fantasy score", "hitter fantasy score",
               "pitcher fantasy score"}


def parse_prizepicks_board(text: str) -> pd.DataFrame:
    """Parse the text copied from the PrizePicks board page (not the JSON feed).
    Anchors on each 'TEAM - POS' line and reads the following name / line /
    stat; strips Demon/Goblin payout tags and skips fantasy-score props."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    rows = []
    for i, ln in enumerate(lines):
        if not _BOARD_TEAMPOS.match(ln) or i + 1 >= len(lines):
            continue
        team = ln.split(" - ")[0]
        name = _BOARD_SUFFIX.sub("", lines[i + 1]).strip()
        # Odds type rides on the pre-anchor name line (e.g. "James WoodGoblin");
        # the clean name at i+1 already has the tag stripped by the site. Demon =
        # harder line / bigger payout, Goblin = easier line / smaller payout;
        # both are one-sided, standard offers both sides.
        prev = lines[i - 1] if i > 0 else ""
        odds = ("demon" if prev.endswith("Demon")
                else "goblin" if prev.endswith("Goblin") else "standard")
        # The line number sits a couple rows down (after the matchup); scan a
        # small window so an occasional extra row doesn't break alignment.
        num_i = next((j for j in range(i + 2, min(i + 6, len(lines)))
                      if _BOARD_NUM.match(lines[j])), None)
        if num_i is None or num_i + 1 >= len(lines):
            continue
        try:
            line = float(lines[num_i])
        except ValueError:
            continue
        if not np.isfinite(line):
            continue
        stat = lines[num_i + 1].strip()
        key = stat.lower()
        if key in _BOARD_SKIP or _BOARD_TEAMPOS.match(stat):
            continue
        # Direction: the Less/More buttons follow the stat label. Both present =
        # you can take either side; one = that side only (typical for Demon/Goblin).
        j = num_i + 2
        btns = set()
        while j < len(lines) and lines[j] in ("Less", "More"):
            btns.add(lines[j])
            j += 1
        direction = ("both" if {"Less", "More"} <= btns
                     else "more" if "More" in btns
                     else "less" if "Less" in btns else "")
        rows.append({"name": name, "team": team,
                     "stat_type": _BOARD_ALIAS.get(key, stat),
                     "line": line, "start_time": None,
                     "direction": direction, "odds_type": odds})
    return pd.DataFrame(rows)


def parse_any(text: str) -> pd.DataFrame:
    """Parse pasted lines in whatever shape they arrive: the JSON feed, the
    copied board page, or a simple 'Name, Stat, Line' list. Returns the first
    parser that yields rows. A JSON paste that parsed but yielded no usable
    props (e.g. a payload with no player names) is returned as-is, empty with
    its diagnostic attrs -- falling through to the text parsers would shred
    the JSON into junk rows."""
    t = (text or "").strip()
    if not t:
        return pd.DataFrame()
    if t[:1] in "{[":
        try:
            df = parse_prizepicks_json(t)
        except Exception:  # noqa: BLE001 -- not valid JSON; try the text parsers
            df = None
        if df is not None and (not df.empty or df.attrs.get("skipped_noname")):
            return df
    df = parse_prizepicks_board(t)
    if df is not None and not df.empty:
        return df
    return parse_line_list(t)


def lines_path(date: str):
    return cache.dc_path(f"pp_lines_{date.replace('-', '_')}_v1.json")


def save_lines(date: str, lines: pd.DataFrame) -> None:
    """Persist the normalized posted lines for `date` so the Game pages can read
    them across a hard <a target=_self> deep link (session_state does not
    survive it). Called from the single Props funnel, so the paste flow finally
    persists too. The to_json/json.loads round-trip is load-bearing: a plain
    to_dict would hand json_save numpy.float64, which json.dumps cannot
    serialize (the write would fail silently). Keeps saved_at stable when the
    content is unchanged, so the freshness caption does not drift on reruns."""
    if lines is None or lines.empty:
        return
    keep = [c for c in ("name", "team", "stat_type", "line", "start_time",
                        "direction", "odds_type")
            if c in lines.columns]
    recs = json.loads(lines[keep].to_json(orient="records"))
    p = lines_path(date)
    if p.exists():
        try:
            if cache.json_load(p).get("lines") == recs:
                return
        except Exception:  # noqa: BLE001
            pass
    cache.json_save(p, {"saved_at": datetime.now(timezone.utc).isoformat(),
                        "lines": recs})


def load_lines(date: str) -> pd.DataFrame | None:
    """Read back the persisted lines for `date` (or None). The save time rides
    on the frame's .attrs['saved_at'] for the freshness caption."""
    p = lines_path(date)
    if not p.exists():
        return None
    try:
        blob = cache.json_load(p)
    except Exception:  # noqa: BLE001
        return None
    df = pd.DataFrame(blob.get("lines", []))
    if df.empty:
        return None
    df.attrs["saved_at"] = blob.get("saved_at")
    return df


def _dedup_batch(df: pd.DataFrame) -> pd.DataFrame:
    """One line per (player, stat) within a single frame. The feed posts a
    LADDER of alt lines for the same prop (a Demon at 8.5, another at 7.5, a
    Goblin at 5.5...), so a full-board paste has many rows per key: prefer the
    standard line (the two-sided market number), then Goblin, then Demon;
    among equals keep the first (the feed's rank order)."""
    if df is None or df.empty:
        return df
    keyed = df.assign(_k=(df["name"].map(normalize_name) + "|"
                          + df["stat_type"].astype(str).str.strip().str.lower()))
    if "odds_type" in df.columns:
        # Unknown odds types rank WORST (3): a future PrizePicks alt-line kind
        # must never shadow the posted standard line.
        pref = (df["odds_type"].fillna("").astype(str).str.lower()
                .map({"standard": 0, "": 0, "goblin": 1, "demon": 2}).fillna(3))
        keyed = (keyed.assign(_p=pref)
                 .sort_values("_p", kind="stable").drop(columns="_p"))
    return (keyed.drop_duplicates("_k", keep="first").drop(columns="_k")
            .reset_index(drop=True))


def merge_lines(existing: pd.DataFrame | None,
                new: pd.DataFrame | None) -> pd.DataFrame:
    """Union two line frames, deduped by (normalized name, stat_type), keeping
    the NEWEST value for a repeated prop. Lets a user paste one PrizePicks stat
    tab at a time and accumulate the whole board (there is no All tab), while a
    re-paste of the same tab just refreshes those lines instead of piling up.
    Each input is ladder-deduped first, so a first-ever full-board paste does
    not save thousands of alt lines."""
    frames = [_dedup_batch(f) for f in (existing, new)
              if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    both = pd.concat(frames, ignore_index=True)
    key = (both["name"].map(normalize_name) + "|"
           + both["stat_type"].astype(str).str.strip().str.lower())
    both = both[~key.duplicated(keep="last")]   # new frame is last -> it wins
    for c in ("direction", "odds_type"):        # keep string cols clean if mixed
        if c in both.columns:
            both[c] = both[c].fillna("")
    return both.reset_index(drop=True)


def bucket_by_date(lines: pd.DataFrame | None) -> dict:
    """Split parsed lines by the ET calendar date of each prop's start_time:
    {'YYYY-MM-DD': frame, ..., '': frame-with-unknown-dates}. The pre-game
    board posted tonight is TOMORROW's games, so lines must be saved under the
    date they are FOR, not the slate date on screen when they were pasted."""
    if lines is None or lines.empty or "start_time" not in lines.columns:
        return {"": lines if lines is not None else pd.DataFrame()}

    def _et_date(stamp) -> str:
        if not isinstance(stamp, str) or not stamp:
            return ""
        try:
            dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            m = re.match(r"^\d{4}-\d{2}-\d{2}", stamp)
            return m.group(0) if m else ""
        if dt.tzinfo is not None and _ET is not None:
            dt = dt.astimezone(_ET)
        return dt.date().isoformat()

    out = {}
    for d, grp in lines.groupby(lines["start_time"].map(_et_date)):
        out[str(d)] = grp.reset_index(drop=True)
    return out


def clear_lines(date: str) -> None:
    """Delete the accumulated lines for `date` (the 'Clear all' control)."""
    p = lines_path(date)
    try:
        if p.exists():
            p.unlink()
    except Exception:  # noqa: BLE001
        pass


_LINES_FILE = re.compile(r"^pp_lines_(\d{4})_(\d{2})_(\d{2})_v1\.json$")


def saved_line_dates() -> list[tuple[str, int]]:
    """[(date_iso, n_lines)] for EVERY date with saved lines, sorted by date.
    Disk is the truth here, deliberately not session state: date-routed pastes
    can land lines on several dates, and the UI must keep pointing at all of
    them (and 'Clear all' must clear all of them) regardless of what happened
    to this session's widgets since."""
    out = []
    try:
        paths = sorted(cache.CACHE_DIR.glob("pp_lines_*_v1.json"))
    except Exception:  # noqa: BLE001
        return []
    for p in paths:
        m = _LINES_FILE.match(p.name)
        if not m:
            continue
        d = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        df = load_lines(d)
        if df is not None and not df.empty:
            out.append((d, len(df)))
    return out


def saved_at_et(iso: str | None) -> str:
    """'Mon D, HH:MM ET' for a saved_at ISO-UTC stamp; '' if missing/
    unparseable. The date is always shown so a stale save on a past-date game
    page can't read as if the lines were pulled today. If the Eastern zone is
    unavailable (no tzdata), the time is labelled UTC rather than mislabelled
    as ET."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)
        if _ET is not None:
            dt = dt.astimezone(_ET)
            tz = "ET"
        else:
            tz = "UTC"
        return dt.strftime(f"%b {dt.day}, %H:%M {tz}")
    except Exception:  # noqa: BLE001
        return ""


# Prediction column -> (gamelog actual column, gamelog-to-prediction divisor).
# actual, in the LINE's units, = stat_scale * sum(gamelog[col] / divisor). Only
# IP differs: the gamelog stores outs (p_outs = IP * 3), so divisor 3 recovers
# innings before the stat_scale (1 for innings props, 3 for outs props) applies.
_ACTUAL_BAT = {c: (c, 1.0) for c in
               ("PA", "H", "HR", "b1", "b2", "b3", "SO", "BB", "TB", "R", "RBI", "SB")}
_ACTUAL_PIT = {"K": ("p_K", 1.0), "BB": ("p_BB", 1.0), "H": ("p_H", 1.0),
               "ER": ("p_ER", 1.0), "IP": ("p_outs", 3.0), "Pitches": ("p_pitches", 1.0)}


def _actual_for(cols, scale, role, arow) -> float | None:
    """The prop's ACTUAL value in the line's units, from a gamelog row, or None
    if the player has no scored result or a column is unmapped."""
    amap = _ACTUAL_BAT if role == "bat" else _ACTUAL_PIT
    try:
        total = 0.0
        for c in cols:
            gcol, div = amap[c]
            v = arow[gcol]
            if v != v:   # NaN -> did not play / not scored
                return None
            total += float(v) / div
        return round(scale * total, 2)
    except (KeyError, TypeError, ValueError):
        return None


def compare(lines: pd.DataFrame, preds: pd.DataFrame,
            actuals: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Join posted lines to our projections and compute the model-vs-line gap.

    Returns (table, meta). table columns: Player, Team, Stat, Model, Line,
    Edge (model - line), Lean (Over/Under), Direction, OddsType, Actual, _abs.
    `actuals` (per-game gamelog counts keyed by personId+gamePk) adds the ACTUAL
    result once a game is final; it stays None before then. meta reports matched
    / unmatched counts. Sorted by the size of the disagreement.
    """
    if lines is None or lines.empty or preds is None or preds.empty:
        return pd.DataFrame(), {"matched": 0, "unmatched": 0, "unmapped": 0}
    by_name: dict[str, list] = {}
    for _, r in preds.iterrows():
        by_name.setdefault(normalize_name(r["fullName"]), []).append(r)

    def _key(pid, gpk):
        try:
            return (int(pid), int(gpk))   # normalize dtype so the join can't
        except (TypeError, ValueError):   # silently miss on int-vs-float parquets
            return None

    act_lookup: dict = {}
    if actuals is not None and not actuals.empty:
        for _, ar in actuals.iterrows():
            k = _key(ar.get("personId"), ar.get("gamePk"))
            if k is not None:
                act_lookup[k] = ar

    out, unmatched, unmapped = [], 0, 0
    unmatched_names: list[str] = []
    for _, ln in lines.iterrows():
        key = normalize_name(ln["name"])
        cands = by_name.get(key)
        if not cands:
            unmatched += 1
            n = str(ln["name"])
            if len(unmatched_names) < 3 and n not in unmatched_names:
                unmatched_names.append(n)
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
        line = ln.get("line")
        if line is None or not np.isfinite(line):  # null/nan survived a paste
            unmapped += 1
            continue
        model = scale * float(sum(float(row[c]) for c in cols))
        line = float(line)
        edge = model - line
        arow = act_lookup.get(_key(row.get("personId"), row.get("gamePk")))
        actual = _actual_for(cols, scale, row["role"], arow) if arow is not None else None
        out.append({
            "Player": row["fullName"],
            "Team": ln.get("team") or "",
            "Stat": STAT_LABEL.get(stat, str(ln.get("stat_type"))),
            "Model": round(model, 2),
            "Line": round(line, 2),
            "Edge": round(edge, 2),
            "Lean": "Over" if edge > 0 else ("Under" if edge < 0 else "Even"),
            "Direction": ln.get("direction") or "",
            "OddsType": ln.get("odds_type") or "",
            "Actual": actual,
            "_abs": abs(edge),
        })
    table = pd.DataFrame(out)
    if not table.empty:
        table = table.sort_values("_abs", ascending=False).drop(columns="_abs")
    return table, {"matched": len(table), "unmatched": unmatched,
                   "unmapped": unmapped, "unmatched_names": unmatched_names}
