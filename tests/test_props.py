"""Tests for the PrizePicks line-comparison engine (mlblib.props)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import props  # noqa: E402


def test_normalize_name_folds_accents_periods_suffixes():
    assert props.normalize_name("José Ramírez") == "jose ramirez"
    assert props.normalize_name("J.T. Realmuto") == "j t realmuto"
    assert props.normalize_name("Ronald Acuña Jr.") == "ronald acuna"
    assert props.normalize_name("  Mike   Trout ") == "mike trout"


def _preds():
    return pd.DataFrame([
        {"fullName": "Ketel Marte", "role": "bat", "TB": 2.0, "H": 1.1,
         "HR": 0.25, "SB": 0.06, "SO": 0.8, "BB": 0.35, "R": 0.6, "RBI": 0.5,
         "b1": 0.6, "b2": 0.2, "b3": 0.01},
        {"fullName": "Zac Gallen", "role": "pit", "K": 6.0, "H": 5.5, "BB": 1.6,
         "Pitches": 92.0, "ER": 2.8, "IP": 5.6},
    ])


def test_compare_resolves_role_and_computes_edge():
    lines = pd.DataFrame([
        {"name": "Ketel Marte", "stat_type": "Total Bases", "line": 1.5},
        {"name": "Zac Gallen", "stat_type": "Pitcher Strikeouts", "line": 6.5},
        {"name": "Zac Gallen", "stat_type": "Hits Allowed", "line": 5.0},
        {"name": "Nobody Here", "stat_type": "Hits", "line": 0.5},
    ])
    table, meta = props.compare(lines, _preds())
    assert meta["matched"] == 3 and meta["unmatched"] == 1
    marte = table[table["Player"] == "Ketel Marte"].iloc[0]
    assert marte["Model"] == 2.0 and marte["Line"] == 1.5
    assert marte["Edge"] == 0.5 and marte["Lean"] == "Over"
    # Gallen's "Strikeouts"/"Hits Allowed" must resolve to PITCHER columns.
    gk = table[table["Stat"] == "Pitcher K"].iloc[0]
    assert gk["Model"] == 6.0 and gk["Lean"] == "Under"
    gh = table[table["Stat"] == "Hits Allowed"].iloc[0]
    assert gh["Model"] == 5.5
    # Sorted by absolute disagreement (Gallen K edge -0.5 is the biggest).
    assert abs(table.iloc[0]["Edge"]) >= abs(table.iloc[-1]["Edge"])


def test_composite_stat_sums_columns():
    lines = pd.DataFrame([
        {"name": "Ketel Marte", "stat_type": "Hits+Runs+RBIs", "line": 2.0},
    ])
    table, _ = props.compare(lines, _preds())
    # H + R + RBI = 1.1 + 0.6 + 0.5 = 2.2
    assert abs(table.iloc[0]["Model"] - 2.2) < 1e-9


def test_save_load_lines_roundtrip(tmp_path, monkeypatch):
    from mlblib import cache
    monkeypatch.setattr(cache, "DATA_CACHE", tmp_path, raising=False)
    monkeypatch.setattr(props.cache, "dc_path",
                        lambda name: tmp_path / name)
    lines = pd.DataFrame([
        {"name": "Ketel Marte", "team": "AZ", "stat_type": "Total Bases",
         "line": 1.5, "start_time": None},
    ])
    assert props.load_lines("2026-07-11") is None      # nothing saved yet
    props.save_lines("2026-07-11", lines)
    back = props.load_lines("2026-07-11")
    assert back is not None and len(back) == 1
    assert back["line"].iloc[0] == 1.5
    assert back.attrs.get("saved_at")                  # freshness stamp present
    # Re-saving identical content keeps the original timestamp.
    first = back.attrs["saved_at"]
    props.save_lines("2026-07-11", lines)
    assert props.load_lines("2026-07-11").attrs["saved_at"] == first
    # Empty / None input is a no-op (no crash, no file churn).
    props.save_lines("2026-07-11", pd.DataFrame())
    props.save_lines("2026-07-11", None)


def test_saved_at_et_formats_and_tolerates_junk():
    assert props.saved_at_et(None) == ""
    assert props.saved_at_et("not-a-date") == ""
    out = props.saved_at_et("2026-07-12T06:55:00+00:00")
    assert out.endswith("ET") and ":" in out


def test_expandable_table_only_marks_players_with_props():
    from mlblib import store
    df = pd.DataFrame([
        {"fullName": "Ketel Marte", "role": "bat", "slot": 1, "PA": 4.0,
         "b1": .6, "b2": .2, "b3": .0, "HR": .2, "BB": .3, "HBP": 0, "SO": .8,
         "R": .6, "RBI": .5, "SB": .1, "H": 1.0, "TB": 2.0},
        {"fullName": "Alek Thomas", "role": "bat", "slot": 2, "PA": 3.5,
         "b1": .5, "b2": .1, "b3": .0, "HR": .1, "BB": .2, "HBP": 0, "SO": .7,
         "R": .4, "RBI": .3, "SB": .1, "H": .8, "TB": 1.2},
    ])
    pbn = {"Ketel Marte": [{"Stat": "Total Bases", "Model": 2.0, "Line": 1.5,
                           "Edge": 0.5, "Lean": "Over"}]}
    html = store.html_expandable_batter_table(df, pbn)
    assert html.count("<details") == 1        # only Marte is expandable
    assert "norow" in html                    # Thomas stays a plain row
    assert "Total Bases" in html and "Over" in html
    assert 'class="xcaret has"' in html        # the teal count badge


def test_parse_prizepicks_board_text():
    board = (
        "Trending\n30.8K\nJames Wood\nWSH - OF\nJames Wood\nvs NYY 47m 24s\n\n"
        "7.5\nHitter FS\nLess\nMore\n"
        "Trending\n30.8K\nJames WoodDemon\nWSH - OF\nJames Wood\n"
        "vs NYY 47m 24s\n1.5\nTB\nMore\n"
        "Trending\n13.4K\nCorbin CarrollDemon\nAZ - OF\nCorbin Carroll\n"
        "@ LAD Sun 1:10pm\n0.5\nWalks\nMore\n"
        "Trending\n14.5K\nTarik Skubal\nDET - P\nTarik Skubal\n"
        "vs PHI 52m 24s\n7.5\nKs\nLess\nMore\n")
    df = props.parse_prizepicks_board(board)
    # Fantasy Score is skipped; Demon tags stripped; TB/Ks normalized.
    got = {(r["name"], r["stat_type"], r["line"]) for _, r in df.iterrows()}
    assert ("James Wood", "Total Bases", 1.5) in got
    assert ("Corbin Carroll", "Walks", 0.5) in got
    assert ("Tarik Skubal", "Strikeouts", 7.5) in got
    assert all(s != "Hitter FS" for _, s, _ in got)    # fantasy score dropped
    assert all("Demon" not in n for n, _, _ in got)    # payout tag stripped
    # parse_any routes board text to the board parser.
    assert len(props.parse_any(board)) == len(df) >= 3


def test_board_captures_direction_and_odds():
    board = (
        "Trending\n38.8K\nJames WoodGoblin\nWSH - OF\nJames Wood\n"
        "WSH 1vs NYY 0\nCur. 4\n4.5\nTB\nMore\n"
        "Trending\n20.4K\nTarik SkubalDemon\nDET - P\nTarik Skubal\n"
        "DET 0vs PHI 0\nCur. 2\n7.5\nKs\nMore\n"
        "Trending\n7.2K\nCasey Schmitt\nSF - OF\nCasey Schmitt\n"
        "vs COL Sun 1:05pm\n1.5\nTB\nLess\nMore\n")
    df = props.parse_prizepicks_board(board)
    got = {r["name"]: (r["direction"], r["odds_type"]) for _, r in df.iterrows()}
    assert got["James Wood"] == ("more", "goblin")
    assert got["Tarik Skubal"] == ("more", "demon")     # Demon = one-sided More
    assert got["Casey Schmitt"] == ("both", "standard")  # standard = both sides


def test_compare_carries_direction_and_flags_unoffered_side():
    from mlblib import store
    lines = pd.DataFrame([
        {"name": "Zac Gallen", "stat_type": "Pitcher Strikeouts", "line": 7.5,
         "direction": "more", "odds_type": "demon"},
    ])
    table, _ = props.compare(lines, _preds())
    r = table.iloc[0]
    assert r["Direction"] == "more" and r["OddsType"] == "demon"
    # Gallen model K = 6.0 vs line 7.5 -> Under; only More offered -> flag.
    meta = store._props_meta(dict(r))
    assert "More only" in meta and "Demon" in meta
    assert "Under side not offered" in meta


def test_props_meta_no_flag_when_side_offered():
    from mlblib import store
    p = {"Stat": "TB", "Model": 2.0, "Line": 1.5, "Edge": 0.5, "Lean": "Over",
         "Direction": "both", "OddsType": "standard"}
    meta = store._props_meta(p)
    assert "More &amp; Less" in meta and "not offered" not in meta


def test_merge_lines_accumulates_and_dedups():
    a = pd.DataFrame([
        {"name": "Ketel Marte", "stat_type": "Total Bases", "line": 1.5},
        {"name": "Zac Gallen", "stat_type": "Pitcher Strikeouts", "line": 6.5},
    ])
    b = pd.DataFrame([
        {"name": "Ketel Marte", "stat_type": "Total Bases", "line": 2.5},   # moved
        {"name": "Mookie Betts", "stat_type": "Hits", "line": 0.5},
    ])
    m = props.merge_lines(a, b)
    keyed = {(r["name"], r["stat_type"]): r["line"] for _, r in m.iterrows()}
    assert len(m) == 3                                    # one dup collapsed
    assert keyed[("Ketel Marte", "Total Bases")] == 2.5   # newest paste wins
    assert ("Zac Gallen", "Pitcher Strikeouts") in keyed  # first tab kept
    assert ("Mookie Betts", "Hits") in keyed              # second tab added
    assert len(props.merge_lines(None, b)) == 2           # None operands
    assert len(props.merge_lines(a, None)) == 2
    assert props.merge_lines(None, None).empty


def test_clear_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(props.cache, "dc_path", lambda name: tmp_path / name)
    props.save_lines("2026-07-12", pd.DataFrame([
        {"name": "X", "stat_type": "Hits", "line": 0.5}]))
    assert props.load_lines("2026-07-12") is not None
    props.clear_lines("2026-07-12")
    assert props.load_lines("2026-07-12") is None
    props.clear_lines("2026-07-12")   # idempotent, no crash


def _bat(game, name, **over):
    row = {"gamePk": game, "fullName": name, "role": "bat", "TB": 2.0, "H": 1.0,
           "HR": 0.3, "R": 0.6, "RBI": 0.5, "BB": 0.3, "SO": 0.8, "SB": 0.1,
           "b1": 0.6, "b2": 0.2, "b3": 0.0, "PA": 4.0, "slot": 1}
    row.update(over)
    return row


def test_line_counts_by_game_sums_both_teams(monkeypatch):
    import props_ui
    from mlblib import props as _props
    # Game 1: two Dodgers lines (Betts TB + Hits) and one D-backs line (Marte TB)
    # -> 3. Game 2: one Yankees line -> 1. A line for a player not on the slate
    # is not counted.
    preds = pd.DataFrame([
        _bat(1, "Mookie Betts"), _bat(1, "Freddie Freeman"), _bat(1, "Ketel Marte"),
        _bat(2, "Aaron Judge"),
    ])
    lines = pd.DataFrame([
        {"name": "Mookie Betts", "stat_type": "Total Bases", "line": 1.5},
        {"name": "Mookie Betts", "stat_type": "Hits", "line": 0.5},
        {"name": "Ketel Marte", "stat_type": "Total Bases", "line": 1.5},
        {"name": "Aaron Judge", "stat_type": "Home Runs", "line": 0.5},
        {"name": "Nobody Here", "stat_type": "Hits", "line": 0.5},
    ])
    monkeypatch.setattr(_props, "load_lines", lambda d: lines)
    assert props_ui.line_counts_by_game(preds, "2026-07-12") == {1: 3, 2: 1}


def test_line_counts_by_game_empty_when_no_lines(monkeypatch):
    import props_ui
    from mlblib import props as _props
    preds = pd.DataFrame([_bat(1, "Mookie Betts")])
    monkeypatch.setattr(_props, "load_lines", lambda d: None)
    assert props_ui.line_counts_by_game(preds, "2026-07-12") == {}
    # No gamePk column -> no crash, empty result.
    monkeypatch.setattr(_props, "load_lines",
                        lambda d: pd.DataFrame([{"name": "Mookie Betts",
                                                 "stat_type": "Hits", "line": 0.5}]))
    assert props_ui.line_counts_by_game(preds.drop(columns="gamePk"),
                                        "2026-07-12") == {}


def test_parse_json_and_list():
    payload = {
        "data": [{"type": "projection", "attributes": {
            "stat_type": "Total Bases", "line_score": 1.5},
            "relationships": {"new_player": {"data": {"id": "1"}}}}],
        "included": [{"type": "new_player", "id": "1",
                      "attributes": {"name": "Ketel Marte", "team": "AZ"}}],
    }
    j = props.parse_prizepicks_json(payload)
    assert len(j) == 1 and j.iloc[0]["name"] == "Ketel Marte"
    lst = props.parse_line_list("Ketel Marte | Total Bases | 1.5\nZac Gallen, Strikeouts, 6")
    assert len(lst) == 2 and lst.iloc[1]["line"] == 6.0
