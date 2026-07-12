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
