"""Point-in-time tests for the round-6 pitch-level v2 machinery.

The new as-of tables use UNSHIFTED windows (the row's own game included) and
rely on the allow_exact_matches=False date join for the strictly-prior
guarantee -- these tests pin exactly that: a same-day row never sees itself,
a next-day row sees yesterday, April never sees last September (season
bound), and the cumulative-prior season lookups never see the query season.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import features as F  # noqa: E402


def _ctx_like(**tables):
    """A bare Context with only the v2 tables set (skip __init__ disk IO)."""
    ctx = F.Context.__new__(F.Context)
    ctx._framing2 = tables.get("framing2", pd.DataFrame())
    ctx._platoon2 = tables.get("platoon2", pd.DataFrame())
    ctx._batvelo2 = tables.get("batvelo2", pd.DataFrame())
    ctx._mix2 = tables.get("mix2", pd.DataFrame())
    return ctx


def test_framing_lookup_is_strictly_prior_and_cumulative():
    t = pd.DataFrame({
        "player_id": [7, 7, 7],
        "season": [2021, 2022, 2023],
        "shadow_n": [1000.0, 1000.0, 1000.0],
        "cs_resid": [30.0, 10.0, -40.0],
    })
    ctx = _ctx_like(framing2=t)
    got = ctx.framing2_lookup([2021, 2022, 2023, 2024], [7, 7, 7, 7])
    assert np.isnan(got[0])                      # no prior season
    # 2022 sees only 2021: (30/1000) * (1000/2000)
    assert abs(got[1] - 0.03 * 0.5) < 1e-12
    # 2023 sees 2021+2022 cumulated: (40/2000) * (2000/3000)
    assert abs(got[2] - 0.02 * (2000 / 3000)) < 1e-12
    # 2024 sees all three: sum resid 0 -> exactly 0, not NaN
    assert abs(got[3]) < 1e-12


def test_platoon2_lookup_never_sees_query_season():
    t = pd.DataFrame({
        "player_id": [5, 5, 5, 5],
        "p_throws": ["L", "R", "L", "R"],
        "season": [2022, 2022, 2023, 2023],
        "pa": [100.0, 300.0, 100.0, 300.0],
        "so": [30.0, 60.0, 10.0, 60.0],
        "bb": [10.0, 30.0, 10.0, 30.0],
        "hr": [3.0, 9.0, 3.0, 9.0],
        "h": [25.0, 75.0, 25.0, 75.0],
    })
    ctx = _ctx_like(platoon2=t)
    r23 = ctx.platoon2_lookup([2023], [5], ["L"])
    r24 = ctx.platoon2_lookup([2024], [5], ["L"])
    # 2023 must reflect ONLY 2022 (so 30/100 vs L feeds the delta); if the
    # 2023 rows leaked in, the vs-L K rate would drop to 40/200.
    assert not np.isnan(r23["pp2_k"][0])
    assert r23["pp2_k"][0] != r24["pp2_k"][0]
    # Unknown hand -> NaN.
    assert np.isnan(ctx.platoon2_lookup([2023], [5], [None])["pp2_k"][0])


def test_pergame_asof_join_same_day_and_season_bound():
    table = pd.DataFrame({
        "player_id": [9, 9, 9],
        "season": [2023, 2023, 2024],
        "_d": pd.to_datetime(["2023-06-01", "2023-06-05", "2024-04-10"]),
        "spin_drop": [-50.0, -80.0, -10.0],
        "rel_drift": [0.1, 0.2, 0.05],
    })
    rows = pd.DataFrame({
        "personId": [9, 9, 9, 9],
        "season": [2023, 2023, 2024, 2024],
        "officialDate": ["2023-06-05", "2023-06-06", "2024-04-05", "2024-04-11"],
    })
    out = F._join_pergame_asof(rows.copy(), table, ["spin_drop", "rel_drift"])
    # Same-day row must NOT see its own date's table row -> sees 06-01.
    assert out.loc[0, "spin_drop"] == -50.0
    # Next day sees 06-05.
    assert out.loc[1, "spin_drop"] == -80.0
    # April 2024 before any 2024 start: NaN, NOT September 2023 (season bound).
    assert np.isnan(out.loc[2, "spin_drop"])
    assert out.loc[3, "spin_drop"] == -10.0


def test_relspin_asof_gates_and_excludes_nothing_before_min_starts():
    rows = []
    for i, d in enumerate(["2024-04-01", "2024-04-06", "2024-04-11",
                           "2024-04-16"]):
        rows.append({"player_id": 3, "season": 2024, "game_pk": 100 + i,
                     "game_date": d, "n_fb": 40.0,
                     "relx_sum": 40.0 * (-1.5), "relz_sum": 40.0 * 5.8,
                     "spin_sum": 40.0 * 2300.0, "spin_n": 40.0})
    import mlblib.cache as C
    df = pd.DataFrame(rows)
    orig = C.read_parquet_or_none
    C.read_parquet_or_none = lambda p: df if "relspin2_2024" in str(p) else None
    try:
        t = F._relspin_asof_table([2024])
    finally:
        C.read_parquet_or_none = orig
    # First start: expanding min_periods=2 unmet -> NaN. Second start: 80 FB
    # in both windows, identical release point -> drift exactly 0.
    first = t[t["_d"] == pd.Timestamp("2024-04-01")]
    second = t[t["_d"] == pd.Timestamp("2024-04-06")]
    assert np.isnan(first["rel_drift"].iloc[0])
    assert abs(second["rel_drift"].iloc[0]) < 1e-9
    assert abs(second["spin_drop"].iloc[0]) < 1e-9
