"""Point-in-time / leakage tests for mlblib.features.

The load-bearing guarantee of the whole project is that no feature for a target
game uses information from that game or later. These tests re-derive the
season-to-date aggregates the slow, obviously-correct way and assert equality,
plus check the doubleheader ordering convention and the zero-history call-up.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, features as F  # noqa: E402


def _synthetic_history():
    rows = []

    def row(pid, pk, season, gdate, gnum, PA, HR):
        r = {c: 0 for c in F.BAT_COUNTS}
        r.update(dict(personId=pid, gamePk=pk, season=season, gameDate=gdate,
                      gameNumber=gnum, officialDate=gdate[:10], played=True,
                      is_batter=True, is_pitcher=False, is_sp=False, isHome=True,
                      position="2B", is_starter_slot=True, teamId=1, oppTeamId=2,
                      side="home", venue_id=1, slot=3, oppStarterId=None,
                      oppStarterHand="R", batSide="L", PA=PA, HR=HR, AB=PA, H=HR, TB=HR))
        rows.append(r)

    # 2020 prior season, 2021 target season incl. a doubleheader on 05-02.
    row(1, 900, 2020, "2020-08-01T18:00:00Z", 1, 4, 1)
    row(1, 901, 2020, "2020-08-02T18:00:00Z", 1, 4, 1)
    row(1, 1000, 2021, "2021-05-01T18:00:00Z", 1, 5, 0)
    row(1, 1001, 2021, "2021-05-02T18:00:00Z", 1, 4, 1)   # DH game 1 (earlier)
    row(1, 1002, 2021, "2021-05-02T22:05:00Z", 2, 3, 0)   # DH game 2 (later)
    return pd.DataFrame(rows)


def test_season_to_date_shift_no_leak():
    hist = _synthetic_history()
    comb = hist.sort_values(["personId", "gameDate", "gameNumber"]).reset_index(drop=True)
    comb = F._add_asof_aggregates(comb, F.BAT_COUNTS, "b_")
    g = comb.set_index("gamePk")
    assert g.loc[1000, "b_std_PA"] == 0            # first game of 2021 sees nothing
    assert g.loc[1001, "b_std_PA"] == 5            # sees game 1000 only
    assert g.loc[1002, "b_std_PA"] == 9            # DH game 2 sees 1000 + 1001
    assert g.loc[1002, "b_std_HR"] == 1            # 0 + 1


def test_doubleheader_game2_sees_game1():
    hist = _synthetic_history()
    comb = hist.sort_values(["personId", "gameDate", "gameNumber"]).reset_index(drop=True)
    comb = F._add_asof_aggregates(comb, F.BAT_COUNTS, "b_")
    g = comb.set_index("gamePk")
    assert g.loc[1002, "b_rest_days"] == 0         # same calendar day as game 1


def test_marcel_uses_only_prior_seasons():
    hist = _synthetic_history()
    league = F._league_rates(hist[hist.is_batter], F.BAT_COUNTS, "PA")
    marcel = F._marcel_prior_rates(hist, F.BAT_COUNTS, "PA", F.MARCEL_HIT_WEIGHTS,
                                   F.HITTER_BALLAST_PA, league)
    m2020 = marcel[(marcel.personId == 1) & (marcel.season == 2020)].iloc[0]
    # 2020 has no prior season -> league fallback, never NaN.
    assert np.isclose(m2020["marcel_HR"], league["HR"])
    m2021 = marcel[(marcel.personId == 1) & (marcel.season == 2021)].iloc[0]
    # PA/game prior from 2020 = 8 PA / 2 games = 4.0
    assert np.isclose(m2021["marcel_PApg"], 4.0)


def test_callup_zero_history_is_league_not_nan():
    hist = _synthetic_history()
    league = F._league_rates(hist[hist.is_batter], F.BAT_COUNTS, "PA")
    marcel = F._marcel_prior_rates(hist, F.BAT_COUNTS, "PA", F.MARCEL_HIT_WEIGHTS,
                                   F.HITTER_BALLAST_PA, league)
    # Season 2020, player 1 has no prior -> marcel equals league fallback.
    m = marcel[(marcel.personId == 1) & (marcel.season == 2020)].iloc[0]
    assert not np.isnan(m["marcel_HR"])


# ── Real-data leakage test (skips if the backfill has not produced a season) ──
def _available_season():
    for s in (2024, 2023, 2022, 2021, 2019):
        if cache.dc_path(f"gamelogs_{s}_v1.parquet").exists():
            return s
    return None


@pytest.mark.parametrize("_", [0])
def test_realdata_no_leakage(_):
    season = _available_season()
    if season is None:
        pytest.skip("no gamelogs parquet available yet")
    logs = F.load_gamelogs([season])
    bat = logs[logs["played"] & logs["is_batter"]].copy()
    comb = bat.sort_values(["personId", "gameDate", "gameNumber"]).reset_index(drop=True)
    comb = F._add_asof_aggregates(comb, F.BAT_COUNTS, "b_")

    rng = np.random.RandomState(7)
    sample = comb.sample(min(200, len(comb)), random_state=rng)
    # Re-derive season-to-date PA and HR the slow way and compare.
    by_player = {pid: sub for pid, sub in bat.groupby("personId")}
    for _, trow in sample.iterrows():
        pid = trow["personId"]
        sub = by_player[pid]
        key = (trow["gameDate"], trow["gameNumber"])
        prior = sub[sub.apply(lambda r: (r["gameDate"], r["gameNumber"]) < key
                              and r["season"] == trow["season"], axis=1)]
        exp_pa = prior["PA"].sum()
        exp_hr = prior["HR"].sum()
        assert trow["b_std_PA"] == exp_pa, f"PA leak for {pid} {trow['gamePk']}"
        assert trow["b_std_HR"] == exp_hr, f"HR leak for {pid} {trow['gamePk']}"
