"""Point-in-time feature construction. Shared by training and inference.

THE #1 CORRECTNESS RULE: every feature for a target game is computed strictly
from information available BEFORE that game's first pitch. Concretely, an as-of
aggregate over a player's history uses only rows whose (gameDate, gameNumber)
sort strictly before the target row's. Season-level reference tables (park
factors, catcher metrics, platoon splits) use PRIOR seasons only. The leakage
test in tests/ re-derives features the slow, obviously-correct way and asserts
equality; if you change anything here, run it.

One code path serves both uses:
  - Training:  compute_batter_features(history) with no extra targets — every
    played row in `history` gets its point-in-time feature vector.
  - Inference: compute_batter_features(history, targets=slate_rows) — the slate
    rows (which carry no stats, they are in the future) get features from the
    real prior games in `history`, and contribute nothing to anyone else.

The trick that makes this safe: we stack history + targets, sort by
(personId, gameDate, gameNumber), and every rolling/expanding aggregate is
SHIFTED by one within the player, so a row never sees itself. Target rows carry
the latest gameDate, so their shifted aggregate is exactly "all prior real
games". Training rows are real, and the shift stops them seeing themselves.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import cache

# ── League-constant knobs (documented in docs/decisions.md) ──────────────────
HITTER_BALLAST_PA = 1200.0
PITCHER_BALLAST_BF = 400.0
MARCEL_HIT_WEIGHTS = [5, 4, 3]   # most-recent prior season first
MARCEL_PIT_WEIGHTS = [3, 2, 1]
# Stabilization-point k for shrinking within-season rates toward the prior.
STAB_K = {"SO": 60.0, "BB": 120.0, "HR": 170.0, "hit": 460.0, "HBP": 460.0}
PIT_STAB_K = {"p_K": 90.0, "p_BB": 170.0, "p_HR": 250.0, "p_H": 250.0}

# Batter component counting columns and the per-PA rate targets derived from them
BAT_COUNTS = ["PA", "AB", "H", "b1", "b2", "b3", "HR", "BB", "HBP", "SO",
              "R", "RBI", "SB", "TB", "SF"]
BAT_RATE_COMPONENTS = ["b1", "b2", "b3", "HR", "BB", "HBP", "SO"]  # per PA
PIT_COUNTS = ["p_outs", "p_BF", "p_K", "p_BB", "p_H", "p_HR", "p_ER", "p_R",
              "p_pitches"]
PIT_RATE_COMPONENTS = ["p_K", "p_BB", "p_H", "p_HR"]  # per BF


# ── Loading ──────────────────────────────────────────────────────────────────
def load_gamelogs(seasons: list[int]) -> pd.DataFrame:
    """Concatenate per-season gamelog parquets. Missing seasons are skipped."""
    frames = []
    for s in seasons:
        p = cache.dc_path(f"gamelogs_{s}_v1.parquet")
        df = cache.read_parquet_or_none(p)
        if df is not None:
            df["season"] = s
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # A stable within-player order for as-of aggregation.
    out["gameDate"] = out["gameDate"].astype(str)
    out["officialDate"] = out["officialDate"].astype(str)
    return out


def attach_catchers(history: pd.DataFrame) -> pd.DataFrame:
    """Add ownCatcherId (the player's own team's starting catcher for the game)
    and oppCatcherId (the opposing team's starting catcher). The starting
    catcher is the row with position 'C' and is_starter_slot on that (gamePk,
    side). Point-in-time safe: this is same-game roster info, known at lineup
    posting, not derived from outcomes.
    """
    catchers = history[(history["position"] == "C") & history["is_starter_slot"]]
    key = catchers.groupby(["gamePk", "side"])["personId"].first()
    key = key.reset_index().rename(columns={"personId": "catcherId"})
    home_c = key[key["side"] == "home"][["gamePk", "catcherId"]].rename(
        columns={"catcherId": "home_c"})
    away_c = key[key["side"] == "away"][["gamePk", "catcherId"]].rename(
        columns={"catcherId": "away_c"})
    h = history.merge(home_c, on="gamePk", how="left").merge(away_c, on="gamePk", how="left")
    h["ownCatcherId"] = np.where(h["side"] == "home", h["home_c"], h["away_c"])
    h["oppCatcherId"] = np.where(h["side"] == "home", h["away_c"], h["home_c"])
    return h.drop(columns=["home_c", "away_c"])


class Context:
    """Season-level reference tables, loaded once and queried with the Y-1 rule."""

    # Statcast quality-of-contact tables: {short name: filename stem}. Loaded
    # per year and queried with the same Y-1 prior-year rule as park factors.
    STATCAST_TABLES = {
        "xbat": "statcast_xstats_bat",
        "xpit": "statcast_xstats_pit",
        "evbat": "statcast_ev_bat",
        "evpit": "statcast_ev_pit",
        "sprint": "sprint_speed",
        "discpit": "disc_pit",
        "discbat": "disc_bat",
        "arspit": "arsenal_pit",
        "arsbat": "arsenal_bat",
    }

    def __init__(self, years: list[int]):
        self.park: dict[int, pd.DataFrame] = {}
        self.framing: dict[int, pd.DataFrame] = {}
        self.throwing: dict[int, pd.DataFrame] = {}
        self.statcast: dict[str, dict[int, pd.DataFrame]] = {
            k: {} for k in self.STATCAST_TABLES}
        for y in years:
            pk = cache.read_parquet_or_none(cache.dc_path(f"park_factors_{y}_v1.parquet"))
            if pk is not None:
                self.park[y] = pk
            fr = cache.read_parquet_or_none(cache.dc_path(f"catcher_framing_{y}_v1.parquet"))
            if fr is not None:
                self.framing[y] = fr
            th = cache.read_parquet_or_none(cache.dc_path(f"catcher_throwing_{y}_v1.parquet"))
            if th is not None:
                self.throwing[y] = th
            for key, stem in self.STATCAST_TABLES.items():
                sc = cache.read_parquet_or_none(cache.dc_path(f"{stem}_{y}_v1.parquet"))
                if sc is not None:
                    self.statcast[key][y] = sc
        self._years = sorted(self.park.keys())
        self._park_all = self._concat(self.park)
        self._framing_all = self._concat(self.framing)
        self._throwing_all = self._concat(self.throwing)
        self._statcast_all = {k: self._concat(v) for k, v in self.statcast.items()}
        # Per-game environment (parsed from cached boxscores) + venue geometry.
        gw = cache.read_parquet_or_none(cache.dc_path("game_weather_v1.parquet"))
        self.game_weather = gw if gw is not None else pd.DataFrame()
        vn = cache.read_parquet_or_none(cache.dc_path("venues_v1.parquet"))
        self.venues = vn if vn is not None else pd.DataFrame()

    def statcast_lookup(self, seasons, person_ids, which: str, col: str) -> np.ndarray:
        """Vectorized Y-1 lookup of one Statcast column for many players.

        `which` is a STATCAST_TABLES key; `col` a column of that table. Rows
        whose player is absent from the prior-year table return NaN (HistGB
        routes missing natively).
        """
        src = self._statcast_all.get(which)
        n = len(seasons)
        if src is None or src.empty or col not in src.columns:
            return np.full(n, np.nan)
        years = sorted(self.statcast[which].keys())
        pyr = {s: max((y for y in years if y < s), default=None)
               for s in pd.unique(seasons)}
        key = pd.DataFrame({
            "_i": np.arange(n),
            "season": [pyr.get(s) for s in seasons],
            "player_id": pd.to_numeric(pd.Series(person_ids), errors="coerce"),
        })
        ref = src[["season", "player_id", col]].rename(columns={col: "_val"})
        ref = ref.drop_duplicates(subset=["season", "player_id"])
        merged = key.merge(ref, on=["season", "player_id"], how="left").sort_values("_i")
        return pd.to_numeric(merged["_val"], errors="coerce").values

    def arsenal_matchup(self, seasons, batter_ids, pitcher_ids) -> np.ndarray:
        """Expected-whiff matchup: sum over the pitcher's Y-1 pitch mix of
        (usage share x the batter's Y-1 whiff rate against that pitch type),
        with the league-average whiff for a pitch type filling batter gaps.
        The round's headline feature: 'this lineup chases sliders, this
        starter is slider-heavy'. Returns NaN when the pitcher has no Y-1
        arsenal data.
        """
        pit = self._statcast_all.get("arspit")
        bat = self._statcast_all.get("arsbat")
        n = len(seasons)
        if pit is None or pit.empty or bat is None or bat.empty:
            return np.full(n, np.nan)
        years = sorted(self.statcast["arspit"].keys())
        pyr = {s: max((y for y in years if y < s), default=None)
               for s in pd.unique(seasons)}
        key = pd.DataFrame({
            "_i": np.arange(n),
            "season": [pyr.get(s) for s in seasons],
            "pid": pd.to_numeric(pd.Series(pitcher_ids), errors="coerce"),
            "bid": pd.to_numeric(pd.Series(batter_ids), errors="coerce"),
        })
        # Pitcher usage shares per pitch type (Y-1 file).
        p = pit[["season", "player_id", "pitch_type", "pitch_usage"]].rename(
            columns={"player_id": "pid", "pitch_usage": "usage"})
        expanded = key.merge(p, on=["season", "pid"], how="left")
        # Batter whiff vs that pitch type, league fallback per (season, type).
        b = bat[["season", "player_id", "pitch_type", "whiff_percent"]].rename(
            columns={"player_id": "bid", "whiff_percent": "b_whiff"})
        lg = (bat.assign(w=bat["whiff_percent"] * bat["pitches"])
              .groupby(["season", "pitch_type"])
              .apply(lambda g: g["w"].sum() / max(g["pitches"].sum(), 1),
                     include_groups=False)
              .rename("lg_whiff").reset_index())
        expanded = expanded.merge(b, on=["season", "bid", "pitch_type"], how="left")
        expanded = expanded.merge(lg, on=["season", "pitch_type"], how="left")
        expanded["eff_whiff"] = expanded["b_whiff"].fillna(expanded["lg_whiff"])
        expanded["contrib"] = (expanded["usage"].astype(float) / 100.0
                               * expanded["eff_whiff"].astype(float))
        agg = expanded.groupby("_i").agg(mu=("contrib", "sum"),
                                         got=("usage", "count"))
        out = np.full(n, np.nan)
        ok = agg[agg["got"] > 0]
        out[ok.index.values] = ok["mu"].values
        return out

    @staticmethod
    def _concat(table: dict) -> pd.DataFrame:
        return pd.concat(table.values(), ignore_index=True) if table else pd.DataFrame()

    def prior_year_map(self, seasons, which: str) -> dict:
        """For each season, the latest available reference year strictly before
        it (the Y-1 rule with graceful fallback to the nearest prior year)."""
        table = {"park": self.park, "framing": self.framing, "throwing": self.throwing}[which]
        years = sorted(table.keys())
        out = {}
        for s in seasons:
            cand = [y for y in years if y < s]
            out[s] = max(cand) if cand else None
        return out

    def park_lookup(self, seasons, venue_ids, bat_sides) -> pd.DataFrame:
        """Vectorized: return a frame (index-aligned to inputs) of park index_*
        columns using the Y-1 file and the batter's hand (falling back to 'all').
        """
        pcols = ["index_hr", "index_so", "index_bb", "index_woba",
                 "index_1b", "index_2b", "index_3b"]
        n = len(seasons)
        if self._park_all.empty:
            return pd.DataFrame({c: [np.nan] * n for c in pcols})
        pyr = self.prior_year_map(pd.unique(seasons), "park")
        side = pd.Series(bat_sides).where(pd.Series(bat_sides).isin(["L", "R"]), "all").values
        key = pd.DataFrame({
            "_i": np.arange(n),
            "year": [pyr.get(s) for s in seasons],
            "venue_id": venue_ids,
            "bat_side": side,
        })
        merged = key.merge(self._park_all, on=["year", "venue_id", "bat_side"], how="left")
        # Fallback to 'all' where the hand-specific row was missing.
        miss = merged[pcols[0]].isna()
        if miss.any():
            allside = self._park_all[self._park_all["bat_side"] == "all"]
            fb = key[miss.values].drop(columns=["bat_side"]).merge(
                allside.drop(columns=["bat_side"]), on=["year", "venue_id"], how="left")
            for c in pcols:
                merged.loc[miss.values, c] = fb[c].values
        return merged.sort_values("_i")[pcols].reset_index(drop=True)

    def catcher_lookup(self, seasons, catcher_ids, kind: str) -> np.ndarray:
        """Vectorized framing rv_tot or throwing runs via the Y-1 file."""
        if kind == "framing":
            src, idcol, valcol = self._framing_all, "id", "rv_tot"
            which = "framing"
        else:
            src, idcol, valcol = self._throwing_all, "player_id", "catcher_stealing_runs"
            which = "throwing"
        n = len(seasons)
        if src.empty or valcol not in src.columns:
            return np.full(n, np.nan)
        pyr = self.prior_year_map(pd.unique(seasons), which)
        key = pd.DataFrame({
            "_i": np.arange(n),
            "season": [pyr.get(s) for s in seasons],
            "cid": catcher_ids,
        })
        ref = src[[  # rename to the join keys
            "season", idcol, valcol]].rename(columns={idcol: "cid", valcol: "_val"})
        merged = key.merge(ref, on=["season", "cid"], how="left").sort_values("_i")
        return merged["_val"].values


# ── Point-in-time player aggregates ──────────────────────────────────────────
def _add_asof_aggregates(df: pd.DataFrame, counts: list[str], prefix: str) -> pd.DataFrame:
    """Add season-to-date and rolling last-15/last-30 SHIFTED sums per player.

    df must be sorted by (personId, gameDate, gameNumber). Every output column
    is shifted one row within the player, so a row sees only its prior games.
    """
    # Fill NaN counts with 0 first: inference target rows carry NaN counts (the
    # game is in the future). Without this the cumsum would turn NaN at the
    # target row and poison its own season-to-date value.
    filled = {c: df[c].fillna(0) for c in counts}
    fdf = df.assign(**{f"_f_{c}": filled[c] for c in counts})
    g = fdf.groupby("personId", sort=False)
    gs = fdf.groupby(["personId", "season"], sort=False)
    for c in counts:
        fc = f"_f_{c}"
        # Season-to-date: cumulative within season, shifted (excludes this row).
        df[f"{prefix}std_{c}"] = gs[fc].cumsum() - fdf[fc]
        # Rolling last-15 / last-30 appearances, shifted by one.
        sh = g[fc].shift(1)
        df[f"{prefix}r15_{c}"] = sh.groupby(fdf["personId"], sort=False).transform(
            lambda s: s.rolling(15, min_periods=1).sum())
        df[f"{prefix}r30_{c}"] = sh.groupby(fdf["personId"], sort=False).transform(
            lambda s: s.rolling(30, min_periods=1).sum())
    # Games played to date (season) and rest days.
    df[f"{prefix}std_G"] = gs.cumcount()
    prev_date = g["gameDate"].shift(1)
    df[f"{prefix}rest_days"] = (
        pd.to_datetime(df["gameDate"], errors="coerce")
        - pd.to_datetime(prev_date, errors="coerce")
    ).dt.days
    return df


def _marcel_prior_rates(logs: pd.DataFrame, counts: list[str], denom: str,
                        weights: list[int], ballast: float,
                        league_rate: dict, seasons: list[int] | None = None) -> pd.DataFrame:
    """Per (personId, season) Marcel prior: weighted mean of the player's
    PRIOR-season per-`denom` rates, regressed toward the league rate with
    `ballast` denom-units of league-average ballast.

    `seasons` is the list of target seasons to produce priors FOR. It must
    include any inference season (e.g. 2026) that is not present in `logs`, or
    those target rows would get no prior. Defaults to the seasons in `logs`.

    Returns columns marcel_{c} for each count, plus marcel_{denom}pg (the prior
    opportunities-per-game, e.g. PA/game).
    """
    # Season totals per player.
    agg = logs.groupby(["personId", "season"]).agg(
        {**{c: "sum" for c in counts}, denom: "sum", "gamePk": "nunique"}
    ).reset_index().rename(columns={"gamePk": "G"})
    agg = agg.sort_values(["personId", "season"])

    if seasons is None:
        seasons = sorted(logs["season"].unique())
    rows = []
    by_player = {pid: sub for pid, sub in agg.groupby("personId")}
    for pid, sub in by_player.items():
        sub = sub.set_index("season")
        for target_season in seasons:
            priors = [s for s in sub.index if s < target_season]
            priors = sorted(priors, reverse=True)[:3]
            if not priors:
                row = {"personId": pid, "season": target_season}
                for c in counts:
                    row[f"marcel_{c}"] = league_rate.get(c, np.nan)
                row[f"marcel_{denom}pg"] = league_rate.get(f"{denom}pg", np.nan)
                rows.append(row)
                continue
            w = weights[:len(priors)]
            num_denom = sum(w[i] * sub.loc[priors[i], denom] for i in range(len(priors)))
            g_tot = sum(w[i] * sub.loc[priors[i], "G"] for i in range(len(priors)))
            row = {"personId": pid, "season": target_season}
            for c in counts:
                num_c = sum(w[i] * sub.loc[priors[i], c] for i in range(len(priors)))
                # Regress the rate toward league with `ballast` league-units.
                lr = league_rate.get(c, 0.0)
                rate = (num_c + ballast * lr) / (num_denom + ballast) if (num_denom + ballast) else lr
                row[f"marcel_{c}"] = rate
            # opportunities per game prior
            row[f"marcel_{denom}pg"] = (num_denom / g_tot) if g_tot else league_rate.get(f"{denom}pg", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _league_rates(logs: pd.DataFrame, counts: list[str], denom: str) -> dict:
    tot_denom = logs[denom].sum()
    out = {}
    for c in counts:
        out[c] = (logs[c].sum() / tot_denom) if tot_denom else 0.0
    g = logs["gamePk"].nunique() * 2  # rough games-per-side scale; only for *pg fallback
    played = logs[logs["played"]] if "played" in logs.columns else logs
    out[f"{denom}pg"] = played[denom].mean()
    return out


def _shrink(std_count, std_denom, marcel_rate, k):
    """Blend the within-season rate (std_count/std_denom) toward the Marcel
    prior by sample size: weight = n/(n+k), n = std_denom.
    """
    n = std_denom.fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.where(n > 0, std_count.fillna(0) / n.replace(0, np.nan), np.nan)
    w = n / (n + k)
    return w * pd.Series(obs, index=std_count.index).fillna(marcel_rate) + (1 - w) * marcel_rate


def _age_years(birth: str, on_date: str):
    try:
        b = dt.date.fromisoformat(str(birth)[:10])
        d = dt.date.fromisoformat(str(on_date)[:10])
        return (d - b).days / 365.25
    except (ValueError, TypeError):
        return np.nan


# ── Public: batter feature matrix ────────────────────────────────────────────
def compute_batter_features(history: pd.DataFrame, targets: pd.DataFrame | None = None,
                            ctx: Context | None = None,
                            universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a feature frame aligned to the batter rows of interest.

    history: all backfill gamelog rows (needs `played`, counting stats, meta).
    targets: inference slate rows (personId, gamePk, gameDate, season, month,
             venue_id, isHome, dayNight, slot, oppStarterId, oppStarterHand,
             oppCatcherId, batSide, teamId). If None, features are produced for
             every played batter row in `history` (training).
    """
    hist = history[history["played"] & history["is_batter"]].copy()
    league = _league_rates(hist, BAT_COUNTS, "PA")

    is_infer = targets is not None
    if is_infer:
        tgt = targets.copy()
        tgt["_is_target"] = True
        for c in BAT_COUNTS:
            if c not in tgt.columns:
                tgt[c] = np.nan
        hist["_is_target"] = False
        combined = pd.concat([hist, tgt], ignore_index=True, sort=False)
    else:
        combined = hist
        combined["_is_target"] = True

    combined = combined.sort_values(["personId", "gameDate", "gameNumber"]).reset_index(drop=True)
    combined = _add_asof_aggregates(combined, BAT_COUNTS, prefix="b_")

    # Marcel priors joined by (personId, season). Include target seasons so an
    # inference season not present in the history still gets a prior.
    tgt_seasons = sorted(pd.Series(combined["season"]).dropna().unique().tolist())
    marcel = _marcel_prior_rates(hist, BAT_COUNTS, "PA", MARCEL_HIT_WEIGHTS,
                                 HITTER_BALLAST_PA, league, seasons=tgt_seasons)
    combined = combined.merge(marcel, on=["personId", "season"], how="left")

    # Opposing starter as-of quality (season-to-date + last-30 form).
    opp = _pitcher_asof_table(history)
    combined = _join_pitcher_asof(combined, opp, id_col="oppStarterId", prefix="opp_")

    # Own-team offense through the prior day (drives R/RBI context) plus
    # last-30-game recent form; and the OPPOSING BULLPEN quality (batters face
    # relievers for a third or more of their PAs).
    team = _team_asof_table(history)
    combined = _join_team_asof(combined, team, id_col="teamId", prefix="own_")
    bullpen = _bullpen_asof_table(history)
    combined = _join_team_asof(combined, bullpen, id_col="oppTeamId", prefix="opp_")

    # Platoon split priors: the batter's regressed component rates vs the
    # starter's hand, from PRIOR seasons only.
    plat = _platoon_split_table(hist, combined[["personId", "season"]])
    combined = combined.merge(plat, on=["personId", "season"], how="left")
    lg_plat = _league_platoon_rates(hist)

    # Game environment: weather/roof (boxscore actuals or pipeline forecast)
    # and the HP umpire's prior-season tendency deltas.
    combined = _attach_environment(combined, ctx, history)

    out = combined[combined["_is_target"]].copy()
    feat = _assemble_batter_feature_cols(out, league, ctx, universe, lg_plat)
    return feat


def _assemble_batter_feature_cols(df, league, ctx, universe,
                                  lg_plat: pd.DataFrame | None = None) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["personId"] = df["personId"].values
    f["gamePk"] = df["gamePk"].values
    f["season"] = df["season"].values
    f["month"] = pd.to_datetime(df["officialDate"], errors="coerce").dt.month.values
    f["is_home"] = df["isHome"].astype(float).values
    f["is_night"] = _series(df, "dayNight").map({"day": 0.0, "night": 1.0}).values
    f["slot"] = df["slot"].astype(float).values
    f["rest_days"] = df["b_rest_days"].clip(upper=15).values
    f["std_G"] = df["b_std_G"].values

    # Opportunity + rate priors (Marcel).
    f["marcel_PApg"] = df["marcel_PApg"].values
    # Shrunk within-season rates per component.
    for c in BAT_RATE_COMPONENTS:
        k = STAB_K.get(c, STAB_K["hit"])
        f[f"rate_{c}"] = _shrink(df[f"b_std_{c}"], df["b_std_PA"], df[f"marcel_{c}"], k).values
        f[f"r15_{c}"] = (df[f"b_r15_{c}"] / df["b_r15_PA"].replace(0, np.nan)).values
    f["marcel_HR"] = df["marcel_HR"].values
    f["marcel_SO"] = df["marcel_SO"].values
    f["marcel_BB"] = df["marcel_BB"].values

    # Platoon advantage flag plus the batter's own regressed component rates vs
    # the starter's hand (prior seasons only, ballast PLATOON_K_PA of the league
    # rate for his batSide x hand cell).
    f["plat_same"] = _platoon_flag(df).values
    if lg_plat is not None:
        hand = _series(df, "oppStarterHand").reset_index(drop=True)
        bs = _series(df, "batSide").reset_index(drop=True)
        lg_key = pd.DataFrame({"batSide": bs.values, "oppStarterHand": hand.values})
        lg = lg_key.merge(lg_plat, on=["batSide", "oppStarterHand"], how="left")
        hand_known = hand.isin(["L", "R"]).values
        pa_vs = np.where(hand.values == "L", _series(df, "vL_PA").values,
                         np.where(hand.values == "R", _series(df, "vR_PA").values, np.nan))
        pa_vs = np.where(hand_known, np.nan_to_num(pa_vs, nan=0.0), np.nan)
        for c in _PLAT_COMPONENTS:
            cnt = np.where(hand.values == "L", _series(df, f"vL_{c}").values,
                           np.where(hand.values == "R", _series(df, f"vR_{c}").values, np.nan))
            cnt = np.where(hand_known, np.nan_to_num(cnt, nan=0.0), np.nan)
            lg_rate = lg[f"lg_{c}"].values
            f[f"plat_{c}"] = (cnt + PLATOON_K_PA * lg_rate) / (pa_vs + PLATOON_K_PA)

    # Opposing starter quality: season-to-date and last-30-appearance form.
    f["opp_sp_k_rate"] = df.get("opp_p_K_rate", pd.Series(np.nan, index=df.index)).values
    f["opp_sp_bb_rate"] = df.get("opp_p_BB_rate", pd.Series(np.nan, index=df.index)).values
    f["opp_sp_hr_rate"] = df.get("opp_p_HR_rate", pd.Series(np.nan, index=df.index)).values
    f["opp_sp_k_r30"] = df.get("opp_p_K_rate_r30", pd.Series(np.nan, index=df.index)).values
    f["opp_sp_bb_r30"] = df.get("opp_p_BB_rate_r30", pd.Series(np.nan, index=df.index)).values
    f["opp_sp_hr_r30"] = df.get("opp_p_HR_rate_r30", pd.Series(np.nan, index=df.index)).values
    f["opp_sp_hand"] = _hand_code(df.get("oppStarterHand")).values

    # Own-team offense through the prior day.
    f["own_team_r_pa"] = df.get("own_team_r_pa", pd.Series(np.nan, index=df.index)).values
    # Structural expected-PA identity (tier-3 candidate): the published
    # slot/team-offense closed form, ~4.65 PA leadoff minus 0.11 per slot,
    # adjusted for team run environment and the home team's skipped 9th.
    _slot = df["slot"].astype(float)
    _rpa = pd.to_numeric(df.get("own_team_r_pa"), errors="coerce")
    f["pa_struct"] = (4.65 - 0.11 * (_slot - 1)
                      + 5.0 * (_rpa - 0.118).fillna(0)
                      - 0.06 * df["isHome"].astype(float)).values
    f["own_team_obp"] = df.get("own_team_obp", pd.Series(np.nan, index=df.index)).values
    # Round-4 candidate blocks (per-target policy decides who trains on them):
    # own-team last-30 form, and the opposing bullpen's quality.
    f["own_form_r_pa"] = df.get("own_form_r_pa", pd.Series(np.nan, index=df.index)).values
    f["own_form_obp"] = df.get("own_form_obp", pd.Series(np.nan, index=df.index)).values
    f["opp_bp_k"] = df.get("opp_bp_k_rate", pd.Series(np.nan, index=df.index)).values
    f["opp_bp_bb"] = df.get("opp_bp_bb_rate", pd.Series(np.nan, index=df.index)).values
    f["opp_bp_hr"] = df.get("opp_bp_hr_rate", pd.Series(np.nan, index=df.index)).values
    f["opp_bp_er"] = df.get("opp_bp_er_out", pd.Series(np.nan, index=df.index)).values
    # Tier-2 blocks: game environment (weather/roof) + HP umpire tendencies.
    for src, name in (("env_temp", "env_temp"), ("env_wind_out", "env_wind_out"),
                      ("env_roof", "env_roof"), ("ump_k_delta", "ump_k_delta"),
                      ("ump_bb_delta", "ump_bb_delta")):
        f[name] = pd.to_numeric(_series(df, src), errors="coerce").values

    # Park factors (Y-1, by batter hand) and opposing catcher metrics, vectorized.
    if ctx is not None:
        seasons = df["season"].values
        venues = df["venue_id"].values
        sides = _series(df, "batSide").values
        pk = ctx.park_lookup(seasons, venues, sides)
        for col, name in [("index_hr", "pf_hr"), ("index_so", "pf_so"),
                          ("index_bb", "pf_bb"), ("index_woba", "pf_woba"),
                          ("index_1b", "pf_1b"), ("index_2b", "pf_2b"),
                          ("index_3b", "pf_3b")]:
            f[name] = pk[col].values
        # NOTE: opposing catcher framing/throwing (Barrett's "supporting
        # catcher") is DEFERRED to v2. Savant's catcher leaderboards are not
        # cleanly year-filterable through the public URL (every year returns the
        # same default snapshot), so using them would inject mild leakage into a
        # feature the spec already flags as small. The starting catcher's
        # identity is still captured (attach_catchers) for display and future
        # use. See docs/decisions.md.

        # Statcast quality-of-contact priors (Y-1): contact quality is more
        # stable than outcomes, so last season's expected stats, exit velocity,
        # and barrel rate sharpen the talent prior. Sprint speed feeds SB.
        pids = df["personId"].values
        f["sc_xwoba"] = ctx.statcast_lookup(seasons, pids, "xbat", "est_woba")
        f["sc_xslg"] = ctx.statcast_lookup(seasons, pids, "xbat", "est_slg")
        f["sc_avg_ev"] = ctx.statcast_lookup(seasons, pids, "evbat", "avg_hit_speed")
        f["sc_brl_pct"] = ctx.statcast_lookup(seasons, pids, "evbat", "brl_percent")
        f["sc_hardhit"] = ctx.statcast_lookup(seasons, pids, "evbat", "ev95percent")
        f["sc_sprint"] = ctx.statcast_lookup(seasons, pids, "sprint", "sprint_speed")
        # Opposing starter's contact quality allowed (Y-1).
        opp_ids = pd.to_numeric(_series(df, "oppStarterId"), errors="coerce").values
        f["opp_sc_xwoba"] = ctx.statcast_lookup(seasons, opp_ids, "xpit", "est_woba")
        f["opp_sc_brl_pct"] = ctx.statcast_lookup(seasons, opp_ids, "evpit", "brl_percent")

        # Tier-1 blocks (per-target policy decides who trains on them):
        # the batter-vs-arsenal expected-whiff matchup, the batter's own
        # plate-discipline profile, and the opposing starter's discipline.
        f["mu_xwhiff"] = ctx.arsenal_matchup(seasons, pids, opp_ids)
        f["bat_whiff"] = ctx.statcast_lookup(seasons, pids, "discbat", "whiff_percent")
        f["bat_chase"] = ctx.statcast_lookup(seasons, pids, "discbat", "oz_swing_percent")
        f["bat_izcon"] = ctx.statcast_lookup(seasons, pids, "discbat", "iz_contact_percent")
        f["sp_whiff"] = ctx.statcast_lookup(seasons, opp_ids, "discpit", "whiff_percent")
        f["sp_fstrike"] = ctx.statcast_lookup(seasons, opp_ids, "discpit", "f_strike_percent")

    # Age.
    if universe is not None:
        bmap = dict(zip(universe["personId"], universe["birthDate"]))
        f["age"] = [_age_years(bmap.get(pid), od)
                    for pid, od in zip(df["personId"], df["officialDate"])]

    return f


# ── Platoon split priors (per-batter rates vs L and vs R starters) ───────────
PLATOON_K_PA = 200.0  # league-ballast PA for regressing a batter's vs-hand rates
_PLAT_COMPONENTS = ["H", "HR", "BB", "SO"]


def _platoon_split_table(hist: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """Per (personId, season): the batter's CUMULATIVE prior-season counts vs L
    and vs R starters (vL_PA, vL_H, ... vR_SO), strictly from seasons before the
    target season. A game counts toward the hand of that day's opposing STARTER
    (a proxy for per-PA pitcher hand, which the boxscore backfill lacks; noted
    in docs/decisions.md).

    query: unique (personId, season) pairs to produce rows for (may include
    seasons absent from hist, e.g. the current season at inference).
    """
    sub = hist[hist["oppStarterHand"].isin(["L", "R"])]
    agg = sub.groupby(["personId", "season", "oppStarterHand"]).agg(
        PA=("PA", "sum"), H=("H", "sum"), HR=("HR", "sum"),
        BB=("BB", "sum"), SO=("SO", "sum")).reset_index()

    out = query[["personId", "season"]].drop_duplicates().copy()
    out["_season"] = pd.to_numeric(out["season"], errors="coerce").astype("float64")
    out["_pid"] = pd.to_numeric(out["personId"], errors="coerce").astype("float64")
    for hand, tag in (("L", "vL"), ("R", "vR")):
        a = agg[agg["oppStarterHand"] == hand].sort_values(["personId", "season"]).copy()
        for c in ["PA"] + _PLAT_COMPONENTS:
            a[f"{tag}_{c}"] = a.groupby("personId")[c].cumsum()
        a["_season"] = pd.to_numeric(a["season"], errors="coerce").astype("float64")
        a["_pid"] = pd.to_numeric(a["personId"], errors="coerce").astype("float64")
        cols = [f"{tag}_{c}" for c in ["PA"] + _PLAT_COMPONENTS]
        left = out.sort_values("_season")
        right = a[["_pid", "_season"] + cols].sort_values("_season")
        # Latest prior season strictly BEFORE the target season (no exact match
        # = the target season's own games never leak into its prior).
        merged = pd.merge_asof(left, right, on="_season", by="_pid",
                               direction="backward", allow_exact_matches=False)
        out = merged
    return out.drop(columns=["_season", "_pid"])


def _attach_environment(combined: pd.DataFrame, ctx, history: pd.DataFrame) -> pd.DataFrame:
    """Merge per-game weather/roof/umpire onto rows, then map the HP umpire to
    his PRIOR-SEASON cumulative K/BB tendency deltas (shrunk, 60-game ballast).

    Training rows get actuals from the boxscore-parsed table by gamePk;
    inference rows carry pipeline-provided temp_f / wind_out / roof_closed /
    hp_ump columns (forecast + GUMBO officials), which win the coalesce.
    """
    if ctx is None or ctx.game_weather.empty:
        return combined
    gw = ctx.game_weather.rename(columns={
        "temp_f": "_gw_temp", "wind_out": "_gw_wind",
        "roof_closed": "_gw_roof", "hp_ump": "_gw_ump"})
    combined = combined.merge(
        gw[["gamePk", "_gw_temp", "_gw_wind", "_gw_roof", "_gw_ump"]],
        on="gamePk", how="left")
    for prov, gwcol, out in (("temp_f", "_gw_temp", "env_temp"),
                             ("wind_out", "_gw_wind", "env_wind_out"),
                             ("roof_closed", "_gw_roof", "env_roof")):
        provided = pd.to_numeric(_series(combined, prov), errors="coerce")
        combined[out] = provided.fillna(pd.to_numeric(combined[gwcol],
                                                      errors="coerce")).values
    ump_name = _series(combined, "hp_ump").fillna(combined["_gw_ump"])
    combined["_ump"] = ump_name.values

    # Umpire K/BB deltas from PRIOR seasons only. Per-game rates come from the
    # batter rows of history (both sides combined).
    bat = history[history["played"] & history["is_batter"]]
    game = bat.groupby(["gamePk", "season"]).agg(
        PA=("PA", "sum"), SO=("SO", "sum"), BB=("BB", "sum")).reset_index()
    game = game[game["PA"] > 0]
    game["k_rate"] = game["SO"] / game["PA"]
    game["bb_rate"] = game["BB"] / game["PA"]
    lg = game.groupby("season").agg(lg_k=("k_rate", "mean"),
                                    lg_bb=("bb_rate", "mean")).reset_index()
    game = game.merge(gw[["gamePk", "_gw_ump"]], on="gamePk", how="left")
    game = game.merge(lg, on="season")
    game["dk"] = game["k_rate"] - game["lg_k"]
    game["dbb"] = game["bb_rate"] - game["lg_bb"]
    per = (game.dropna(subset=["_gw_ump"])
           .groupby(["_gw_ump", "season"])
           .agg(n=("dk", "size"), dk_sum=("dk", "sum"), dbb_sum=("dbb", "sum"))
           .reset_index().sort_values(["_gw_ump", "season"]))
    for c in ("n", "dk_sum", "dbb_sum"):
        per[f"cum_{c}"] = per.groupby("_gw_ump")[c].cumsum()
    per["_season"] = per["season"].astype("float64")

    query = combined[["_ump", "season"]].drop_duplicates().dropna(subset=["_ump"]).copy()
    query["_season"] = pd.to_numeric(query["season"], errors="coerce").astype("float64")
    merged = pd.merge_asof(
        query.sort_values("_season"),
        per[["_gw_ump", "_season", "cum_n", "cum_dk_sum", "cum_dbb_sum"]]
        .rename(columns={"_gw_ump": "_ump"}).sort_values("_season"),
        on="_season", by="_ump", direction="backward", allow_exact_matches=False)
    K_BALLAST = 60.0
    merged["ump_k_delta"] = merged["cum_dk_sum"].fillna(0) / (merged["cum_n"].fillna(0) + K_BALLAST)
    merged["ump_bb_delta"] = merged["cum_dbb_sum"].fillna(0) / (merged["cum_n"].fillna(0) + K_BALLAST)
    combined = combined.merge(
        merged[["_ump", "season", "ump_k_delta", "ump_bb_delta"]],
        on=["_ump", "season"], how="left")
    return combined


def _league_platoon_rates(hist: pd.DataFrame) -> pd.DataFrame:
    """League component rates by (batSide, starter hand) — the shrink target."""
    sub = hist[hist["oppStarterHand"].isin(["L", "R"]) & hist["batSide"].isin(["L", "R", "S"])]
    g = sub.groupby(["batSide", "oppStarterHand"]).agg(
        PA=("PA", "sum"), **{c: (c, "sum") for c in _PLAT_COMPONENTS}).reset_index()
    for c in _PLAT_COMPONENTS:
        g[f"lg_{c}"] = g[c] / g["PA"].replace(0, np.nan)
    return g[["batSide", "oppStarterHand"] + [f"lg_{c}" for c in _PLAT_COMPONENTS]]


# ── Pitcher as-of helpers (used for opposing-starter features) ───────────────
def _pitcher_asof_table(history: pd.DataFrame) -> pd.DataFrame:
    """Per pitcher-appearance shifted rate table keyed for as-of lookup by
    (personId, gameDate): season-to-date rates plus last-30-appearance form.
    """
    p = history[history["played"] & history["is_pitcher"]].copy()
    p = p.sort_values(["personId", "gameDate", "gameNumber"]).reset_index(drop=True)
    p = _add_asof_aggregates(p, ["p_BF", "p_K", "p_BB", "p_H", "p_HR", "p_ER"], prefix="p_")
    bf = p["p_std_p_BF"].replace(0, np.nan)
    bf30 = p["p_r30_p_BF"].replace(0, np.nan)
    out = pd.DataFrame({
        "personId": p["personId"], "gameDate": p["gameDate"], "gameNumber": p["gameNumber"],
    })
    for c in ("p_K", "p_BB", "p_H", "p_HR"):
        out[f"{c}_rate"] = (p[f"p_std_{c}"] / bf).values
        out[f"{c}_rate_r30"] = (p[f"p_r30_{c}"] / bf30).values
    return out.sort_values(["gameDate", "personId"]).reset_index(drop=True)


def _team_asof_table(history: pd.DataFrame) -> pd.DataFrame:
    """Per (teamId, officialDate) team offensive quality through the prior day:
    season-to-date K% and OBP-ish and runs/PA, shifted so a game does not see
    itself. Built from batter rows aggregated to the team-game level first.
    """
    b = history[history["played"] & history["is_batter"]].copy()
    tg = b.groupby(["teamId", "season", "officialDate", "gameNumber"]).agg(
        PA=("PA", "sum"), SO=("SO", "sum"), H=("H", "sum"), BB=("BB", "sum"),
        HBP=("HBP", "sum"), R=("R", "sum"), gameDate=("gameDate", "first"),
    ).reset_index()
    tg = tg.sort_values(["teamId", "gameDate", "gameNumber"]).reset_index(drop=True)
    g = tg.groupby(["teamId", "season"], sort=False)
    for c in ["PA", "SO", "H", "BB", "HBP", "R"]:
        tg[f"cum_{c}"] = g[c].cumsum() - tg[c]
        # Last-30 TEAM GAMES recent form, shifted one game (excludes the row's
        # own game); min 5 games so opening-week form is NaN rather than noise.
        sh = g[c].shift(1)
        tg[f"r30_{c}"] = sh.groupby([tg["teamId"], tg["season"]], sort=False).transform(
            lambda s: s.rolling(30, min_periods=5).sum())
    pa = tg["cum_PA"].replace(0, np.nan)
    pa30 = tg["r30_PA"].replace(0, np.nan)
    out = pd.DataFrame({
        "teamId": tg["teamId"], "gameDate": tg["gameDate"], "gameNumber": tg["gameNumber"],
        "team_k_rate": (tg["cum_SO"] / pa).values,
        "team_obp": ((tg["cum_H"] + tg["cum_BB"] + tg["cum_HBP"]) / pa).values,
        "team_r_pa": (tg["cum_R"] / pa).values,
        "form_k_rate": (tg["r30_SO"] / pa30).values,
        "form_obp": ((tg["r30_H"] + tg["r30_BB"] + tg["r30_HBP"]) / pa30).values,
        "form_r_pa": (tg["r30_R"] / pa30).values,
    })
    return out.sort_values(["gameDate", "teamId"]).reset_index(drop=True)


def _bullpen_asof_table(history: pd.DataFrame) -> pd.DataFrame:
    """Per (teamId, gameDate) BULLPEN quality through the prior day: the team's
    non-starter pitching aggregates (K/BB/HR per batter faced, ER per out),
    season-to-date, shifted so a game never sees itself. Batters spend the late
    innings against these arms; the model previously knew only the starter.
    """
    rp = history[history["played"] & history["is_pitcher"]
                 & (history["is_sp"] == False)].copy()  # noqa: E712
    if rp.empty:
        return pd.DataFrame()
    tg = rp.groupby(["teamId", "season", "officialDate", "gameNumber"]).agg(
        BF=("p_BF", "sum"), K=("p_K", "sum"), BB=("p_BB", "sum"),
        HR=("p_HR", "sum"), ER=("p_ER", "sum"), outs=("p_outs", "sum"),
        gameDate=("gameDate", "first")).reset_index()
    tg = tg.sort_values(["teamId", "gameDate", "gameNumber"]).reset_index(drop=True)
    g = tg.groupby(["teamId", "season"], sort=False)
    for c in ["BF", "K", "BB", "HR", "ER", "outs"]:
        tg[f"cum_{c}"] = g[c].cumsum() - tg[c]
    bf = tg["cum_BF"].replace(0, np.nan)
    outs = tg["cum_outs"].replace(0, np.nan)
    out = pd.DataFrame({
        "teamId": tg["teamId"], "gameDate": tg["gameDate"], "gameNumber": tg["gameNumber"],
        "bp_k_rate": (tg["cum_K"] / bf).values,
        "bp_bb_rate": (tg["cum_BB"] / bf).values,
        "bp_hr_rate": (tg["cum_HR"] / bf).values,
        "bp_er_out": (tg["cum_ER"] / outs).values,
    })
    return out.sort_values(["gameDate", "teamId"]).reset_index(drop=True)


def _join_team_asof(combined, team_table, id_col, prefix) -> pd.DataFrame:
    if team_table.empty:
        return combined
    left = combined.copy()
    left["_gd"] = pd.to_datetime(left["gameDate"], errors="coerce")
    left["_by"] = pd.to_numeric(left[id_col], errors="coerce").astype("float64")
    left = left.reset_index().rename(columns={"index": "_row"})
    left_sorted = left.dropna(subset=["_by", "_gd"]).sort_values("_gd")
    right = team_table.copy()
    right["_gd"] = pd.to_datetime(right["gameDate"], errors="coerce")
    right["_by"] = pd.to_numeric(right["teamId"], errors="coerce").astype("float64")
    right = right.dropna(subset=["_by", "_gd"]).sort_values("_gd")
    rate_cols = [c for c in right.columns
                 if c not in ("teamId", "gameDate", "gameNumber", "_gd", "_by")]
    merged = pd.merge_asof(
        left_sorted, right[["_by", "_gd"] + rate_cols],
        on="_gd", by="_by",
        direction="backward", allow_exact_matches=False,
    )
    for c in rate_cols:
        combined[f"{prefix}{c}"] = np.nan
        combined.loc[merged["_row"].values, f"{prefix}{c}"] = merged[c].values
    return combined


def compute_pitcher_features(history: pd.DataFrame, targets: pd.DataFrame | None = None,
                             ctx: Context | None = None,
                             universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature frame for starting-pitcher rows (is_sp). Same point-in-time
    contract and stacking trick as compute_batter_features.
    """
    hist_sp = history[history["played"] & history["is_pitcher"] & history["is_sp"]].copy()
    # League rates over ALL pitcher appearances (broad denominator for priors).
    all_pit = history[history["played"] & history["is_pitcher"]].copy()
    league = _league_rates(all_pit, PIT_COUNTS, "p_BF")

    is_infer = targets is not None
    if is_infer:
        tgt = targets.copy()
        tgt["_is_target"] = True
        for c in PIT_COUNTS:
            if c not in tgt.columns:
                tgt[c] = np.nan
        hist_sp["_is_target"] = False
        combined = pd.concat([hist_sp, tgt], ignore_index=True, sort=False)
    else:
        combined = hist_sp
        combined["_is_target"] = True

    combined = combined.sort_values(["personId", "gameDate", "gameNumber"]).reset_index(drop=True)
    combined = _add_asof_aggregates(combined, PIT_COUNTS, prefix="p_")

    tgt_seasons = sorted(pd.Series(combined["season"]).dropna().unique().tolist())
    marcel = _marcel_prior_rates(all_pit, PIT_COUNTS, "p_BF", MARCEL_PIT_WEIGHTS,
                                 PITCHER_BALLAST_BF, league, seasons=tgt_seasons)
    combined = combined.merge(marcel, on=["personId", "season"], how="left")

    # Opposing team offensive quality, as-of.
    team = _team_asof_table(history)
    combined = _join_team_asof(combined, team, id_col="oppTeamId", prefix="opp_")

    # Lineup-aggregated arsenal matchup for TRAINING rows: the mean expected
    # whiff of the nine batters who actually started against this pitcher
    # (lineups are known pregame, so this is point-in-time). Inference rows may
    # instead carry a pipeline-provided opp_lineup_xwhiff column computed from
    # the day's posted/projected lineups; the assembly coalesces the two.
    if ctx is not None:
        lu = history[history["played"] & history["is_batter"]
                     & history["is_starter_slot"] & history["oppStarterId"].notna()]
        if not lu.empty:
            mu = ctx.arsenal_matchup(lu["season"].values, lu["personId"].values,
                                     lu["oppStarterId"].values)
            agg = (lu[["gamePk", "oppStarterId"]].assign(_mu=mu)
                   .groupby(["gamePk", "oppStarterId"])["_mu"].mean()
                   .rename("opp_lineup_xwhiff_hist").reset_index()
                   .rename(columns={"oppStarterId": "personId"}))
            combined = combined.merge(agg, on=["gamePk", "personId"], how="left")

    combined = _attach_environment(combined, ctx, history)

    out = combined[combined["_is_target"]].copy()
    return _assemble_pitcher_feature_cols(out, league, ctx, universe)


def _assemble_pitcher_feature_cols(df, league, ctx, universe) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["personId"] = df["personId"].values
    f["gamePk"] = df["gamePk"].values
    f["season"] = df["season"].values
    f["month"] = pd.to_datetime(df["officialDate"], errors="coerce").dt.month.values
    f["is_home"] = df["isHome"].astype(float).values
    # NOTE: is_night was tried for pitchers and REVERTED: it nudged p_outs/p_BF/
    # p_K MAE 0.1-0.3% WORSE on the 2025 walk-forward (noise, no signal). It
    # stays in the batter features, where the same test showed small gains.
    f["rest_days"] = df["p_rest_days"].clip(upper=30).values
    f["std_G"] = df["p_std_G"].values
    f["hand"] = _hand_code(_series(df, "pitchHand")).values

    # Marcel opportunity + rate priors.
    f["marcel_BFpg"] = df["marcel_p_BFpg"].values
    for c in PIT_RATE_COMPONENTS:
        k = PIT_STAB_K.get(c, 150.0)
        f[f"rate_{c}"] = _shrink(df[f"p_std_{c}"], df["p_std_p_BF"], df[f"marcel_{c}"], k).values
        f[f"r30_{c}"] = (df[f"p_r30_{c}"] / df["p_r30_p_BF"].replace(0, np.nan)).values
    # Prior ER/BF and pitch-count trend.
    f["marcel_ER"] = df["marcel_p_ER"].values
    f["std_pitches_r30"] = (df["p_r30_p_BF"]).values  # BF over last 30 as a workload proxy

    # Opposing team offense (season-to-date, plus round-4 last-30 form block).
    f["opp_team_k"] = df.get("opp_team_k_rate", pd.Series(np.nan, index=df.index)).values
    f["opp_team_obp"] = df.get("opp_team_obp", pd.Series(np.nan, index=df.index)).values
    f["opp_team_r_pa"] = df.get("opp_team_r_pa", pd.Series(np.nan, index=df.index)).values
    f["opp_form_k"] = df.get("opp_form_k_rate", pd.Series(np.nan, index=df.index)).values
    f["opp_form_obp"] = df.get("opp_form_obp", pd.Series(np.nan, index=df.index)).values

    # Park (Y-1, all-hand) and own catcher framing, vectorized.
    if ctx is not None:
        seasons = df["season"].values
        venues = df["venue_id"].values
        allside = np.array(["all"] * len(df))
        pk = ctx.park_lookup(seasons, venues, allside)
        for col, name in [("index_hr", "pf_hr"), ("index_so", "pf_so"),
                          ("index_bb", "pf_bb"), ("index_woba", "pf_woba")]:
            f[name] = pk[col].values
        # Own-catcher framing deferred to v2 (see batter assembly note).

        # Statcast quality-of-contact allowed (Y-1): xwOBA-against and barrel
        # rate against are stabler talent signals than outcome rates.
        pids = df["personId"].values
        f["sc_xwoba_ag"] = ctx.statcast_lookup(seasons, pids, "xpit", "est_woba")
        f["sc_avg_ev_ag"] = ctx.statcast_lookup(seasons, pids, "evpit", "avg_hit_speed")
        f["sc_brl_pct_ag"] = ctx.statcast_lookup(seasons, pids, "evpit", "brl_percent")

        # Tier-1 blocks: the pitcher's own Y-1 plate-discipline profile and the
        # lineup-aggregated arsenal matchup (pipeline column wins at inference).
        f["p_whiff"] = ctx.statcast_lookup(seasons, pids, "discpit", "whiff_percent")
        f["p_fstrike"] = ctx.statcast_lookup(seasons, pids, "discpit", "f_strike_percent")
        f["p_chase"] = ctx.statcast_lookup(seasons, pids, "discpit", "oz_swing_percent")
        provided = pd.to_numeric(_series(df, "opp_lineup_xwhiff"), errors="coerce")
        hist_mu = pd.to_numeric(_series(df, "opp_lineup_xwhiff_hist"), errors="coerce")
        f["opp_lineup_xwhiff"] = provided.fillna(hist_mu).values
    # Tier-2 blocks: game environment + HP umpire tendencies.
    for src, name in (("env_temp", "env_temp"), ("env_wind_out", "env_wind_out"),
                      ("env_roof", "env_roof"), ("ump_k_delta", "ump_k_delta"),
                      ("ump_bb_delta", "ump_bb_delta")):
        f[name] = pd.to_numeric(_series(df, src), errors="coerce").values

    if universe is not None:
        bmap = dict(zip(universe["personId"], universe["birthDate"]))
        f["age"] = [_age_years(bmap.get(pid), od)
                    for pid, od in zip(df["personId"], df["officialDate"])]
    return f


def _join_pitcher_asof(combined, opp_table, id_col, prefix) -> pd.DataFrame:
    """Attach the opposing starter's most-recent prior-appearance rates via a
    merge_asof on gameDate.
    """
    if opp_table.empty:
        return combined
    left = combined.copy()
    left["_gd"] = pd.to_datetime(left["gameDate"], errors="coerce")
    left["_by"] = pd.to_numeric(left[id_col], errors="coerce").astype("float64")
    left = left.reset_index().rename(columns={"index": "_row"})
    left_sorted = left.dropna(subset=["_by", "_gd"]).sort_values("_gd")
    right = opp_table.copy()
    right["_gd"] = pd.to_datetime(right["gameDate"], errors="coerce")
    right["_by"] = pd.to_numeric(right["personId"], errors="coerce").astype("float64")
    right = right.dropna(subset=["_by", "_gd"]).sort_values("_gd")
    rate_cols = [c for c in right.columns if c.endswith("_rate") or c.endswith("_rate_r30")]
    merged = pd.merge_asof(
        left_sorted, right[["_by", "_gd"] + rate_cols],
        on="_gd", by="_by",
        direction="backward", allow_exact_matches=False,
    )
    for c in rate_cols:
        combined[f"{prefix}{c}"] = np.nan
        combined.loc[merged["_row"].values, f"{prefix}{c}"] = merged[c].values
    return combined


# ── tiny utilities ───────────────────────────────────────────────────────────
def _series(df, col):
    if col in df.columns:
        return df[col].reset_index(drop=True)
    return pd.Series([np.nan] * len(df))


def _hand_code(s):
    if s is None:
        return pd.Series([np.nan])
    return s.map({"L": 0.0, "R": 1.0, "S": 0.5}).astype(float)


def _platoon_flag(df):
    """1.0 when the batter has the platoon advantage (opposite hand to the
    starter), 0.0 when not, 0.5 for switch hitters / unknown.
    """
    bs = _series(df, "batSide")
    ph = _series(df, "oppStarterHand")
    out = []
    for b, p in zip(bs, ph):
        if b == "S" or pd.isna(b) or pd.isna(p):
            out.append(0.5)
        elif b != p:
            out.append(1.0)  # L vs R or R vs L: advantage batter
        else:
            out.append(0.0)
    return pd.Series(out, index=df.index)
