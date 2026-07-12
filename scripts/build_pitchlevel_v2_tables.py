"""Round-6 pitch-level v2 tables, reduced from the red2_* full-column parts.

Per season, ONE concat of that season's parts (about 1M rows, well within
memory), gated by a completeness check against the committed gamelogs (the
ground truth for which dates had games), then six committed tables:

  framing2_{y}_v1.parquet   per catcher: shadow-band taken pitches with a
      location-bin expected-strike adjustment (10 bins of 0.05 ft on the
      signed distance-from-zone-edge d), so the stat is called strikes ABOVE
      expectation given where the pitches were, not staff command. Columns:
      player_id, season, shadow_n, cs_resid (sum of cs - E[cs | season, bin]).
  platoon2_{y}_v1.parquet   per (batter, p_throws): TRUE per-PA splits from
      PA-ending pitches. PA convention matches the tto tables (events
      non-null); BB counts walk + intent_walk to match the boxscore target.
  batvelo2_{y}_v1.parquet   per batter vs hard fastballs (FF/SI >= 95):
      pitches, swings, whiffs, xwcon_sum, xwcon_n. xw features must divide by
      xwcon_n (rows with a non-null xwOBA), NEVER by contact counts -- fouls
      are contact but carry no xw value.
  relspin2_{y}_v1.parquet   per (pitcher, game_pk, game_date) on FF/SI:
      release-point and spin sums + counts for the as-of drift features.
  mix2_{y}_v1.parquet       per (pitcher, season): pitch count, Shannon
      entropy of the pitch-type mix, fastball share (from full-season type
      counts -- entropy is not additively composable, so it is derived once).
  batproc2_{y}_v1.parquet   per (batter, game_pk, game_date): process counts
      (swings, whiffs, out-of-zone pitches and swings, xw contact sums) for
      the rolling form-vs-baseline features.

For the CURRENT season only the two per-game tables (relspin2, batproc2) are
written: the four season-reference tables would otherwise become next year's
Y-1 lookups while built from a half season.

Usage: python scripts/build_pitchlevel_v2_tables.py 2020 2021 ...  (or 'all')
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache  # noqa: E402
from mlblib.cache import logger  # noqa: E402

X_EDGE = 0.83          # plate half-width + ball radius, ft
SHADOW = 0.25          # band half-depth around the zone edge, ft
BIN_W = 0.05           # expected-strike location bins within the band
HARD_FB = 95.0         # hard-fastball cutoff, mph (fixed; share printed)
SOFT_FB = 93.0         # soft-fastball cutoff for the hard-minus-soft contrast
FB_TYPES = ("FF", "SI")
SWINGS = {"foul", "foul_tip", "hit_into_play", "swinging_strike",
          "swinging_strike_blocked", "foul_bunt", "missed_bunt",
          "bunt_foul_tip"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
H_EVENTS = {"single", "double", "triple", "home_run"}
BB_EVENTS = {"walk", "intent_walk"}


def _load_season(season: int) -> pd.DataFrame | None:
    raw_dir = cache.CACHE_DIR / "raw_statcast"
    parts = sorted(raw_dir.glob(f"red2_{season}-*.parquet"))
    if not parts:
        logger.warning("season %s: no red2 parts", season)
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["game_date"] = df["game_date"].astype(str).str[:10]
    return df


def _completeness_gate(season: int, df: pd.DataFrame) -> bool:
    """Every gamelog date of the season (up to the parts' own horizon for the
    current season) must appear in the parts; else the tables would be silent
    half-season truncations."""
    gl = cache.read_parquet_or_none(cache.dc_path(f"gamelogs_{season}_v1.parquet"))
    if gl is None:
        logger.warning("season %s: no gamelogs to gate against", season)
        return False
    have = set(df["game_date"])
    need = set(gl["officialDate"].astype(str))
    if season >= dt.date.today().year:
        need = {d for d in need if d <= max(have)}
    missing = sorted(need - have)
    if missing:
        logger.warning("season %s: INCOMPLETE red2 coverage, %d missing dates "
                       "(%s ... %s) -- refusing to build", season,
                       len(missing), missing[0], missing[-1])
        return False
    return True


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_take"] = df["description"].isin(["called_strike", "ball"])
    df["is_cs"] = df["description"] == "called_strike"
    df["is_swing"] = df["description"].isin(SWINGS)
    df["is_whiff"] = df["description"].isin(WHIFFS)
    df["is_fb"] = df["pitch_type"].isin(FB_TYPES)
    df["is_pa_end"] = df["events"].notna() & (df["events"] != "")
    dx = df["plate_x"].abs() - X_EDGE
    dz = np.maximum(df["sz_bot"] - df["plate_z"], df["plate_z"] - df["sz_top"])
    df["d_edge"] = np.maximum(dx, dz)
    df["has_xw"] = df["estimated_woba_using_speedangle"].notna()
    return df


def build_framing(df: pd.DataFrame, season: int) -> None:
    band = df[df["is_take"] & df["d_edge"].abs().le(SHADOW)
              & df["plate_x"].notna() & df["sz_top"].notna()].copy()
    band["bin"] = ((band["d_edge"] + SHADOW) / BIN_W).astype(int).clip(0, 9)
    exp = band.groupby("bin")["is_cs"].mean().rename("e_cs")
    band = band.merge(exp, on="bin", how="left")
    band["resid"] = band["is_cs"].astype(float) - band["e_cs"]
    out = (band.groupby("fielder_2")
           .agg(shadow_n=("resid", "size"), cs_resid=("resid", "sum"))
           .reset_index().rename(columns={"fielder_2": "player_id"}))
    out["season"] = season
    cache.atomic_to_parquet(out, cache.dc_path(f"framing2_{season}_v1.parquet"))
    logger.warning("framing2_%s: %d catchers, %d shadow pitches",
                   season, len(out), int(out["shadow_n"].sum()))


def build_platoon(df: pd.DataFrame, season: int) -> None:
    pa = df[df["is_pa_end"]].copy()
    pa["so"] = pa["events"].isin(["strikeout", "strikeout_double_play"])
    pa["bb"] = pa["events"].isin(BB_EVENTS)
    pa["hr"] = pa["events"] == "home_run"
    pa["h"] = pa["events"].isin(H_EVENTS)
    out = (pa.groupby(["batter", "p_throws"])
           .agg(pa=("batter", "size"), so=("so", "sum"), bb=("bb", "sum"),
                hr=("hr", "sum"), h=("h", "sum"))
           .reset_index().rename(columns={"batter": "player_id"}))
    out["season"] = season
    cache.atomic_to_parquet(out, cache.dc_path(f"platoon2_{season}_v1.parquet"))
    logger.warning("platoon2_%s: %d batter-hand rows", season, len(out))


def build_batvelo(df: pd.DataFrame, season: int) -> None:
    fb = df[df["is_fb"] & df["release_speed"].notna()]
    hard = fb[fb["release_speed"] >= HARD_FB]
    soft = fb[fb["release_speed"] < SOFT_FB]
    logger.warning("batvelo2_%s: hard share of fastballs %.3f",
                   season, len(hard) / max(len(fb), 1))
    xw = hard["estimated_woba_using_speedangle"]
    out = (hard.assign(xw=xw.fillna(0.0))
           .groupby("batter")
           .agg(pitches=("batter", "size"), swings=("is_swing", "sum"),
                whiffs=("is_whiff", "sum"), xwcon_sum=("xw", "sum"),
                xwcon_n=("has_xw", "sum"))
           .reset_index().rename(columns={"batter": "player_id"}))
    softg = (soft.groupby("batter")
             .agg(soft_swings=("is_swing", "sum"),
                  soft_whiffs=("is_whiff", "sum"))
             .reset_index().rename(columns={"batter": "player_id"}))
    out = out.merge(softg, on="player_id", how="left")
    out[["soft_swings", "soft_whiffs"]] = out[["soft_swings",
                                               "soft_whiffs"]].fillna(0.0)
    out["season"] = season
    cache.atomic_to_parquet(out, cache.dc_path(f"batvelo2_{season}_v1.parquet"))
    logger.warning("batvelo2_%s: %d batters", season, len(out))


def reduce_relspin(df: pd.DataFrame, season: int) -> pd.DataFrame:
    fb = df[df["is_fb"]].copy()
    fb["spin"] = fb["release_spin_rate"]
    out = (fb.assign(spin0=fb["spin"].fillna(0.0),
                     has_spin=fb["spin"].notna())
           .groupby(["pitcher", "game_pk", "game_date"])
           .agg(n_fb=("pitcher", "size"),
                relx_sum=("release_pos_x", "sum"),
                relz_sum=("release_pos_z", "sum"),
                spin_sum=("spin0", "sum"), spin_n=("has_spin", "sum"))
           .reset_index().rename(columns={"pitcher": "player_id"}))
    out["season"] = season
    return out


def reduce_batproc(df: pd.DataFrame, season: int) -> pd.DataFrame:
    d = df.copy()
    d["oz"] = d["d_edge"] > 0
    d["oz_swing"] = d["oz"] & d["is_swing"]
    d["xw0"] = d["estimated_woba_using_speedangle"].fillna(0.0)
    out = (d.groupby(["batter", "game_pk", "game_date"])
           .agg(pitches=("batter", "size"), swings=("is_swing", "sum"),
                whiffs=("is_whiff", "sum"), oz_n=("oz", "sum"),
                oz_swings=("oz_swing", "sum"), xwcon_sum=("xw0", "sum"),
                xwcon_n=("has_xw", "sum"))
           .reset_index().rename(columns={"batter": "player_id"}))
    out["season"] = season
    return out


def build_mix(df: pd.DataFrame, season: int) -> None:
    known = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
    counts = (known.groupby(["pitcher", "pitch_type"]).size()
              .rename("n").reset_index())
    tot = counts.groupby("pitcher")["n"].sum().rename("pitches")
    counts = counts.merge(tot, on="pitcher")
    counts["p"] = counts["n"] / counts["pitches"]
    ent = (counts.assign(e=-counts["p"] * np.log(counts["p"]))
           .groupby("pitcher")["e"].sum().rename("mix_entropy"))
    fbsh = (counts[counts["pitch_type"].isin(FB_TYPES)]
            .groupby("pitcher")["p"].sum().rename("fb_share"))
    out = pd.concat([tot, ent, fbsh], axis=1).reset_index().rename(
        columns={"pitcher": "player_id"})
    out["fb_share"] = out["fb_share"].fillna(0.0)
    out["season"] = season
    cache.atomic_to_parquet(out, cache.dc_path(f"mix2_{season}_v1.parquet"))
    logger.warning("mix2_%s: %d pitchers", season, len(out))


def build_season(season: int) -> bool:
    df = _load_season(season)
    if df is None or not _completeness_gate(season, df):
        return False
    df = _derive(df)
    current = season >= dt.date.today().year
    if not current:
        build_framing(df, season)
        build_platoon(df, season)
        build_batvelo(df, season)
        build_mix(df, season)
    rs = reduce_relspin(df, season)
    bp = reduce_batproc(df, season)
    if current:
        # Merge (not overwrite) so a manual re-backfill augments rather than
        # clobbers the daily updater's fresher edge (same dedup key).
        from scripts.build_pitchlevel_backfill import _merge_pergame
        _merge_pergame(rs, cache.dc_path(f"relspin2_{season}_v1.parquet"),
                       ["player_id", "game_pk"])
        _merge_pergame(bp, cache.dc_path(f"batproc2_{season}_v1.parquet"),
                       ["player_id", "game_pk"])
    else:
        cache.atomic_to_parquet(rs, cache.dc_path(f"relspin2_{season}_v1.parquet"))
        cache.atomic_to_parquet(bp, cache.dc_path(f"batproc2_{season}_v1.parquet"))
    logger.warning("relspin2_%s: %d starts; batproc2_%s: %d batter-games%s",
                   season, len(rs), season, len(bp),
                   " (current season: per-game tables only)" if current else "")
    return True


def main(argv: list[str]) -> None:
    seasons = (list(range(2020, dt.date.today().year + 1))
               if (not argv or argv[0] == "all") else [int(a) for a in argv])
    ok = [s for s in seasons if build_season(s)]
    logger.warning("built seasons: %s", ok)


if __name__ == "__main__":
    main(sys.argv[1:])
