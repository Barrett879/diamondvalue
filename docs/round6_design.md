# Round 6 design: pitch-level v2 feature tables

Source data: `cache/raw_statcast/red2_{d0}_{d1}.parquet` chunk parts, 2020-2026,
columns: game_pk, game_date, pitcher, batter, fielder_2 (catcher), stand,
p_throws, n_thruorder_pitcher, pitch_type, release_speed, release_pos_x,
release_pos_z, release_spin_rate, plate_x, plate_z, sz_top, sz_bot, zone,
description, events, estimated_woba_using_speedangle. Regular season only
(hfGT=R). Non-overlapping 3-day windows, <25k rows asserted per chunk.

New script `scripts/build_pitchlevel_v2_tables.py` makes ONE pass over each
season's parts and writes six committed tables. All rate features are shrunk
at lookup time with n/(n+k) ballasts, never in the stored table (store sums
and counts).

## Zone geometry (shared)

- Half-width of the called zone: X_EDGE = 0.83 ft (plate half-width 0.708 +
  ball radius ~0.12).
- Vertical edges: sz_bot, sz_top per pitch (batter-specific).
- Signed distance outside the zone: dx = |plate_x| - X_EDGE;
  dz = max(sz_bot - plate_z, plate_z - sz_top); d = max(dx, dz)
  (negative = inside on both axes).
- SHADOW band: -0.25 <= d <= 0.25 ft. Taken pitches only
  (description in {called_strike, ball}; blocked_ball excluded, it is a
  dirt ball and pollutes the band).

## Tables

### 1. framing2_{y}_v1.parquet  — per catcher (fielder_2)
Shadow-band taken pitches: `shadow_n`, `shadow_cs` (called strikes), plus the
season league shadow CSR for reference. Feature value at lookup:
(cs/n - league_csr) * n/(n+1000). Y-1 lookup (like ump/park features).
- Batter rows: opp catcher framing (oppCatcherId) -> `oppc_framing`;
  candidate targets: SO, BB (a good framer steals strikes FROM the batter).
- Pitcher rows: own catcher framing (ownCatcherId) -> `ownc_framing`;
  candidate targets: p_K, p_BB, p_H.

### 2. platoon2_{y}_v1.parquet — per (batter, p_throws)
PA-ending rows only (events non-null). Counts: PA, SO, BB (walk +
intent_walk?), HR, H (single/double/triple/home_run), HBP excluded from BB.
Feature at lookup (batter rows, vs oppStarterHand, Y-1, ballast 200 PA like
round-2 platoon): pp2_k, pp2_bb, pp2_hr, pp2_h — TRUE per-PA attribution,
replacing round 2's whole-game-vs-starter-hand approximation as a CANDIDATE
block (round-2 features stay; gauntlet decides).

### 3. batvelo2_{y}_v1.parquet — per batter
Vs hard fastballs (pitch_type FF/SI, release_speed >= 95): pitches, swings,
whiffs, contact events, xwOBA-on-contact sum/n. Same vs soft (< 93) for the
contrast. Features (Y-1, ballast): bat_whiff95 (shrunk whiff rate on hard
FBs), bat_xw95 (shrunk xwOBA-contact), bat_whiff_gap (hard minus soft).
Paired with a new as-of feature on batter rows: `opp_sp_ffvelo` = the
opposing STARTER's rolling last-3-start fastball velo (from existing
fbvelo_{y} tables, joined via oppStarterId with the same shifted/no-same-day
rules as velo_r3). Candidate targets: SO, b1/b2/b3/HR (contact quality).

### 4. relspin2_{y}_v1.parquet — per (pitcher, game_pk, game_date)
Fastballs (FF/SI) only: rel_x_mean, rel_z_mean, spin_mean, n_fb.
As-of features (pitcher rows, same machinery as velo_r3/velo_trend —
strictly-prior DATE comparison, no same-day):
- spin_drop = last-3-start mean spin - season-to-date mean spin (as-of)
- rel_drift = euclidean distance between last-3-start mean release point
  and season-to-date mean release point (as-of)
Candidate targets: p_outs, p_BF, p_pitches, p_K, p_H (injury/mechanics
canary, same family as velo_trend).

### 5. mix2_{y}_v1.parquet — per (pitcher, season)
Pitch-type distribution: total pitches, Shannon entropy of the mix,
n_types_5pct (types thrown >= 5%), fb_share. Y-1 lookup on pitcher rows:
mix_entropy, mix_ntypes. Candidate targets: p_pitches, p_K, p_H (mix
diversity is TTO resilience).

### 6. batproc2_{y}_v1.parquet — per (batter, game_pk, game_date)
Per game: pitches_seen, swings, whiffs, chases (swing at d > 0 pitch),
contact_n, xwcon_sum (xwOBA on contact). As-of rolling features (batter
rows, shifted, last-15-games vs season-to-date):
- proc_whiff15 = rolling-15 whiff/swing minus season-to-date (form via
  process, not outcomes)
- proc_chase15 = rolling-15 chase rate minus season-to-date
- proc_xw15 = rolling-15 mean xwOBA-on-contact
Candidate targets: SO, BB, b1/b2/b3/HR.

## Feature blocks (gauntlet candidates)

bat role: `framing_bat2` [oppc_framing] -> {SO, BB};
`platoon2_bat` [pp2_k, pp2_bb, pp2_hr, pp2_h] -> batter rate targets;
`batvelo_bat` [bat_whiff95, bat_xw95, bat_whiff_gap, opp_sp_ffvelo] -> {SO,
b1, b2, b3, HR}; `batproc_bat` [proc_whiff15, proc_chase15, proc_xw15] ->
{SO, BB, b1, b2, b3, HR}.
pit role: `framing_pit2` [ownc_framing] -> {p_K, p_BB, p_H};
`relspin_pit` [spin_drop, rel_drift] -> {p_outs, p_BF, p_pitches, p_K, p_H};
`mix_pit` [mix_entropy, mix_ntypes] -> {p_pitches, p_K, p_H}.

Ablate each block on 2024 (exp_feature_blocks.py), grant survivors, confirm
jointly on 2025 via validate_models.py, strike non-replicators, retrain,
regen, push. Noise floor ±0.1% dev.

## Inference plumbing (build_daily_predictions.py)

- Catcher IDs on target rows: posted lineup C if available; else the most
  frequent starting catcher over the team's last 10 games from history
  (mirrors _projected_lineup). Batter rows get oppCatcherId (opposing team's
  C), pitcher rows get ownCatcherId (own team's C).
- opp_sp_ffvelo: joined from fbvelo tables via oppStarterId as-of.
- relspin2/batproc2 for 2026 must exist: bootstrap from red2_2026 parts, and
  extend the daily updater (update_current_velo) to fetch KEEP2 columns and
  refresh fbvelo + relspin2 + batproc2 (+ current-season batvelo2? NO — Y-1
  lookups need no in-season upkeep) in the same rolling 8-day pull.

## Known traps to respect

- merge_asof by-keys must be float64; allow_exact_matches=False everywhere;
  velo-style joins compare DATES not timestamps to kill same-day leak.
- 2020 season: 60 games, small samples everywhere — ballasts handle it; Y-1
  lookups for 2021 will be noisy but present.
- The tracker/statcast tables use Y-1 via prior_year_map — reuse it.
- Stored tables: sums + counts only; shrinkage at lookup.
- gitignore: red2_* parts stay ignored; the six new table families must be
  UN-ignored (add to the existing whitelist).
- Deterministic retrain proof: only granted targets' joblibs may change.

## Implementation revisions (post design-review, applied)

The design review changed several specifics before/during implementation; the
code reflects these and this section is the source of truth where it differs
from the tables above:

- **Cumulative strictly-prior seasons, not Y-1**, for framing2 / platoon2 /
  batvelo2 (the _platoon_split_table / merge_asof-backward pattern). Fixes the
  2021-from-60-game-2020 attenuation and stabilizes part-time catchers.
  mix2 stays single-Y-1 (entropy/fb_share are stable traits).
- **Platoon2 shrinkage**: the batter's own overall rate is the anchor and his
  vs-hand SPLIT delta is shrunk toward the league handedness offset with
  K=1000 PA (splits stabilize slowly), not a 200-PA shrink of the level.
- **mix2 second feature = fb_share (fastball reliance), not n_types_5pct.**
  The review judged n_types_5pct near-duplicate with entropy and fb_share the
  more independent second axis. mix_entropy still carries repertoire breadth.
- **batvelo2 keeps bat_whiff_gap** = shrunk hard-FB whiff minus shrunk soft-FB
  (<93) whiff, each toward its own league rate (so the gap shrinks toward the
  league gap for small samples). Isolates velocity-specific whiff from a
  batter's overall swing-and-miss.
- **Current-season per-game tables merge, not overwrite**: a manual re-backfill
  of relspin2/batproc2 for the in-progress season uses _merge_pergame so it
  augments rather than clobbers the daily updater's fresher edge.
- **red2 parts cached only through today-2** (Savant finalizes recent days);
  the rolling daily updater owns the recent edge with a self-healing window.
