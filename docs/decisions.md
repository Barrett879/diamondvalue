# Decisions log

One line per choice that the build spec (MLB_PROJECT_INSTRUCTIONS.md) did not
pin down, so Barrett can audit later. Newest at the bottom.

## Phase 1 (data layer)

- Raw MLB Stats API via `requests` behind a thin client in `mlblib/fetch.py`; no
  `MLB-StatsAPI` pip dependency (one less thing to pin, the API is simple).
- Default request headers set a plain `User-Agent` and `Accept: application/json`
  to avoid the occasional HTTP 406 seen with unusual Accept headers.
- Raw boxscore JSON cached under `cache/raw_boxscores/{gamePk}.json` so the
  backfill is resumable and only ever hits each game once.
- `officialDate` (not an Eastern conversion of `gameDate`) is the authoritative
  local game date and the point-in-time day key, per spec rule 7.
- Same-day ordering key is `(officialDate, gameNumber)`; `gameDate` timestamp is
  used only for cross-day point-in-time comparisons.

## Phase 2 (features)

- Platoon splits are computed FROM the backfill game logs (batter's prior-season
  rates vs the starter's throwing hand, using each log's oppStarterHand), not
  from the statSplits endpoint. This is point-in-time by construction, avoids
  thousands of extra fetches, and keeps one data source. Regressed to the league
  platoon effect by sample size.
- Rolling "last 15 / last 30 team games" is implemented as the player's last 15 /
  30 APPEARANCES (games he played), a close and simpler proxy for the spec's
  "team games, aggregating only games the player appeared in".
- Point-in-time ordering key is (gameDate timestamp, gameNumber); an as-of
  aggregate is every prior row in that order, so game 2 of a doubleheader sees
  game 1 (its gameDate is strictly later, verified in backfill).
- Marcel ballast constants: hitters 1200 PA of league-average rate (weights
  5/4/3 over the three prior seasons); pitchers 400 BF of league-average rate
  (weights 3/2/1). Fewer than three prior seasons -> available weights
  renormalized; the ballast absorbs the missing history.
- Small-sample shrinkage of within-season rates toward the Marcel prior uses
  weight n/(n+k) with k from stabilization research: K% 60, BB% 120, HR 170,
  hit-type/OBP-like 460 (PA); pitcher rates use BF-scaled analogs.
- Park factors for a season-Y row read the Y-1 file (or the nearest available
  prior year), never season Y.
- CATCHER FRAMING/THROWING DEFERRED TO v2. Barrett asked for the "supporting
  catcher" as a feature, but Baseball Savant's catcher-framing and
  catcher-throwing leaderboards are not cleanly year-filterable through the
  public URL: every year value returns the same default 60-catcher snapshot
  (verified: Freddy Fermin's framing run value is identical for 2018 and 2024).
  Using that snapshot as a per-season feature would inject mild leakage into a
  feature the spec itself flags as a small effect, so it is dropped from the v1
  model. The starting catcher's identity IS still captured (features.attach_
  catchers) for display and as a v2 hook; a proper per-year framing source
  (pitch-level Statcast aggregation) would wire it back in.

## Phase 3 (models)

- Poisson mechanics: rate models train on y = count / opportunities with
  sample_weight = opportunities and loss="poisson" (the GLM exposure trick;
  HistGBR has no offset). Opportunity and direct targets train on raw counts.
- Shipped models train on 2021-2025 (all data). validate_models.py trains a
  separate 2021-2023 model and tests on 2025 for the honest report.

## Accuracy round 2 (2026-07-10, post-launch)

- Tested four feature additions on the 2025 walk-forward (ship only what wins):
  1. SHIPPED (batters): per-batter regressed platoon split rates vs the
     starter's hand (plat_H/HR/BB/SO), prior seasons only, ballast 200 PA of
     the league rate by batSide x hand. A game's PA count toward the hand of
     that day's opposing STARTER (proxy; boxscores lack per-PA pitcher hand).
  2. SHIPPED (batters): own-team offense through the prior day (own_team_r_pa,
     own_team_obp). This is what flipped R and RBI from marginal to PASSING
     the season-average baseline.
  3. SHIPPED (batters): opposing starter last-30-appearance form rates
     (opp_sp_k/bb/hr_r30) alongside season-to-date.
  4. SHIPPED (batters) / REVERTED (pitchers): is_night. Small gain for
     batters; made pitcher p_outs/p_BF/p_K 0.1-0.3% WORSE, so removed there.
- Net effect: batting MAE improved on 10/11 stats (0.1-0.8%), R and RBI now
  beat the season-average baseline; pitching unchanged (revert restored it
  exactly). Consistent with the variance ceiling: real but small gains.
- Not attempted this round (future levers): Statcast quality-of-contact
  enrichment (biggest known lever, heavy pulls), weather, umpire tendencies,
  proper per-PA platoon data, opposing bullpen quality.

## Accuracy round 3: Statcast (2026-07-11)

- Built season-level Statcast tables from Savant (scripts/build_statcast_
  tables.py): expected stats (bat+pit), exit velo/barrels (bat+pit), sprint
  speed, years 2018-2026. Year filtering VERIFIED per leaderboard (unlike the
  catcher boards): Judge avg EV 97.6 (2023) vs 96.2 (2024); Ohtani sprint
  27.8 vs 28.1.
- Whole-block walk-forward on 2025 was mixed (deviance better 15/18, MAE
  slightly worse on most batter stats, R/RBI lost their season-avg pass), so
  ran a PER-TARGET ablation on the 2024 VALIDATION year (exp_statcast_
  ablation.py) to avoid test-set shopping.
- Verdict: the quality-of-contact block does not reliably beat the existing
  Marcel + shrunk-rate priors for any target. The exception is SPRINT SPEED
  for stolen bases: SB Poisson deviance improved 3.8% on 2024 AND 3.0% on
  2025 (independent years). SB MAE ticked up ~0.5-2%, which is the expected
  artifact of a rare-event model daring to predict nonzero; the spec's 4.5
  pre-registered deviance-over-MAE as the SB decision rule.
- SHIPPED: per-target feature policy in mlblib/model.py (target_feature_cols):
  Statcast columns train ONLY the SB model; all other targets train without
  them. Features are still computed for all rows (cheap, and the SB model and
  future rounds use them).
- Lesson recorded: at per-game granularity, outcome-based priors already
  carry most of the contact-quality signal; Statcast's marginal value here is
  in rare-event skills (speed -> steals), not general hitting quality.

## Accuracy round 4: bullpen + team form (2026-07-11)

- Three candidate blocks, all point-in-time from existing gamelogs, ablated
  per-target on 2024 via the generalized scripts/exp_feature_blocks.py:
  1. REJECTED: opposing BULLPEN quality (K/BB/HR per BF, ER/out of the
     opponent's non-starters). Mechanistically appealing (batters face
     relievers for a third of their PAs) but nothing above the noise floor;
     season-aggregate bullpen quality is too diffuse (availability varies
     nightly).
  2. REJECTED: own-team last-30 form for batters. Nothing above noise. Third
     consecutive round where hot/cold-streak-style signals fail, consistent
     with the professional systems' view.
  3. SHIPPED for p_K ONLY: opposing lineup's last-30 form (form_k_rate,
     form_obp) for starting pitchers. K improved on BOTH years (2024 dev
     -0.41%/MAE -0.23%; 2025 dev -0.19%/MAE -0.20%). Its 2024 ER gain
     (dev -0.64%) FAILED the 2025 confirmation (slightly worse) and was
     rejected -- the two-stage design catching a validation-year mirage.
- Deterministic-retrain proof: only p_K_histgb_m1.joblib changed bytes.
- Feature-block gating generalized in mlblib/model.py FEATURE_BLOCKS;
  rejected blocks' columns remain computed (cheap) but train nothing.

## Phase 4 (daily pipeline)

- INFERENCE HISTORY FILTER (important): build_daily_predictions filters history
  to officialDate strictly < the target date before computing features. The
  whole slate is predicted pregame, and for a past target date the slate games
  are already in the loaded gamelogs; without the filter a target row would see
  its own real game as a "prior" appearance (rest_days ~ 0, leaked rolling
  stats), which drove pitcher IP predictions down to ~3. With the filter,
  starter IP predictions land at a realistic ~5.

## Tier-1 Phase A: arsenal matchup + plate discipline (2026-07-11)

- Four new Savant tables per year 2018-2026 (build_arsenal_tables.py):
  arsenal_pit/arsenal_bat (per pitch type: usage, whiff, k%, put_away) and
  disc_pit/disc_bat (8 plate-discipline keys, all verified populating).
- KEY RESULT: the batter-level arsenal matchup (mu_xwhiff, batter whiff vs
  the starter's mix) showed NOTHING above noise for any batter target, but
  the LINEUP-AGGREGATED version (mean expected whiff of the 9 starters he
  faces) was the strongest pitcher block of the round. Averaging 9 batters
  cancels the single-batter noise. SHIPPED for p_K (2024 dev -0.49%, 2025
  -0.32%, stacks with round-4 oppform: p_K MAE 1.7798 -> 1.7763 -> 1.7733)
  and p_H (2024 -0.10%, 2025 -0.14%).
- FAILED 2025 confirmation: p_ER (SECOND ER mirage in two rounds), p_HR,
  disc_pit workload keeps (p_outs/p_pitches regressed hard), disc_bat HBP
  (targeted check: dev worse). Both discipline blocks and the batter-level
  matchup ship nowhere; columns remain computed for future rounds.
- Phase B (pitch-level TTO + fastball-velo trends) backfill running:
  build_pitchlevel_backfill.py, 3-day chunks, 25k-row cap asserted.

## Tier-2: game environment + umpires (2026-07-11)

- Zero-fetch training data: weather/wind/roof/HP-ump parsed from the 16,887
  cached boxscores (build_weather_tables.py; 100% temp + ump coverage). The
  boxscore wind string is already park-relative, so no azimuth math in
  training. Venue geometry (lat/lon/elevation/azimuth/roofType) from the
  Stats API venues endpoint.
- POLICY-GATE BUG found and fixed during confirmation: target_feature_cols
  used any-block-can-veto semantics, so weather_bat's empty target set
  silently stripped env cols from weather_pit's granted targets -> a
  confirmation run trained byte-identical models (0.00% everywhere).
  Grant-wins semantics now (a col is kept iff ANY block containing it grants
  the target). No earlier round shared cols across blocks, so shipped
  policies were unaffected.
- 2024 ablation + 2025 confirmation: weather SHIPPED for p_H and p_HR (HR
  allowed dev -0.92%/MAE -0.44%, the largest confirmed effect since
  sprint->SB; physics says temp/wind move HR and the data agrees). Umpire
  deltas SHIPPED for p_BB (-0.14%) and p_K (-0.17%, its FOURTH stacking
  gain). Batter-side weather and umpire blocks: noise, rejected. p_ER failed
  a THIRD confirmation; ER is now the project's designated mirage -- its
  noise swallows even physics-backed signals.
- Inference plumbing: Open-Meteo hourly forecast (keyless) at venue coords
  for the first-pitch UTC hour, wind projected onto the park's CF azimuth;
  domes fixed at 72F/no wind; retractable roofs dampen wind by the venue's
  historical closed share; HP ump via GUMBO once a game hits Pre-Game
  (morning runs carry NaN, afternoon runs populate).

## Tier-3: training techniques (2026-07-11)

- Trainer upgraded (mlblib/train.py): optional recency sample-weight decay
  (w *= 0.5^(days_ago/half_life)) and per-target monotonic_cst maps
  (conservative known-direction signs only, mlblib/model.py MONOTONIC_MAPS).
  Per-target grants live in TRAIN_POLICY, consumed via train_kwargs() by the
  production trainer, the validator, and future experiment baselines.
- SHIPPED after 2024 ablation + 2025 confirmation:
  PA + recency-300d (dev -1.37%/-1.10%, dose-response: 300d beat 550d,
  exactly what a role-driven stat should show; PA is the exposure input to
  every batter counting stat); monotonic for p_outs (-0.47% on 2025, this
  target's FIRST win in seven rounds), p_K (FIFTH stacked gain), p_BB, b3.
- CAUGHT: p_BF recency was the largest mirage yet (2024 -0.42% -> 2025
  +1.82%); p_ER is now 0-for-4 across rounds; batter-PA monotonic was
  catastrophic on 2024 (+4.4%, the slot constraint fights pinch-hit rows);
  HBP monotonic failed confirmation; structural-PA identity feature
  (pa_struct) rejected outright: the trees had already learned the closed
  form. Recency HURTS most pitcher rate stats (p_BB +1.1-1.6%) -- pitcher
  skill is stable; forgetting old data just adds variance.
- Deferred to a future round: hurdle models for rare counts, GLM anchor
  stacking, isotonic recalibration, per-target hyperparameter tuning, the
  manager hook model.
- POST-SHIP AUDIT (same day): the technique harness loaded a thinner 2025
  history (no 2024 season) than the real validator, so its p_outs/p_K mono
  "confirmations" did not transfer (+0.10%/+0.06% under full history) and
  were REVERTED before push; PA recency (-1.09%), b3 mono (-0.28%), and
  p_BB mono (dev+MAE better everywhere) survived. Harness fixed to load
  validator-identical history. Lesson: a confirmation harness must match the
  production configuration EXACTLY or its verdicts are about a different
  model.

## Tier-4: pitch-level TTO, velocity fatigue, pen fatigue (2026-07-11)

- The pitch-level backfill (6 seasons, ~4M pitches reduced to two small
  tables) paid for itself: the tier-4 JOINT confirmation on 2025 via the
  real validator produced the largest confirmed gains in project history.
  SHIPPED: velocity fatigue trend (velo_r3/velo_trend, shifted as-of from
  per-start fastball velo) for p_outs/p_BF/p_pitches/p_K/p_H; TTO decay
  profiles (Y-1) for p_pitches/p_K/p_H; own-pen day-of fatigue for
  p_outs/p_BF/p_pitches; opposing-pen fatigue for PA.
  Headline numbers: p_pitches dev -2.09%/MAE -1.35% (9.52 -> 9.39), p_BF
  -1.75%, p_H -1.26%, p_K -0.22% (SIXTH stacked gain).
- STRUCK at joint confirmation: p_HR (+0.35%) and p_ER (+0.51%; ER is now
  0-for-5 and permanently suspect).
- In-season upkeep: update_current_velo() fetches the last ~8 days of
  pitch data (2-3 chunks) each daily run and merges into fbvelo_{year}, so
  the velocity features stay live; bootstrapped 2026 with 45 days (4,920
  starts). TTO uses Y-1 only, so no in-season upkeep needed.
- Deferred (needs a re-pull keeping fielder_2/plate coords): catcher
  framing v2, batter-vs-velocity splits, catcher-throwing vs SB.
