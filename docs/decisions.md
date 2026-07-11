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

## Phase 4 (daily pipeline)

- INFERENCE HISTORY FILTER (important): build_daily_predictions filters history
  to officialDate strictly < the target date before computing features. The
  whole slate is predicted pregame, and for a past target date the slate games
  are already in the loaded gamelogs; without the filter a target row would see
  its own real game as a "prior" appearance (rest_days ~ 0, leaked rolling
  stats), which drove pitcher IP predictions down to ~3. With the filter,
  starter IP predictions land at a realistic ~5.
