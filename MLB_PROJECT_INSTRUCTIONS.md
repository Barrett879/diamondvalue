# Build instructions: DiamondValue, an MLB per-game player projection site

You (Opus) are building a new website for Barrett. It is a sibling project to HoopsValue, his NBA analytics site. Read this whole document before writing any code. Work through the phases in order; each phase has acceptance checks that must pass before moving on. Several facts below (endpoint behavior, API traps) were verified live in July 2026; when reality disagrees with this document, note it in docs/decisions.md, adapt, and keep moving. Do not silently deviate.

Working title: DiamondValue (Barrett may rename it later; do not spend time on branding).
Repo location (build the site here; this document lives here too): /Users/barrettcollins/Desktop/MLB Predict
Reference project (read it, copy its proven patterns): /Users/barrettcollins/Desktop/Claude/nba-value-app

## 0. What you are building

A Streamlit site where Barrett picks any date with MLB games and sees, for every game on that slate, predicted statistics for every available player on both rosters. Predictions come from scikit-learn HistGradientBoostingRegressor models trained on multi-season per-game logs, using context features Barrett named explicitly: month of the year, the opposing starting pitcher, and the catchers involved (his "supporting catcher"), plus platoon matchup, ballpark, lineup slot, rest, and regressed multi-season talent priors. The headline batter rate he asked for, "OBS%", is delivered as OPS (on-base plus slugging), derived from predicted components as explained below.

### The canonical stat table (single source of truth; every other section defers to this)

Batters (per game):
| Stat | How it is produced | Displayed |
|---|---|---|
| PA | direct model (opportunity), Poisson loss | yes |
| 1B, 2B, 3B, HR, BB, HBP, SO | per-PA rate models (see 4.1 mechanics), counts = E[PA] x rate | 2B, 3B, HR, BB, SO shown; 1B, HBP feed derived stats |
| H | derived: 1B + 2B + 3B + HR | yes |
| TB | derived: 1B + 2x2B + 3x3B + 4xHR (never modeled directly; keeps TB >= H consistent) | yes |
| R, RBI | direct per-game models, Poisson loss (team-dependent, not derivable from own components; features include lineup slot and team offense) | yes |
| SB | direct per-game model, Poisson loss | yes |
| AB | derived: PA - BB - HBP - league-rate SF | feeds AVG/SLG |
| AVG | derived: H / AB | yes |
| OBP | derived: (H + BB + HBP) / PA, labeled approximate (official OBP uses AB+BB+HBP+SF) | yes |
| SLG | derived: TB / AB | yes |
| OPS | derived: OBP + SLG (this is Barrett's "OBS%") | yes |

Starting pitchers (per start):
| Stat | How it is produced | Displayed |
|---|---|---|
| Outs recorded / IP | direct model (opportunity), Poisson loss | yes (as IP) |
| BF | direct model, Poisson loss | feeds rate targets |
| K, BB, H, HR allowed | per-BF rate models, counts = E[BF] x rate | yes |
| ER | direct per-game model, Poisson loss (BABIP-noisy; accept it) | yes |

### What v1 deliberately does NOT predict, and why (show Barrett this list before starting)

- Single-game OPS/OBP/SLG are never modeled directly. They are undefined for 0-PA games and essentially unpredictable per game; deriving them from predicted components is both more honest and more stable.
- Relief pitcher game lines. Whether a reliever pitches at all is a bullpen-management decision, not a stats problem. Relievers are listed with season-context rates and a "relief usage not modeled in v1" note.
- Bench batters get conditional predictions ("per game IF he starts", computed with a neutral 6th lineup slot), clearly labeled, rather than being hidden. Barrett wants every rostered player visible; this is how that happens honestly.
- Batter-vs-pitcher history, hot-streak indicators, and un-regressed small-sample splits are excluded as features. The best professional systems (THE BAT, SaberSim) treat them as noise.
- Pitcher W/L, CS, GIDP, holds/saves are not displayed: per game they are team-dependent or pure noise.

### Honest expectations (put this in the About page too, in Barrett's voice, no hype)

Single-game baseball outcomes are dominated by variance. Every number the site shows is an expected value, the mean of a distribution, not a forecast of what will happen. "1.1 expected hits" means the model's distribution centers there, not that the player will get 1 hit. Pitcher strikeouts are the most predictable per-game stat (game-to-game SD about 2 to 2.5 against a talent range of roughly 3.5 to 9), so treat them as the marquee number. Batter counting stats carry a per-game R^2 ceiling on the order of 0.02 to 0.10 even for professional systems. Success for this project is defined as (a) beating the mandatory baselines in Section 4.5 out of sample and (b) calibration, predicted means matching realized means by decile. It is NOT defined as "calling" games. Do not let disappointing-looking hitter R^2 numbers trigger a redesign; that is the physics of baseball, not a bug.

## 1. Ground rules and conventions (from HoopsValue, non-negotiable)

1. Writing style: no em dashes in UI text, page copy, or explanatory prose. No emojis in UI text. Escape dollar signs in Streamlit markdown (\$). Missing table values render as an em-dash sentinel defined ONCE as a named constant (SENTINEL) in mlblib/theme.py; that constant is the only em dash allowed in string literals.
2. Theme: copy HoopsValue's token system. All CSS uses custom-property tokens (var(--panel), var(--fg-1..4), accent tokens). Port inject_theme(), render_page_chrome(), render_nav(), and theme_fig() from nba-value-app/utils.py, trimming NBA-specific bits. Support the ?theme= URL param the same way.
3. Streamlit gotchas (both cost HoopsValue real debugging time):
   - Never reorder a keyed selectbox's options between reruns. Streamlit hashes str(options) into the element ID; a reorder orphans the selection. Date and player pickers must use stable option ordering, index=None, a session-state key seeded once from a query param, and on_change mirroring back to st.query_params.
   - Never use components.html iframes for critical UI. On a Render cold start the iframe request can be answered by the app homepage itself. Use native widgets (st.pills, st.selectbox) inside @st.fragment.
4. Module boundaries: do NOT recreate HoopsValue's 255KB utils.py monolith. Create a package:
   - mlblib/cache.py (disk cache toolkit ported nearly verbatim from utils.py: CACHE_DIR selection with /data/cache and repo ./cache fallback, _seed_disk_cache_from_repo gap-fill, atomic writes via tmp sibling + os.replace, pkl/parquet/json helpers. The ONE function rewritten rather than ported is _dc_fresh: see rule 10 for the date-keyed freshness logic that replaces HoopsValue's season logic.)
   - mlblib/fetch.py (all MLB Stats API and Savant calls, every one using the stale-beats-empty template)
   - mlblib/features.py (point-in-time feature construction, shared by training and inference)
   - mlblib/model.py (artifact loading, feature-vector assembly, prediction)
   - mlblib/theme.py (tokens, chrome, nav, SENTINEL)
5. scripts/ naming discipline from day one, with a scripts/README.md: build_* (cache and artifact builders), train_* (model training), validate_*/backtest_* (checks), exp_* (research probes, kept even when rejected). All run from repo root.
6. Pins: Python 3.12.8 (runtime.txt AND .python-version AND PYTHON_VERSION env var on Render; Render's default moved to 3.14 which breaks pyarrow/numpy wheels). requirements.txt uses exact pins. scikit-learn and joblib must be pinned to the exact versions the shipped model artifacts were saved with, with a comment in requirements.txt saying so (version drift silently breaks joblib.load on a Render rebuild). Start from HoopsValue's pins: streamlit==1.51.0, numpy==2.3.5, pandas==2.3.3, pyarrow==21.0.0, plotly==6.3.0, scikit-learn==1.7.2, joblib==1.5.2, requests==2.32.5. Add pytest as a dev dependency (tests are mandated below).
7. Keys and IDs: every join keys on gamePk + MLBAM personId. Never join on date + player name (doubleheaders create two games per date; names collide and carry diacritics). The authoritative local game date is the API's officialDate field; use it for day bucketing (do not derive the day by converting gameDate to Eastern). Same-day game ordering uses (officialDate, gameNumber); gameDate timestamps are placeholders for game 2 of traditional doubleheaders, so use gameDate only for cross-day ordering. Two-way players (Ohtani type) can legitimately have both a batter row and a pitcher row for the same gamePk; support that.
8. Data corrections layer: hand-curated CSVs in data/ that override scraped feeds (the HoopsValue roster_corrections.csv pattern). Create these early with headers even while empty, and make the pipeline read them: data/roster_corrections.csv (personId, teamId, action[add|remove], note) and data/player_corrections.csv (personId, field, value, note).
9. Stale beats empty, everywhere: every network fetcher reads the possibly-stale disk cache first, returns it if fresh, does a BOUNDED retry loop (3 tries if a stale copy exists, 8 if not, exponential backoff, per-request timeout=15), and on total failure serves the stale copy with a logger warning. Never block a page behind a hung endpoint, never return empty when stale exists. fetch_league_stats in nba-value-app/utils.py (around line 2616) is the reference implementation.
10. Cache freshness for a date-keyed site (this is the rewritten _dc_fresh): past dates become immutable only after a 48-hour grace window (MLB issues stat corrections for a day or two after games); today and future dates get a short TTL (30 to 60 minutes) because probables and lineups change. Version-stamp derived files in the filename (predictions_2026_07_10_m1.parquet where m1 is MODEL_VERSION); bump the version to invalidate, never edit in place. On Render, /data seeding is gap-fill only: a changed committed cache file with the same name never reaches production, so shipping a changed file requires a filename version bump.
11. Single named logger ("diamondvalue"), env-var level, every fetch failure logs a warning rather than failing silently.

## 2. Phase 1: Data layer (MLB Stats API + Baseball Savant)

The MLB Stats API at https://statsapi.mlb.com/api/v1 is free and keyless (verified July 2026). It is also unofficial and undocumented for the public: no SLA, and endpoints can change. Isolate every call inside mlblib/fetch.py and cache every raw response to disk so the site never depends on a live call succeeding. Community docs: https://github.com/toddrob99/MLB-StatsAPI/wiki/Endpoints and https://github.com/pseudo-r/Public-MLB-API. Use plain requests with default-ish headers. You may use the MLB-StatsAPI pip package (toddrob99, v1.9.x) for convenience, but raw requests behind your own thin client is equally fine and one less dependency. Shell-testing note: hydrate syntax uses square brackets, so curl needs -g; Python requests handles it natively.

Endpoints you will need:
- Schedule for a date: GET /api/v1/schedule?sportId=1&date=YYYY-MM-DD (startDate/endDate for ranges). Returns dates[].games[] with gamePk, gameDate (UTC ISO; display times in US/Eastern), officialDate (authoritative local game date), gameNumber, status, teams, venue.
- Probable pitchers: add &hydrate=probablePitcher(note) to the schedule call.
- Starting lineups, pregame and live: &hydrate=lineups on the schedule (lineups.homePlayers/awayPlayers once posted), or the boxscore's team-level battingOrder array BEFORE the game goes final.
- TRAP (verified live): in a FINISHED game's boxscore, teams.home|away.battingOrder is the FINAL batting order after substitutions, not the starting nine. For training, slot labels must come from the per-player battingOrder string in teams.X.players: the hundreds digit is the slot, a value ending in "00" is a starter, "01" and up are substitutes into that slot (e.g. "401" = subbed into slot 4). The team-level array is only safe for pregame/live lineups.
- Active rosters: GET /api/v1/teams/{teamId}/roster?rosterType=active (supports date= for historical snapshots).
- Player game logs: GET /api/v1/people/{personId}/stats?stats=gameLog&group=hitting&season=YYYY (and group=pitching); rows carry game.gamePk. Batch trick: GET /api/v1/people?personIds=1,2,3&hydrate=stats(group=[hitting,pitching],type=[gameLog],season=YYYY).
- Platoon splits: GET /api/v1/people/{id}/stats?stats=statSplits&group=hitting&season=YYYY&sitCodes=vl,vr (also careerStatSplits). These are FULL-SEASON splits; see the point-in-time rules in Phase 2 for how they may be used.
- Player universe: GET /api/v1/sports/1/players?season=YYYY (about 1,450 players with id, fullName, currentTeam, primaryPosition, batSide, pitchHand).
- Boxscores (training backfill): GET /api/v1/game/{gamePk}/boxscore.

Hydrate failures are SILENT: a misspelled hydrate returns a normal response with the hydration simply missing. After every hydrated fetch, assert the expected keys exist and log a warning if not.

### Training backfill (one-time script: scripts/build_training_backfill.py)

Build the historical per-game log table from the Stats API itself so training and inference share identical personIds and schemas. Backfill seasons 2019 through 2025 (2019-2020 exist only to feed Marcel priors for 2021-2022 training rows; training itself uses 2021-2025). For each season: enumerate games from /schedule over the season date range (regular season only, gameType R), then fetch /game/{gamePk}/boxscore for each final game.

TRAP (verified on 2024): the schedule range contains postponed/rescheduled DUPLICATES. 2024 has 2,469 game entries but 2,430 unique gamePks; 37 gamePks appear on two or more dates, and the boxscore response carries no date, so taking the date from the schedule iteration assigns April dates to games actually played months later, silently corrupting every point-in-time feature downstream. Dedupe by gamePk, keep only entries with status.codedGameState == "F", and take game_date from that Final entry's officialDate.

Throttle to 2-3 requests/second, cache every raw boxscore JSON to disk keyed by gamePk so the backfill is resumable and only ever runs once, and write the parsed result to cache/gamelogs_{season}_v1.parquet with one row per (gamePk, personId, side): full hitting and pitching lines, pitch counts (needed for a Phase 2 feature), batting-order slot AND a starter/substitute flag from the per-player battingOrder string, a played flag (keep non-participating dressed players as rows with played=False; training filters on it, the UI roster logic uses it), home/away, opposing starter personId and throwing hand, venue id, officialDate, gameNumber, day/night. Licensing note: keep the throttle polite; MLBAM's terms permit individual, non-commercial, non-bulk use, so this stays a free hobby site (no ads, no subscriptions) and the backfill stays gentle.

The daily pipeline (Phase 4) appends each newly completed game's parsed boxscore to cache/gamelogs_2026_v1.parquet USING THE SAME PARSER, so current-season features come from the same schema. No separate gameLog-endpoint code path for features.

### Ancillary sources (scripts/build_context_tables.py, refresh weekly in season)

- Park factors: https://baseballsavant.mlb.com/leaderboard/statcast-park-factors. TRAP (verified live): csv=true does NOT work here; it returns the full HTML page. The 30-row dataset is embedded in the page as a "var data = [...]" JSON array (fields include venue_id, name_display_club, index_1b, index_2b, index_3b, index_hr, index_bb, index_so, index_woba, key_bat_side, n_pa). Fetch the HTML and parse that array with a regex; the batSide=L/R parameter gives handedness splits. Build one file per year, park_factors_{year}_v1.parquet, for every year 2018-2026 (training needs historical years; Phase 2 says which year a row may use).
- Catcher framing: https://baseballsavant.mlb.com/leaderboard/catcher-framing?year=YYYY&csv=true DOES return real CSV (verified). Columns are id, name, pitches, rv_tot, pct_tot and per-zone fields, with no year column: stamp the season into the cached parquet yourself.
- Catcher throwing (for the SB model): https://baseballsavant.mlb.com/leaderboard/catcher-throwing?year=YYYY&csv=true (hyphen in the path; the underscore variant 404s). Verified CSV.

Do NOT build anything on pybaseball's FanGraphs or Baseball-Reference scrapers (unmaintained, 403-prone since May 2025). pybaseball is allowed only for optional Statcast enrichment later (statcast_batter/statcast_pitcher) and playerid_lookup, and nothing in the daily pipeline may depend on it.

### Acceptance checks for Phase 1
- Backfill completes for 2019-2025, one parquet per season; zero duplicate gamePks and zero non-Final rows (assert both in the script); row counts consistent with the full-roster convention (roughly 2,430 games x 2 sides x 26 dressed players, of which 13-18 per side have played=True); spot-check three known games against MLB.com box scores, including a 2024 doubleheader day (two distinct gamePks, correct gameNumber ordering) and one game with a pinch hitter (verify the slot/starter flags come from the per-player battingOrder strings, not the team array).
- Park factor parquets exist for 2018-2026 with 30 venues each; framing and throwing parquets carry a season column.
- A fetch of today's schedule with probables and (if posted) lineups round-trips through the cache: second call within TTL does zero network requests.
- Killing the network (or a monkeypatched failing session) mid-fetch serves stale data with a logged warning, never an exception or empty frame.

## 2.5 Phase 1.5: Slate viewer (Barrett's first visible milestone)

Before any modeling, ship a one-page Streamlit app: date picker (wired exactly per Section 6: seeded once from ?date=, mirrored back to st.query_params), the slate of games for that date with start times in Eastern, probable pitchers, and posted lineups straight from the Phase 1 cache. No predictions yet. This exercises the theme port, the chrome/nav, the stable-picker machinery, and the fetch layer end to end, and gives Barrett a working site within days instead of after the whole model build. Show it to him.

## 3. Phase 2: Feature store (point-in-time, shared by training and inference)

Training rows are (personId, gamePk). The point-in-time rule: every feature must be computable strictly from games with an EARLIER gameDate timestamp than the target game (which handles doubleheaders: game 1 of a same-day pair is usable for game 2), plus reference tables from earlier SEASONS as specified per feature below. This is the number one failure mode of the whole project: a season-to-date average that includes the target game inflates hitter accuracy dramatically and is trivially easy to introduce with a groupby-cumsum off-by-one. mlblib/features.py must be the single implementation used by both scripts/train_*.py and the daily pipeline.

Batter features (per target game):
- Talent prior: Marcel-style multi-season component rates. Hitters: seasons weighted 5/4/3 with about 1200 PA of league-average ballast and a small age adjustment. Where fewer than three prior seasons exist in the backfill (2021 rows see only 2019-2020; 2020 is a 60-game season), use the seasons available with the same weights renormalized; the ballast absorbs the missing history. Note the convention in docs/decisions.md.
- Season-to-date and rolling windows over the team's last 15 and last 30 games, aggregating only games the player appeared in, all strictly before the target game. Shrink small samples toward the prior: weight = n / (n + k) with k from stabilization research (about 60 PA for K%, 120 for BB%, 170 for HR rate, 460 for OBP-like rates).
- Platoon: the batter's split versus the probable starter's throwing hand. Point-in-time rule for splits: the statSplits endpoint returns full-season numbers, so a target-season split would leak future games. Training AND inference both use career-through-prior-season splits (sum the per-season statSplits for seasons strictly before the target season), regressed toward the league-average platoon effect by sample size. Never raw small-sample splits.
- Opposing starter quality strictly through the prior day: K%, BB%, HR/9 style rates from the backfill logs, his handedness, plus his own Marcel prior. Pitcher Marcel differs from hitters (verified against Tango's Marcel): weights are 3/2/1 with an innings/BF-scaled regression ballast, not 5/4/3 with 1200 PA; pick the ballast constant, document it in decisions.md.
- Opposing team pitching context: overall staff K% and wOBA allowed through the prior day. Do not attempt team-vs-handedness aggregates in v1 (the backfill has no per-PA pitcher-hand detail; noted as a possible v2 item).
- Context: park factors for the venue, by batter handedness. Point-in-time rule: training rows for season Y use the park factor file of season Y-1 (or a 3-year window ending at Y-1); inference uses the latest published file. Also: home/away, month of the year (Barrett asked for this explicitly; encode as integer month and let HistGB split on it), day/night, batting-order slot (posted or projected lineup at inference; the per-player battingOrder strings in training), rest days since the player's last game, team runs-per-game through the prior day.
- Catchers (Barrett's "supporting catcher"): the opposing catcher's framing runs and throwing metrics (feeds SO/BB margins and especially the SB model). Same prior-season rule as park factors.

Pitcher features (starters): same shape. Talent priors (3/2/1) and rolling windows for K%, BB%, contact quality allowed; the opposing team's lineup aggregate (team K%, wOBA through the prior day); park; month; rest days and days since last start; season pitch-count trend (pitch counts are in the backfill); his own catcher's framing runs.

Explicitly EXCLUDE (noise, per the best professional systems): batter-vs-pitcher history, hot-streak indicators beyond the rolling windows, un-regressed splits, career-in-park numbers.

Missing values stay as NaN (HistGB routes NaNs natively; no imputation). Categorical features (handedness matchup, home/away) can use native categorical support.

### Acceptance checks for Phase 2
- Leakage test in tests/ (pytest): for a sample of 200 random (personId, gamePk) rows, recompute every rolling feature from the raw logs filtered to gameDate strictly earlier than the target game's gameDate and assert equality with the feature store's values. This convention intentionally matches the doubleheader rule (game 1 counts for game 2); test it explicitly on a real 2024 doubleheader.
- Split/park/catcher tables: assert no feature for a season-Y row reads a season-Y reference file.
- A September call-up with 0 prior MLB PA gets pure-prior features (league mean), not NaN-everything or a crash.

## 4. Phase 3: Models

### 4.1 Decomposition: opportunities x rates
Never regress a counting stat directly against context (exceptions below are deliberate); predict opportunities and rates separately, then multiply.
- Batters: model E[PA] from batting-order slot, team offense, park, opposing starter (slot is worth about 0.10-0.11 PA per game; leadoff about 4.65 PA/game, ninth about 3.8). Then per-PA component rates. Counting predictions = E[PA] x rate. Derived stats come from the canonical table in Section 0.
- Starting pitchers: model E[outs recorded] and E[batters faced], then per-BF rates for K, BB, H, HR.
- Deliberate exceptions modeled directly per game with Poisson loss: R, RBI (team-dependent), SB, and pitcher ER.

Poisson mechanics in sklearn (decide once, record in decisions.md): HistGradientBoostingRegressor has no exposure/offset parameter, so rate models are trained as y = count / opportunities with sample_weight = opportunities and loss="poisson" (the standard GLM weighting trick; 0-PA rows carry zero weight and drop out naturally). Opportunity models (PA, outs, BF) and the direct-exception targets train on raw counts with loss="poisson". Squared error is wrong for all of these right-skewed targets.

### 4.2 Estimators
One HistGradientBoostingRegressor per target from the Section 0 table. Optionally add loss="quantile" (0.1/0.9) twins later for floor/ceiling display; not in v1. Start near HoopsValue's proven hyperparameters (max_iter 500-800, max_depth 3-4, learning_rate 0.02-0.05, min_samples_leaf 25-50, l2_regularization 0.1) and tune modestly with the temporal validation below; do not burn days on hyperparameter search, the variance ceiling makes it pointless.

Training-data hygiene: train on played=True rows only; exclude position players pitching; flag and separate openers/bulk relievers from true starters (innings expectations differ wildly); relief pitchers are OUT of v1 scope for per-game lines; two-way players contribute both a batter row and a pitcher row.

### 4.3 Artifacts
scripts/train_models.py trains everything and persists ONE artifact per target as a joblib dict: {model, feature_cols, model_class, target, loss, train_end_date, MODEL_VERSION, data_manifest} where data_manifest lists the input cache files and row counts the model was trained from (do not try to store the artifact's own file hash inside itself; downstream pipeline outputs stamp the sha1 of the joblib FILE, the HoopsValue build_player_hub.py pattern). models/{target}_histgb_m1.joblib. The feature-vector assembly lives in mlblib/model.py and both trainer and app import it; put the "feature order MUST match" contract comment on both sides like HoopsValue does. The app loads artifacts once via @st.cache_resource with a graceful fallback (show SENTINEL rows and a "model unavailable" note, never crash).

### 4.4 Validation (walk-forward only)
Random K-fold is forbidden; it leaks player-season level and gives fake numbers. Train on 2021-2023, validate on 2024 for tuning; final test on 2025 exactly once. scripts/validate_models.py reports, per target: Poisson deviance, MAE, and calibration by prediction decile (predicted mean vs realized mean per bucket).

### 4.5 Mandatory baselines and the ship gate
1. League-average constant per target.
2. Player season-to-date per-game mean, excluding the target game.
3. Marcel-style weighted multi-season rate x expected opportunities.
Gate: for each target, the model should beat baselines 2 and 3 on MAE and Poisson deviance out of sample. A target that fails is NOT silently dropped and does not block the site: it goes to the Section 8 STOP with a recommendation, and Barrett decides between dropping the column and displaying it with a "no better than a season average" caveat. He asked for every statistic; honesty about which ones carry signal is the compromise, and it is his call. Expect the margins on batter targets to be small in absolute terms; document them in docs/model_report.md rather than chasing them. Low-frequency targets (SB especially) can "win" MAE by predicting near zero, so weigh deviance and calibration over MAE for those.

### Acceptance checks for Phase 3
- validate_models.py writes docs/model_report.md: it must OPEN with a plain-English per-target summary Barrett can act on ("pitcher K has real signal; RBI barely beats a season average; recommendation: show with caveat"), followed by the full tables (every target vs all three baselines on the 2025 test season) and decile calibration plots.
- Pitcher K model: MAE clearly under baseline 2 and calibration monotone across deciles (this is the target with real signal; if it fails, something is broken upstream, stop and debug).
- A smoke test proving trainer and app produce identical predictions for the same feature row (feature-order contract).

## 5. Phase 4: Daily pipeline

scripts/build_daily_predictions.py (offline, same hardening as HoopsValue's build_player_hub.py: socket.setdefaulttimeout(30), faulthandler watchdog, periodic flushes, and the sha1 of each model joblib stamped into the output):
1. Fetch the slate for the target date with probables and lineups.
2. Fetch/refresh active rosters for the 30 teams (cheap, cache with short TTL).
3. Append any newly completed games to cache/gamelogs_2026_v1.parquet using the Phase 1 parser, then build inference feature rows for every relevant player through the prior game.
4. Predict every target, assemble one tidy output: cache/predictions_{YYYY_MM_DD}_m{MODEL_VERSION}.parquet plus a small slate JSON with game metadata and a lineup_status flag per team (confirmed vs projected). The predictions parquet also carries the handful of headline feature inputs per row (Marcel prior, last-30 form, slot, opposing starter id and hand) so the Player page can show "what the model saw" without recomputing features at request time.
5. The Streamlit pages ONLY read these files. Never run models at request time for list views.

Lineup reality (design around it, do not fight it): probable pitchers appear days ahead and are firm by game-day morning; starting lineups post whenever clubs announce, usually 1 to 4 hours before first pitch, and scratches happen up to game time. So: run the pipeline in the morning with PROJECTED lineups (the most common batting order over the team's last 10 games, filtered to the active roster, with the correct platoon side vs the probable starter). Re-run affected games when real lineups or changed probables appear. Every prediction row carries lineup_status so the UI can badge "projected" vs "confirmed". If a probable pitcher is missing entirely, fall back to a league-average starter of unknown hand and badge it.

Refresh mechanics on Render: simplest v1 is TTL-driven refresh inside the app process for today's slate (a background thread kicked by serve.py, using the stale-beats-empty rules), upgrading later to a Render cron job hitting the same script. Past dates render from their immutable prediction files; if Barrett picks a past date whose file predates final stat corrections (48h grace), rebuild it once.

scripts/build_accuracy_tracker.py: scores past predictions against actuals and appends to one compact committed file, cache/accuracy_history_v1.parquet (per date, per target: model predicted vs actual vs baselines 2 and 3, rolling MAE). The Accuracy page and the Player page's predicted-vs-actual history read ONLY this file, which keeps the fresh-clone-offline guarantee of Phase 6. This is the HoopsValue real-signings-tracker pattern Barrett likes: the projection stays pure, the scoring happens against it.

### Acceptance checks for Phase 4
- Run the pipeline for yesterday's real slate; verify every game appears and every lineup player has predictions for all displayed targets; a standalone script times the parquet read at under 200ms.
- Simulate "lineups not posted": morning run produces projected lineups with the flag set; injecting the posted lineup and re-running flips the flag and changes slot-dependent numbers.
- A doubleheader date renders two separate games with correct gameNumber ordering.

## 6. Phase 5: Streamlit UI

Pages (multipage app, each page starting with the HoopsValue boilerplate: sys.path.insert, st.set_page_config, render_page_chrome(), render_nav()):
1. Home / Slate: the Phase 1.5 page, extended. Date picker (st.date_input, seeded once from ?date=, mirrored back to st.query_params; default today). Below it, the slate: one card or expander per game showing teams, start time (US/Eastern), probable pitchers, lineup status badges. Selecting a game reveals the matchup view: two batter tables (nine lineup rows in order, then bench players in a collapsed "Bench" section) and the starting pitcher lines. Batter columns, from the Section 0 canonical table: slot, player, expected PA, H, 2B, 3B, HR, TB, R, RBI, BB, SO, SB, then derived AVG/OBP/SLG/OPS. Pitcher columns: expected IP, K, BB, H, ER. Missing values render as SENTINEL.
2. Player page: search a player (stable-options rules), see his prediction for the selected date, his recent predicted-vs-actual history (from accuracy_history_v1.parquet), and the headline model inputs carried in the predictions parquet (the transparency Barrett likes).
3. Accuracy: per target, model vs baselines over time, honest, from accuracy_history_v1.parquet.
4. About / Methodology: the honest-expectations text and the deliberate-exclusions list from Section 0, what the model uses, what it deliberately ignores, and the MLBAM attribution.

"Every active player": batters in the posted or projected lineup get full predictions. Bench players (active roster, not in the lineup) get the conditional predictions defined in Section 0 (neutral 6th slot, labeled "per game IF he starts"). Relief pitchers are listed with season-context rates and the "relief usage not modeled in v1" note. Two-way players appear in both tables.

Footer on every page, formatted with the current year at render time: "Statistics data via MLB Stats API. Copyright {year} MLB Advanced Media, L.P. Use of any content acknowledges agreement to the terms at http://gdx.mlb.com/components/copyright.txt". Keep the site free, no ads, no paid tiers; that is what keeps the data usage legitimate.

### Acceptance checks for Phase 5
- Deep link /?date=2026-07-04 renders that slate directly; changing the date via the picker updates the URL.
- Theme toggle persists across page navigation; all colors come from tokens (grep pages/ and mlblib/ for hex literals; there should be none outside theme.py).
- A date with no games (e.g., All-Star break Monday) shows a clean empty state, not an error.
- No em dashes in rendered UI string literals in pages/ and mlblib/ except the SENTINEL constant definition (code comments exempt); no emojis in UI text.

## 7. Phase 6: Deployment (Render)

Copy the HoopsValue serving pattern: Procfile "web: python serve.py"; serve.py warms before opening the port (import heavy modules, seed the disk cache, load model artifacts, warm today's slate) then execs streamlit run app.py. HoopsValue's synthetic-websocket self-warm trick and the SEO/robots/sitemap patches are OUT of v1 scope; port them later if the site goes public. CRITICAL, learned the hard way: for services created from the Render dashboard, render.yaml is IGNORED. The dashboard Build Command (pip install -r requirements.txt) and Start Command (python serve.py) are authoritative and Barrett must set them manually; put a comment block at the top of render.yaml saying exactly this. Persistent disk mounted at /data; CACHE_DIR prefers /data/cache; committed cache seeds gap-fill only (the _vN bump rule from Section 1.10).

Repo size policy (per-game data is ~30x bulkier than HoopsValue's per-season data): commit model artifacts, park factor / framing / throwing tables, the player universe, accuracy_history_v1.parquet, and only the last ~14 days of prediction files. Training backfill parquets and raw boxscore JSONs stay OUT of git (they rebuild from the API); document the rebuild command in scripts/README.md.

### Acceptance checks for Phase 6
- Fresh clone + pip install + streamlit run app.py works locally with committed caches only (no network) for any committed date, including the Player and Accuracy pages.
- On Render: cold deploy serves the current slate without a visible spinner-hang, and a second deploy does not lose /data cache contents.

## 8. Order of work

1. Phase 1 data layer + backfill (most of the calendar time; the backfill can run overnight).
2. Phase 1.5 slate viewer. Show Barrett a working site early.
3. Phase 2 feature store + leakage tests.
4. Phase 3 models + validation report. STOP here and walk Barrett through docs/model_report.md, including any targets that failed the Section 4.5 gate, with your display recommendation per target. He decides the final column set.
5. Phase 4 daily pipeline + accuracy tracker.
6. Phase 5 UI.
7. Phase 6 deploy.

Keep a docs/decisions.md log of anything you chose that this document did not specify, one line each, so Barrett can audit later. When something in this document conflicts with reality (an endpoint changed, a number is off), note it there, adapt, and keep moving; do not silently deviate.
