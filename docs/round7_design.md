# Round 7 design: air-density batted-ball carry

Hypothesis: a fly ball carries farther in thinner air. The model currently
sees game TEMPERATURE (shipped for p_H/p_HR in the weather round; it granted
NOTHING on the batter side) but never sees ELEVATION or PRESSURE, so it cannot
tell that Coors Field (~1580 m, air density ~15% below sea level) turns warning-
track outs into home runs regardless of temperature. Air density is the
physically-correct, altitude-aware version of the temperature feature.

## The quantity

Per game, from the first-pitch hour's temperature T (°F→K), relative humidity
RH (%), and station surface pressure P (hPa, already altitude-adjusted by
Open-Meteo since it is `surface_pressure` at the venue elevation):

  Tc   = (T_F - 32) * 5/9                          # Celsius
  Tk   = Tc + 273.15                               # Kelvin
  Psat = 6.1078 * 10^(7.5*Tc / (Tc + 237.3))       # hPa, Magnus saturation
  Pv   = (RH/100) * Psat                            # hPa vapor pressure
  Pd   = P - Pv                                     # hPa dry-air partial pressure
  rho  = (Pd*100)/(287.058*Tk) + (Pv*100)/(461.495*Tk)   # kg/m^3
  carry_index = 1.225 / rho          # 1.225 = sea-level 15C std; >1 = thin air

Coors should land near carry_index ~1.15-1.20, sea-level cold games near ~0.97.
Both terms are physically standard (humid air is LESS dense -- water vapor is
lighter than dry air -- so high humidity slightly raises carry_index).

`carry_index` and `rho` are collinear (carry_index = 1.225/rho); the block ships
only **carry_index** (one feature = minimal multiple-testing, clean ablation).

## Tables

`cache/game_airdensity_v1.parquet` per gamePk: `carry_index` (and `rho`,
`temp_f_arc`, `rh`, `press` for provenance/debug, not features).

Built by `scripts/build_airdensity_tables.py`:
- Join gamelogs (all seasons) -> per gamePk: officialDate, gameDate (first-
  pitch UTC), venue_id. Attach venue lat/lon from venues_v1.
- Batch by (venue, season): ONE Open-Meteo ARCHIVE call per venue-season
  returns that season's hourly temperature_2m / relative_humidity_2m /
  surface_pressure (UTC). Index each game to its first-pitch UTC hour
  (round gameDate to the nearest hour).
- Compute carry_index; write per gamePk. Games at venues with no coords
  (~4%, spring-training/neutral sites) get no row -> NaN feature -> HistGB
  routes natively. Stale-beats-empty: a failed venue-season call skips those
  games, never writes garbage.

Archive endpoint: https://archive-api.open-meteo.com/v1/archive (keyless,
same provider as the shipped forecast). ~30 venues x 7 seasons ~= 210 calls
with polite sleeps.

## Inference

Extend `fetch.get_forecast` to also return relative_humidity_2m and
surface_pressure per hour (superset; the existing (temp,ws,wd) tuple becomes
(temp,ws,wd,rh,press)). In `build_daily_predictions._game_environment`,
compute carry_index at the first-pitch hour from the forecast for open/
retractable parks; DOME games use a fixed indoor normal (72F, 50% RH, and the
venue's elevation-implied pressure -> a per-dome constant carry_index, since
domes still sit at altitude e.g. Chase Field in Phoenix). Set `carry_index` on
the target frames; `_attach_environment`-style coalesce provided vs table.

## Feature blocks (gauntlet candidates, pre-registered)

- `airdensity_bat` [carry_index] -> {HR, b2, b3}  (carry-sensitive extra bases)
- `airdensity_pit` [carry_index] -> {p_HR, p_H}   (fly-ball damage allowed)

Marginal test: the ablation adds carry_index ON TOP of the current policy,
which already includes the temperature weather block for p_H/p_HR -- so the
pitcher side measures carry_index's value BEYOND temperature (the altitude/
pressure part). The batter side is fresh (weather granted nothing there).

Gauntlet: 2024 ablation (exp_feature_blocks.py) -> per-block 2025 confirmation
on full history -> ship replicators -> retrain -> regen -> tests -> push.
Noise floor +/-0.1% dev. Deterministic-retrain proof: only granted joblibs
change bytes.

## Point-in-time / leakage

Air density is a physical property of the game's conditions, known pre-game
(forecast) and identical at train (archive actual) and inference (forecast).
No player history is involved, so there is no shift/as-of machinery -- it is a
per-gamePk merge like the existing weather columns. The only parity gap is
forecast-vs-actual weather error at inference, the same accepted gap the
shipped temp/wind features already carry.

## Traps

- Open-Meteo `surface_pressure` is at the station's own elevation (NOT sea-
  level), so it already encodes altitude -- do NOT re-adjust for elevation.
  Verify Coors reads low pressure (~840 hPa) as a build-time assertion.
- First-pitch hour: gameDate is UTC ISO; round to nearest hour for the archive
  index. Doubleheaders share a venue-day but differ by hour -- key on gamePk,
  not date.
- Humidity term is small but correct; keep it (Coors also humidifies balls, a
  separate effect not modeled here).
- gitignore: un-ignore cache/game_airdensity_v1.parquet.
- Daily updater: game_airdensity is INFERENCE-time (forecast), computed in the
  daily pipeline like weather; the committed table is training-only and static
  (historical archive), so no in-season upkeep of the table is needed.

## Design-review revisions (applied before the gauntlet)

The 2-lens design review reshaped the round. Key changes:

- **carry_index is largely redundant.** It is dominated by the 1/T term
  (already-shipped temperature, which was tested and REJECTED for batters), and
  its cross-park variation duplicates the Y-1 park HR factor pf_hr (Coors' low
  pressure and Coors' high pf_hr are the same altitude effect). So RAW
  carry_index is expected to sit near the noise floor. The ONLY component
  orthogonal to both temp and pf_hr is the park-demeaned day-to-day anomaly.
  -> Ship BOTH as SEPARATE blocks (airdensity_* = raw carry, carryanom_* =
  game carry minus the venue's climatological mean) so the gauntlet gives a
  clean raw-vs-anomaly verdict. The anomaly is the physically-motivated one.
- **Roof parity keys only on roof TYPE + closed_share, never actual roof.**
  A shared helper F.blend_carry(roof_type, indoor_ci, outdoor_ci, closed_share)
  is called byte-identically by the training table build and the daily
  inference pipeline: Dome -> indoor constant; Retractable -> closed_share
  blend of indoor/outdoor; Open -> outdoor. (The lone fixed dome is Tropicana,
  sea level; Chase Field is Retractable.)
- **Elevation is FEET** in venues_v1 (Coors 5190) -> station_pressure_hpa
  converts ft->m. Verified: Coors ~837 hPa, carry ~1.24 (clear top park;
  next park ~1.09; anomaly sd ~0.02 = the few-percent within-park swing).
- **Drop b3** (thin air REDUCES triples -- deep-gap would-be triples become
  HRs). Tight pre-registered targets: batter {HR, b2}, pitcher {p_HR, p_H}.
- **Hour key TRUNCATES** to the UTC hour on both sides (matches the shipped
  temp/wind inference convention), not rounds.
- Deferred (low impact): Estadio Alfredo Harp Helu (Mexico City, ~7350 ft --
  the single most extreme park) lacks coords in venues_v1, so its ~4 games get
  NaN carry. Negligible vs Coors' 514 games; noted for a future coords patch.
