---
title: DiamondValue
emoji: ⚾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DiamondValue

Per-game player projections for every MLB slate. Pick a date, open a game, and
see the expected statistics for every available player on both rosters:
batters (PA, hits, home runs, total bases, runs, RBI, walks, strikeouts, stolen
bases, and derived AVG/OBP/SLG/OPS) and starting pitchers (IP, K, BB, H, ER).

Every number is an expected value, the mean of a distribution, not a prediction
of what will happen. Predictions come from one scikit-learn
HistGradientBoostingRegressor per stat (Poisson loss), trained on multi-season
per-game logs with strictly point-in-time features: regressed multi-season
talent priors, recent form, the opposing starting pitcher, platoon matchup,
ballpark, month of the season, lineup slot, and rest.

See the About page in the app for the full methodology and the honest
expectations. Statistics data via the MLB Stats API. This is an independent,
non-commercial project and is not affiliated with or endorsed by Major League
Baseball.
