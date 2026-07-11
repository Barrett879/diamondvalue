# scripts/

Run everything from the repo root (each script does
`sys.path.insert(0, <repo root>)` and imports `mlblib`).

Naming discipline:
- `build_*`  build cache files or model input tables
- `train_*`  train models and write joblib artifacts to `models/`
- `validate_*` / `backtest_*`  out-of-sample checks and reports
- `exp_*`  research probes (kept even when the idea is rejected)

## One-time / occasional

| Script | What it does |
|---|---|
| `build_training_backfill.py [seasons\|all]` | Fetch every regular-season boxscore for 2019-2025 and write `cache/gamelogs_{season}_v1.parquet`. Resumable (raw JSON cached under `cache/raw_boxscores/`). This is the long one (~1.5 h). Rebuild: `python scripts/build_training_backfill.py all`. |
| `build_context_tables.py [years]` | Park factors (embedded JSON), catcher framing/throwing (CSV) from Baseball Savant, years 2018-2026 -> `cache/park_factors_*`, `cache/catcher_*`. Refresh the current year weekly in season. |
| `train_models.py [--bat\|--pit]` | Train the shipped models on 2021-2025 (2019-2020 feed priors) and write `models/{target}_histgb_m1.joblib`. |
| `validate_models.py` | Walk-forward report (train 2021-2023, test 2025) -> `docs/model_report.md`. |

## Daily / per-slate

| Script | What it does |
|---|---|
| `build_daily_predictions.py [YYYY-MM-DD]` | Fetch the slate, append any newly completed games to the current-season gamelog, build point-in-time features, predict every target, write `cache/predictions_{date}_m1.parquet` + `cache/slate_pred_{date}_v1.json`. Defaults to today. |
| `build_accuracy_tracker.py YYYY-MM-DD ...` | Score past predictions against actual box scores -> `cache/accuracy_history_v1.parquet` (read by the Accuracy and Player pages). |

## Rebuild-from-scratch order

```
python scripts/build_context_tables.py
python scripts/build_training_backfill.py all
python scripts/train_models.py
python scripts/validate_models.py
python scripts/build_daily_predictions.py            # today's slate
```

`cache/gamelogs_*.parquet` and `cache/raw_boxscores/` are NOT committed (they
rebuild from the API). Model artifacts, context tables, the player universe,
`accuracy_history_v1.parquet`, and the last ~14 days of prediction files ARE
committed.
