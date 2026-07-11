"""One-shot daily refresh, designed to run unattended (GitHub Action cron).

Does, in order:
  1. Generate predictions for TODAY (US/Eastern) — this also appends any newly
     completed games to the current season's gamelog parquet.
  2. Score YESTERDAY's predictions against actual results (accuracy tracker).
  3. Prune prediction/slate files older than KEEP_DAYS (the accuracy history
     keeps their scores; the 2025 demo dates are kept as permanent examples).

The Action commits whatever changed under cache/ afterward; Streamlit Cloud
redeploys on the push. Exit code is 0 even when a step degrades gracefully
(e.g. no games today), so the cron does not page anyone; hard errors still
raise and fail the run visibly.

Usage: python scripts/daily_update.py [YYYY-MM-DD]   # date override for tests
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache  # noqa: E402
from mlblib.cache import logger  # noqa: E402

KEEP_DAYS = 21
KEEP_ALWAYS = {"2025_07_08", "2025_07_09", "2025_07_10"}  # demo dates


def _today_et() -> dt.date:
    return dt.datetime.now(ZoneInfo("America/New_York")).date()


def prune_old_predictions(today: dt.date) -> None:
    pat = re.compile(r"^(predictions|slate_pred)_(\d{4})_(\d{2})_(\d{2})_")
    cutoff = today - dt.timedelta(days=KEEP_DAYS)
    for p in sorted(cache.CACHE_DIR.glob("*.parquet")) + sorted(cache.CACHE_DIR.glob("*.json")):
        m = pat.match(p.name)
        if not m:
            continue
        tag = f"{m.group(2)}_{m.group(3)}_{m.group(4)}"
        if tag in KEEP_ALWAYS:
            continue
        try:
            d = dt.date(int(m.group(2)), int(m.group(3)), int(m.group(4)))
        except ValueError:
            continue
        if d < cutoff:
            logger.warning("pruning %s (older than %d days)", p.name, KEEP_DAYS)
            p.unlink(missing_ok=True)


def main(argv: list[str]) -> None:
    today = dt.date.fromisoformat(argv[0]) if argv else _today_et()
    yesterday = today - dt.timedelta(days=1)

    import scripts.build_daily_predictions as bdp
    logger.warning("daily update for %s (ET)", today)
    bdp.main([today.isoformat()])

    import scripts.build_accuracy_tracker as bat
    try:
        bat.main([yesterday.isoformat()])
    except SystemExit:
        logger.warning("no predictions to score for %s", yesterday)

    prune_old_predictions(today)
    logger.warning("daily update complete")


if __name__ == "__main__":
    main(sys.argv[1:])
