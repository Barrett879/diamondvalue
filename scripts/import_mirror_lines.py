"""One-off backfill: grade the model against historical PrizePicks MLB lines.

WHERE THE LINES COME FROM, AND THE CATCH
----------------------------------------
PrizePicks publishes no line history and blocks automated access, so the lines
here come from a third-party public mirror that commits a daily MLB snapshot:
  https://github.com/enkday/prizepicks-data-mirror
Its working tree overwrites daily, but git history preserves every version, so
every past snapshot is recoverable.

That repo has NO LICENCE and is openly an automated mirror of the PrizePicks
API, which means the data derives from collection PrizePicks' terms prohibit.
It is therefore fine for private curiosity and NOT safe to republish. Two
consequences are enforced here:
  - the per-line graded history stays OUT of git (see .gitignore)
  - only a derived AGGREGATE (counts and hit rates, no line values) is
    committed, and that is what the public Accuracy page reads

WHERE THE PREDICTIONS COME FROM
-------------------------------
The daily pipeline prunes prediction files after 21 days, so July is long gone
from the working tree. But those files were committed, so the ORIGINAL
point-in-time projection for each date is recoverable from git history. This
uses the OLDEST add of each file, i.e. the projection as first published, not
any later regeneration.

SNAPSHOT CHOICE
---------------
For each (date, player, stat) this keeps the LATEST snapshot taken strictly
BEFORE first pitch. That is the closest thing to a closing line the mirror
offers, and it is the harder, more honest test: an earlier snapshot is softer
because the market has had less time to absorb lineups and weather.

Usage:
  python scripts/import_mirror_lines.py /path/to/prizepicks-data-mirror
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import cache, props, store  # noqa: E402
from mlblib.cache import logger  # noqa: E402

import scripts.build_props_tracker as bpt  # noqa: E402

MIRROR_FILES = ["data/prizepicks-mlb.json", "data/prizepicks-mlb-next-7-days.json"]
_ET = props._ET


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True).stdout


def extract_mirror_props(clone: Path) -> dict[str, pd.DataFrame]:
    """{game_date_ET: lines frame} from the mirror's whole git history."""
    best: dict[tuple, tuple] = {}
    for f in MIRROR_FILES:
        for sha in _git(["log", "--format=%H", "--", f], clone).split():
            blob = _git(["show", f"{sha}:{f}"], clone)
            if not blob.strip():
                continue
            try:
                payload = json.loads(blob)
            except ValueError:
                continue
            scraped = payload.get("scrapedAt")
            if not scraped:
                continue
            try:
                s_at = datetime.fromisoformat(scraped.replace("Z", "+00:00"))
            except ValueError:
                continue
            for p in payload.get("props", []):
                iso = p.get("startTimeIso")
                if not iso:
                    continue
                try:
                    start = datetime.fromisoformat(iso)
                except ValueError:
                    continue
                if s_at >= start:        # scraped after first pitch, not pregame
                    continue
                gdate = (start.astimezone(_ET) if _ET else start).date().isoformat()
                key = (gdate, p.get("player"), p.get("stat"))
                prev = best.get(key)
                if prev is None or s_at > prev[0]:
                    best[key] = (s_at, p)

    by_date: dict[str, list] = {}
    for (gdate, player, stat), (s_at, p) in best.items():
        try:
            line = float(p.get("line"))
        except (TypeError, ValueError):
            continue
        by_date.setdefault(gdate, []).append({
            "name": player, "team": p.get("teamCode"), "stat_type": stat,
            "line": line, "start_time": p.get("startTimeIso"),
            # The mirror carries only standard lines and no Less/More buttons,
            # so direction is unknown -> permissive (see props._offered_sides).
            "direction": "", "odds_type": (p.get("oddsType") or "standard"),
            "scraped_at": s_at.isoformat(),
        })
    return {d: pd.DataFrame(rows) for d, rows in by_date.items()}


def recover_predictions(date: str, repo: Path) -> pd.DataFrame | None:
    """The ORIGINAL committed prediction file for `date`, from git history."""
    name = f"cache/predictions_{date.replace('-', '_')}_m1.parquet"
    adds = _git(["log", "--all", "--diff-filter=A", "--format=%H", "--", name],
                repo).split()
    if not adds:
        return None
    blob = subprocess.run(["git", "show", f"{adds[-1]}:{name}"], cwd=repo,
                          capture_output=True)      # oldest add = as published
    if not blob.stdout:
        return None
    tmp = cache.CACHE_DIR / f".recovered_{date}.parquet"
    try:
        tmp.write_bytes(blob.stdout)
        df = pd.read_parquet(tmp)
    except Exception:  # noqa: BLE001
        return None
    finally:
        tmp.unlink(missing_ok=True)
    return store._repair_names(df)   # July bug: raw ids leaked in as names


def main(argv):
    if not argv:
        raise SystemExit("usage: import_mirror_lines.py /path/to/mirror/clone")
    clone = Path(argv[0]).expanduser().resolve()
    if not (clone / ".git").is_dir():
        raise SystemExit(f"not a git clone: {clone}")
    repo = Path(__file__).resolve().parent.parent

    logger.warning("walking mirror history at %s", clone)
    lines_by_date = extract_mirror_props(clone)
    logger.warning("mirror: %d dates, %d props", len(lines_by_date),
                   sum(len(v) for v in lines_by_date.values()))

    frames, skipped = [], []
    for date in sorted(lines_by_date):
        preds = recover_predictions(date, repo)
        if preds is None or preds.empty:
            skipped.append((date, "no predictions"))
            continue
        actuals = store.load_actuals(date)
        if actuals is None or actuals.empty:
            skipped.append((date, "no actuals"))
            continue
        graded = bpt.grade(lines_by_date[date], preds, actuals, date)
        if graded.empty:
            skipped.append((date, "nothing gradeable"))
            continue
        frames.append(graded)
        logger.warning("%s: %d graded of %d posted", date,
                       int(graded["graded"].sum()), len(graded))

    if not frames:
        raise SystemExit("nothing graded")
    hist = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["date", "player", "stat"], keep="last")
    cache.atomic_to_parquet(hist, cache.dc_path("props_history_v1.parquet"))
    logger.warning("wrote props_history_v1.parquet (%d rows, PRIVATE)", len(hist))
    write_summary(hist)
    if skipped:
        logger.warning("skipped %d dates: %s", len(skipped), skipped[:8])


def write_summary(hist: pd.DataFrame) -> None:
    """Committed AGGREGATE only: counts and hit rates, never the line values.
    This is what the public page reads, so the mirror's data is not republished."""
    g = hist[hist["graded"]]
    conf = (g["p_over"] - 0.5).abs()
    buckets = []
    for label, lo, hi in (("coin flip (50-55%)", 0.0, 0.05), ("55-60%", 0.05, 0.10),
                          ("60-70%", 0.10, 0.20), ("70%+", 0.20, 1.0)):
        sel = g[(conf >= lo) & (conf < hi)] if hi < 1.0 else g[conf >= lo]
        if len(sel):
            buckets.append({"bucket": label, "n": int(len(sel)),
                            "right": int(sel["model_right"].sum()),
                            "hit_pct": round(100 * sel["model_right"].mean(), 1)})
    # Top-confidence slices: the practical question, "how good are the model's
    # strongest calls". Ranked by |P(Over) - 0.5|, NOT by the raw mean-vs-line
    # gap, which ranks them backwards (see exp_market_blend.py).
    import math as _m
    conf_all = (g["p_over"] - 0.5).abs()
    slices = []
    for label, q in (("top 5%", 0.95), ("top 10%", 0.90), ("top 20%", 0.80),
                     ("top 25%", 0.75)):
        sel = g[conf_all >= conf_all.quantile(q)]
        if len(sel) < 50:
            continue
        n = len(sel); pr = sel["model_right"].mean()
        se = _m.sqrt(pr * (1 - pr) / n) * _m.sqrt(1.9)   # clustering design effect
        slices.append({"slice": label, "n": int(n),
                       "hit_pct": round(100 * pr, 1),
                       "ci_lo": round(100 * (pr - 1.96 * se), 1),
                       "ci_hi": round(100 * (pr + 1.96 * se), 1)})
    # THE CONTROL that keeps the top-slice number honest: this board's
    # outcomes are not 50/50. Unders hit 52.7% unconditionally, so a lean set
    # that is 75% Unders looks skilled without being skilled. Report the base
    # rate and each direction against its own always-that-side benchmark.
    base_over = float((g["result"] == "over").mean())
    by_dir = []
    for lean, bench in (("Over", base_over), ("Under", 1.0 - base_over)):
        sel = g[g["lean"] == lean]
        if not len(sel):
            continue
        pr = sel["model_right"].mean()
        by_dir.append({"lean": lean, "n": int(len(sel)),
                       "hit_pct": round(100 * pr, 1),
                       "benchmark_pct": round(100 * bench, 1),
                       "skill_pts": round(100 * (pr - bench), 1)})
    by_stat = []
    for stat, sel in g.groupby("stat"):
        by_stat.append({"stat": stat, "n": int(len(sel)),
                        "hit_pct": round(100 * sel["model_right"].mean(), 1)})
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "third-party public mirror of the PrizePicks API; "
                  "aggregates only, line values are not published",
        "dates": int(hist["date"].nunique()),
        "date_min": str(hist["date"].min()), "date_max": str(hist["date"].max()),
        "posted": int(len(hist)), "graded": int(len(g)),
        "right": int(g["model_right"].sum()),
        "hit_pct": round(100 * g["model_right"].mean(), 1) if len(g) else None,
        "exact": int((hist["result"] == "exact").sum()),
        "by_confidence": buckets,
        "by_top_slice": slices,
        "base_over_pct": round(100 * base_over, 1),
        "always_under_pct": round(100 * (1 - base_over), 1),
        "by_direction": by_dir,
        "all_standard": bool((hist["odds_type"] == "standard").all()),
        "by_stat": sorted(by_stat, key=lambda r: -r["n"]),
    }
    cache.json_save(cache.dc_path("props_summary_v1.json"), summary)
    logger.warning("wrote props_summary_v1.json (aggregates, committed)")


if __name__ == "__main__":
    main(sys.argv[1:])
