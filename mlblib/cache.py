"""Disk-cache toolkit for DiamondValue.

Ported nearly verbatim from HoopsValue's utils.py cache section. The one
function rewritten for this project is `dc_fresh`, whose freshness rule is
keyed on GAME DATE rather than NBA season (see the docstring there).

Design invariants carried over from HoopsValue:
  - CACHE_DIR uses Render's persistent disk (/data/cache) when mounted, else a
    repo-local ./cache that is wiped on ephemeral restarts.
  - seed_disk_cache_from_repo() gap-fills the persistent disk from the committed
    repo snapshot ONCE per process at import, never clobbering fresher files.
    Shipping a CHANGED committed cache file therefore requires a filename
    version bump (_vN) or the disk copy wins forever.
  - Every write is atomic (tmp sibling + os.replace) so a process killed
    mid-write (deploy SIGTERM, a one-shot script exiting) can never leave a
    truncated file whose fresh mtime fools the stale-beats-empty logic.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import threading
import time
from pathlib import Path

import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
# One named logger so cache misses and fetch failures never disappear silently.
# Set DIAMONDVALUE_LOG=DEBUG for verbose output.
logger = logging.getLogger("diamondvalue")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[diamondvalue] %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(os.environ.get("DIAMONDVALUE_LOG", "WARNING").upper())

# ── CACHE_DIR ────────────────────────────────────────────────────────────────
# Render: attach a disk, mount path /data, size ~1 GB. Locally / ephemeral: repo
# ./cache (persists in the repo for committed snapshot files).
_RENDER_DISK = Path("/data/cache")
_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _RENDER_DISK if _RENDER_DISK.parent.exists() else _REPO_ROOT / "cache"


def seed_disk_cache_from_repo() -> None:
    """Copy the committed repo cache into the persistent disk for any files the
    disk does not already have. Gap-fill only, best-effort, never fatal.

    On Render CACHE_DIR is /data/cache; a fresh disk would otherwise force cold
    fetches on first load. This seeds it from the deploy image's committed
    snapshot. Because it never overwrites, a changed committed file with the
    same name never reaches an already-seeded disk: bump the filename version.
    """
    repo_cache = _REPO_ROOT / "cache"
    if CACHE_DIR == repo_cache or not repo_cache.is_dir():
        return  # local/dev already reads the repo cache directly
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        have = set(os.listdir(CACHE_DIR))
        copied = 0
        for src in repo_cache.iterdir():
            if src.is_file() and src.name not in have:
                shutil.copy2(src, CACHE_DIR / src.name)
                copied += 1
        if copied:
            logger.info("seeded %d cache files from repo -> %s", copied, CACHE_DIR)
    except Exception as e:  # noqa: BLE001 — seeding is best-effort, never fatal
        logger.warning("disk-cache seed skipped: %s", e)


seed_disk_cache_from_repo()


# ── Path + freshness ─────────────────────────────────────────────────────────
def dc_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def dc_fresh(
    path: Path,
    game_date: str | None = None,
    today: str | None = None,
    ttl: int | None = None,
) -> bool:
    """Freshness rule for a date-keyed sports site.

    This is the rewritten analog of HoopsValue's season-based `_dc_fresh`.

    Priority:
      1. Explicit `ttl` (seconds) always wins.
      2. If `game_date` is given: past dates are IMMUTABLE only after a 48h
         grace window (MLB posts stat corrections for a day or two after a
         game). Today and future dates get a short TTL (probables/lineups move).
      3. No date context: fall back to a 1h TTL.

    `game_date` and `today` are ISO 'YYYY-MM-DD' strings. Filename version bumps
    (_vN) are how we invalidate immutable past-date files, never edits in place.
    """
    if not path.exists():
        return False
    if ttl is not None:
        return (time.time() - path.stat().st_mtime) < ttl

    if game_date is not None:
        _today = today or time.strftime("%Y-%m-%d", time.localtime())
        if game_date < _today:
            # A past date. Immutable once it is at least ~2 days old; within the
            # grace window keep a 1h TTL so corrected box scores still refresh.
            age_days = _days_between(game_date, _today)
            if age_days >= 2:
                return True
            return (time.time() - path.stat().st_mtime) < 3600
        # today or a future date — short TTL, lineups and probables change.
        return (time.time() - path.stat().st_mtime) < 1800

    return (time.time() - path.stat().st_mtime) < 3600


def _days_between(d0: str, d1: str) -> int:
    """Whole days from ISO date d0 to d1 (d1 - d0). Cheap, no tz math."""
    import datetime as _dt

    a = _dt.date.fromisoformat(d0)
    b = _dt.date.fromisoformat(d1)
    return (b - a).days


# ── Atomic read/write helpers ────────────────────────────────────────────────
def _tmp_sibling(path: Path) -> Path:
    return path.with_suffix(path.suffix + f".tmp{os.getpid()}-{threading.get_ident()}")


def atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet atomically (tmp file + os.replace on the same fs)."""
    tmp = _tmp_sibling(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_parquet_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001 — a truncated/corrupt file is not fatal
        logger.warning("parquet read failed for %s: %s", path, e)
        return None


def pkl_load(path: Path):
    return pickle.loads(path.read_bytes())


def pkl_save(path: Path, obj) -> None:
    tmp = _tmp_sibling(path)
    try:
        tmp.write_bytes(pickle.dumps(obj))
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        logger.warning("cache write failed for %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def json_load(path: Path):
    return json.loads(path.read_text())


def json_save(path: Path, obj) -> None:
    tmp = _tmp_sibling(path)
    try:
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        logger.warning("cache write failed for %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
