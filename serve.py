"""Production entrypoint (Render Start Command: python serve.py).

Warms the process before opening the port so the first visitor never waits on a
cold import or disk seed: pre-imports the heavy modules, seeds the disk cache
(import side effect), loads the model artifacts, and warms today's prediction
files in a background thread. Then it execs `streamlit run app.py` on $PORT.

HoopsValue's synthetic-websocket self-warm and SEO/robots patches are out of
scope for v1; port them later if the site goes public.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _warm():
    try:
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
        from mlblib import cache, model as M, store  # noqa: F401
        from mlblib.util import today_iso

        # Seeding happens on `import cache`. Load artifacts so they're resident.
        M.load_artifacts(list(M.BAT_TARGETS) + list(M.PIT_TARGETS))
        # Warm today's prediction file if it exists (read into page cache).
        store.load_predictions(today_iso())
        cache.logger.info("warm complete")
    except Exception as e:  # noqa: BLE001 — warming is best-effort
        print(f"[serve] warm skipped: {e}", flush=True)


def main() -> None:
    threading.Thread(target=_warm, daemon=True).start()
    port = os.environ.get("PORT", "8501")
    sys.argv = [
        "streamlit", "run", str(ROOT / "app.py"),
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ]
    from streamlit.web.cli import main as st_main
    sys.exit(st_main())


if __name__ == "__main__":
    main()
