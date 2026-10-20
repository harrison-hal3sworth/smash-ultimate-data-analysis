# Claude generated script for pulling tournament metadata for given timeframe

"""
01_pull_tournaments.py

Pull tournament metadata from start.gg for a given time frame
on a monthly basis. Store tournament data in per-month .json files.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import os

from tqdm import tqdm

from data_pull import StartGG, iter_us_tournaments

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "tournaments"
PER_PAGE = 50


def month_windows(start_year=2019, end_year=2026):
    now = datetime.now(timezone.utc)
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            a = datetime(y, m, 1, tzinfo=timezone.utc)
            if a > now:                      # don't pull months that haven't started
                return
            b = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
            yield f"{y}-{m:02d}", int(a.timestamp()), int(b.timestamp())


def save_json(path: Path, rows: list) -> None:
    """Atomic write so an interrupted run never leaves a half-written month."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows), encoding="utf-8")
    os.replace(tmp, path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tmp in OUT_DIR.glob("*.json.tmp"):
        tmp.unlink()

    months = list(month_windows())
    todo = [(label, a, b) for label, a, b in months if not (OUT_DIR / f"{label}.json").exists()]
    print(f"{len(months)} months in range, {len(months) - len(todo)} already saved, {len(todo)} to pull\n")
    if not todo:
        return

    gg = StartGG(log=tqdm.write)
    total_tournaments = 0

    months_bar = tqdm(todo, desc="Months", unit="mo", position=0)
    for label, a, b in months_bar:
        months_bar.set_postfix(month=label, tournaments=f"{total_tournaments:,}")
        pages_bar = tqdm(desc=f"  {label}", unit="page", position=1, leave=False)

        def on_page(page, total_pages):
            pages_bar.total = total_pages
            pages_bar.update(1)

        try:
            rows = list(iter_us_tournaments(gg, a, b, per_page=PER_PAGE, on_page=on_page))
        except KeyboardInterrupt:
            pages_bar.close()
            months_bar.close()
            print(f"\nStopped during {label}; it was not saved. Re-run to resume.")
            return
        finally:
            pages_bar.close()

        save_json(OUT_DIR / f"{label}.json", rows)
        total_tournaments += len(rows)
        tqdm.write(f"{label}: {len(rows):,} tournaments")

    months_bar.close()
    print(f"\nDone: {total_tournaments:,} tournaments, "
          f"{gg.n_requests:,} requests, {gg.n_retries:,} retries")


if __name__ == "__main__":
    main()