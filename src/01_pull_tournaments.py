"""
01_pull_tournaments.py

Pull tournament metadata from start.gg for a given time frame
on a monthly basis. Store tournament data in per-month .json files.
"""

from data_pull import StartGG, iter_us_tournaments
from datetime import datetime, timezone
import json
from pathlib import Path

def month_windows(start_year=2019, end_year=2026):
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            a = datetime(y, m, 1, tzinfo=timezone.utc)
            b = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
            yield f"{y}-{m:02d}", int(a.timestamp()), int(b.timestamp())

out = Path("data/raw/tournaments")
out.mkdir(parents=True, exist_ok=True)
gg = StartGG()
for label, a, b in month_windows():
    f = out / f"{label}.json"
    # Skip months already saved
    if f.exists():        
        continue
    rows = list(iter_us_tournaments(gg, a, b, per_page=50))
    f.write_text(json.dumps(rows))
    print(label, len(rows))