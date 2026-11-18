# Contains Claude generated code

"""
06_fetch_icons.py  —  character icon URLs from start.gg, for the dashboard.

One API request. Writes data/models/character_icons.csv (char, url, type).
Images are start.gg-hosted Nintendo artwork: link to them, credit start.gg,
and don't redistribute them commercially.

Run from the project root:  python src/06_fetch_icons.py
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pull import StartGG, ULTIMATE_ID  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "models" / "character_icons.csv"

QUERY = """
query Characters($id: ID!) {
  videogame(id: $id) {
    characters { id name images { url type } }
  }
}
"""

# preferred image type first; start.gg labels them e.g. "stockIcon" / "icon"
PREFERRED = ["stockIcon", "icon"]


def main():
    data = StartGG().query(QUERY, {"id": ULTIMATE_ID})
    rows = []
    for c in data["videogame"]["characters"]:
        imgs = c.get("images") or []
        by_type = {i.get("type"): i.get("url") for i in imgs if i.get("url")}
        pick = next((t for t in PREFERRED if t in by_type), next(iter(by_type), None))
        if pick:
            rows.append({"char": c["name"], "url": by_type[pick], "type": pick})
    df = pd.DataFrame(rows).sort_values("char")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    types = sorted({i.get("type") for c in data["videogame"]["characters"] for i in (c.get("images") or [])})
    print(f"Wrote {len(df)} character icons to {OUT} (image types available: {types})")


if __name__ == "__main__":
    main()
