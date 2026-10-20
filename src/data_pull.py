# Claude generated client for pulling Smash data

"""
Minimal start.gg GraphQL client for the Smash Ultimate project.
 
Token: put it in secrets/startgg_token.txt (gitignored) or set STARTGG_TOKEN.
Docs / explorer: https://developer.start.gg/docs/intro
"""
 
import os
import time
from pathlib import Path

import requests

API_URL = "https://api.start.gg/gql/alpha"
ULTIMATE_ID = 1386          # start.gg videogame id for Smash Ultimate
MIN_INTERVAL = 0.8          # seconds between calls (~75/min, under the 80/min cap)
NETWORK_WAIT = 30           # seconds between checks while the connection is down


def load_token() -> str:
    if os.getenv("STARTGG_TOKEN"):
        return os.environ["STARTGG_TOKEN"].strip()
    path = Path(__file__).resolve().parents[1] / "secrets" / "key.txt"
    return path.read_text().strip()


class StartGG:
    def __init__(self, token: str | None = None, log=print):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token or load_token()}"
        self._last_call = 0.0
        self.log = log          # scripts pass tqdm.write so messages don't break progress bars
        self.n_requests = 0
        self.n_retries = 0
        self._down_since = None
        self._last_down_msg = 0.0

    def _backoff(self, attempt: int, retries: int, reason: str) -> None:
        delay = 2 ** attempt * 5
        self.n_retries += 1
        self.log(f"  ⏸ {reason}; waiting {delay}s before retry {attempt + 1}/{retries - 1}")
        time.sleep(delay)

    def _network_down(self, err: Exception) -> None:
        """Wait out a lost connection. Does not count as a failed attempt."""
        now = time.time()
        if self._down_since is None:
            self._down_since = now
            self._last_down_msg = now
            self.log(f"  ⚠ network down ({type(err).__name__}); waiting for it to come back...")
        elif now - self._last_down_msg >= 600:
            self._last_down_msg = now
            self.log(f"  ⚠ still waiting for network ({(now - self._down_since) / 60:.0f} min)")
        time.sleep(NETWORK_WAIT)

    def _network_ok(self) -> None:
        if self._down_since is not None:
            self.log(f"  ✓ network back after {(time.time() - self._down_since) / 60:.1f} min; continuing")
            self._down_since = None

    def query(self, gql: str, variables: dict | None = None, retries: int = 5) -> dict:
        """POST one query; throttle, retry on rate limits / server errors,
        and wait indefinitely (without using up retries) while the network is down."""
        attempt = 0
        while attempt < retries:
            wait = MIN_INTERVAL - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            self.n_requests += 1

            try:
                resp = self.session.post(API_URL, json={"query": gql, "variables": variables or {}}, timeout=60)
            except (requests.ConnectionError, requests.Timeout) as e:
                self._network_down(e)
                continue
            self._network_ok()

            if resp.status_code == 429 or resp.status_code >= 500:
                self._backoff(attempt, retries, f"HTTP {resp.status_code}")
                attempt += 1
                continue
            resp.raise_for_status()
            body = resp.json()

            errors = body.get("errors") or []
            if any("rate limit" in str(e.get("message", "")).lower() for e in errors):
                self._backoff(attempt, retries, "rate limited")
                attempt += 1
                continue
            if errors:
                # e.g. complexity limit -> lower perPage in the caller
                raise RuntimeError(errors)
            return body["data"]
        raise RuntimeError(f"Gave up after {retries} attempts")


# ── Queries ─────────────────────────────────────────────────────────────

US_TOURNAMENTS = """
query USUltimateTournaments($page: Int!, $perPage: Int!, $after: Timestamp, $before: Timestamp) {
  tournaments(query: {
    page: $page, perPage: $perPage, sortBy: "startAt asc",
    filter: { countryCode: "US", videogameIds: [1386], afterDate: $after, beforeDate: $before }
  }) {
    pageInfo { total totalPages }
    nodes {
      id name slug startAt city addrState
      events(filter: { videogameId: [1386] }) {
        id name isOnline numEntrants
        teamRosterSize { maxPlayers }
      }
    }
  }
}
"""

EVENT_SETS = """
query EventSets($eventId: ID!, $page: Int!, $perPage: Int!) {
  event(id: $eventId) {
    sets(page: $page, perPage: $perPage, sortType: STANDARD) {
      pageInfo { total totalPages }
      nodes {
        id fullRoundText winnerId displayScore totalGames
        slots { entrant { id participants { player { id gamerTag } } } }
        games {
          orderNum winnerId entrant1Score entrant2Score
          stage { name }
          selections { entrant { id } character { name } }
        }
      }
    }
  }
}
"""


def count_us_tournaments(client: StartGG, after: int, before: int) -> int:
    """Cheap count (one request) for a date range given as Unix timestamps."""
    data = client.query(US_TOURNAMENTS, {"page": 1, "perPage": 1, "after": after, "before": before})
    return data["tournaments"]["pageInfo"]["total"]


def iter_us_tournaments(client: StartGG, after: int, before: int, per_page: int = 50, on_page=None):
    """on_page(page, total_pages) is called after each page, for progress bars."""
    page = 1
    while True:
        data = client.query(US_TOURNAMENTS, {"page": page, "perPage": per_page, "after": after, "before": before})
        t = data["tournaments"]
        total_pages = t["pageInfo"]["totalPages"] or 0
        yield from t["nodes"]
        if on_page:
            on_page(page, total_pages)
        if page >= total_pages:
            break
        page += 1


def iter_event_sets(client: StartGG, event_id: int, per_page: int = 25, on_page=None):
    """on_page(page, total_pages) is called after each page, for progress bars."""
    page = 1
    while True:
        data = client.query(EVENT_SETS, {"eventId": event_id, "page": page, "perPage": per_page})
        sets = data["event"]["sets"]
        total_pages = sets["pageInfo"]["totalPages"] or 0
        yield from sets["nodes"]
        if on_page:
            on_page(page, total_pages)
        if page >= total_pages:
            break
        page += 1


def is_target_event(ev: dict, min_entrants: int = 32) -> bool:
    """Ultimate singles, offline, big enough."""
    roster = (ev.get("teamRosterSize") or {}).get("maxPlayers")
    return (not ev["isOnline"]) and roster in (None, 1) and (ev["numEntrants"] or 0) >= min_entrants

# Uncomment for test
# if __name__ == "__main__":
#     from datetime import datetime, timezone
#
#     gg = StartGG()
#     for year in range(2019, 2027):
#         a = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
#         b = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp())
#         print(year, count_us_tournaments(gg, a, b))