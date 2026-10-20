"""
03_pull_sets.py

For every event from previous stage, fetch all of its sets
with games, characters, stages, and players. Save raw API response.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import math
import os
import time

import pandas as pd
from tqdm import tqdm

import data_pull
from data_pull import StartGG, iter_event_sets

ROOT       = Path(__file__).resolve().parents[1]
EVENTS_CSV = ROOT / "data" / "interim" / "events.csv"
SETS_DIR   = ROOT / "data" / "raw" / "sets"
LOG_CSV    = ROOT / "data" / "interim" / "pull_log.csv"
PER_PAGE   = 25
MIN_PER_PAGE = 5

LOG_FIELDS = ["logged_at", "event_id", "status", "n_sets", "seconds", "per_page", "error"]


def load_events(path: Path) -> pd.DataFrame:
    """Kept events, largest first (majors = most sets and best character reporting)."""
    df = pd.read_csv(path)
    return df.sort_values("num_entrants", ascending=False).reset_index(drop=True)


def done_event_ids() -> set[int]:
    """One directory listing instead of one disk check per event."""
    ids = set()
    for name in os.listdir(SETS_DIR):
        if name.endswith(".json"):
            try:
                ids.add(int(name[:-5]))
            except ValueError:
                pass
    return ids


def estimate_event_requests(num_entrants) -> int:
    """~2 sets per entrant (double elim), PER_PAGE sets per request."""
    n = 0 if pd.isna(num_entrants) else int(num_entrants)
    return max(1, math.ceil(2 * n / PER_PAGE))


def estimate_requests(df: pd.DataFrame) -> int:
    return int(sum(estimate_event_requests(n) for n in df["num_entrants"]))


def _is_complexity_error(err: Exception) -> bool:
    return "complex" in str(err).lower()


def fetch_event(client: StartGG, event_id, per_page: int,
                on_page=None, on_restart=None) -> tuple[list[dict], int]:
    """All sets for one event. Halves per_page and restarts on complexity errors.
    Returns (sets, per_page_used). On failure, the exception carries .per_page."""
    while True:
        try:
            sets = list(iter_event_sets(client, int(event_id), per_page=per_page, on_page=on_page))
            return sets, per_page
        except RuntimeError as e:
            if _is_complexity_error(e) and per_page > MIN_PER_PAGE:
                per_page = max(MIN_PER_PAGE, per_page // 2)
                tqdm.write(f"  complexity limit -> retrying event {event_id} at per_page={per_page}")
                if on_restart:
                    on_restart()
                continue
            e.per_page = per_page
            raise


def save_event(event_id, meta: dict, sets: list[dict]) -> None:
    """Atomic write: .tmp then rename, so a crash never leaves a half-written .json."""
    payload = {**meta, "event_id": int(event_id), "n_sets": len(sets), "sets": sets}
    final = SETS_DIR / f"{int(event_id)}.json"
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    for attempt in range(5):            # OneDrive/antivirus can briefly lock files
        try:
            os.replace(tmp, final)
            return
        except PermissionError:
            time.sleep(1 + attempt)
    os.replace(tmp, final)


def log_result(event_id, status: str, n_sets: int, seconds: float,
               per_page: int, error: str | None = None) -> None:
    new_file = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event_id": int(event_id), "status": status, "n_sets": n_sets,
            "seconds": round(seconds, 1), "per_page": per_page,
            "error": (error or "")[:500],
        })


def _fmt_dur(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main(limit: int | None = None, shard: int = 0, n_shards: int = 1,
         interval: float | None = None) -> None:
    global LOG_CSV
    tag = f" [shard {shard}/{n_shards}]" if n_shards > 1 else ""
    print(f"Starting step 03{tag}...", flush=True)
    if interval:
        data_pull.MIN_INTERVAL = interval      # seconds between this worker's requests
    if n_shards > 1:
        LOG_CSV = LOG_CSV.with_name(f"pull_log_shard{shard}.csv")
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    t = time.time()
    print(f"Loading {EVENTS_CSV.name}...", end=" ", flush=True)
    events = load_events(EVENTS_CSV)
    print(f"{len(events):,} events ({time.time() - t:.1f}s)", flush=True)

    t = time.time()
    print(f"Checking {SETS_DIR.name}/ for finished events...", end=" ", flush=True)
    done = done_event_ids()
    print(f"{len(done):,} found ({time.time() - t:.1f}s)", flush=True)

    is_done = events["event_id"].isin(done)
    n_done = int(is_done.sum())
    todo = events[~is_done]
    if n_shards > 1:
        # split by event ID (stable across restarts; IDs are unrelated to event size,
        # so both workers still get a mix of big and small events)
        todo = todo[todo["event_id"] % n_shards == shard]
    if limit:
        todo = todo.head(limit)

    # clean up leftovers from an interrupted run (only this worker's events,
    # so we never delete a file the other worker is writing right now)
    mine = set(int(x) for x in todo["event_id"])
    for name in os.listdir(SETS_DIR):
        if name.endswith(".json.tmp"):
            try:
                if int(name[:-9]) in mine:
                    os.remove(SETS_DIR / name)
            except ValueError:
                pass

    n_req = estimate_requests(todo)
    print(f"\n{len(events):,} kept events, {n_done:,} already done")
    secs_per_req = max(data_pull.MIN_INTERVAL, 1.5)   # observed: ~1.5 s/request incl. response time
    print(f"{len(todo):,} to pull{tag}, ~{n_req:,} requests, ~{_fmt_dur(n_req * secs_per_req)}\n")
    if todo.empty:
        return

    client = StartGG(log=tqdm.write)
    counts = {"ok": 0, "empty": 0, "error": 0}

    # Overall bar counts requests (not events), so the ETA reflects event size.
    # Its total starts as an estimate and is corrected after each event.
    req_bar = tqdm(total=n_req, desc=f"Overall{tag}", unit="req", position=0, dynamic_ncols=True)

    try:
        for i, ev in enumerate(todo.itertuples(index=False), start=1):
            t0 = time.time()
            est = estimate_event_requests(ev.num_entrants)
            pages_used = 0
            ev_bar = tqdm(desc=f"  event {ev.event_id} ({int(ev.num_entrants):,} entrants)",
                          unit="page", position=1, leave=False, dynamic_ncols=True)

            def on_page(page, total_pages):
                nonlocal pages_used
                pages_used += 1
                ev_bar.total = total_pages
                ev_bar.update(1)
                req_bar.update(1)

            try:
                sets, per_page = fetch_event(client, ev.event_id, PER_PAGE,
                                             on_page=on_page, on_restart=ev_bar.reset)
                meta = {
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "per_page": per_page,
                    "tournament_id": int(ev.tournament_id),
                    "num_entrants": int(ev.num_entrants),
                }
                save_event(ev.event_id, meta, sets)
                status = "ok" if sets else "empty"
                log_result(ev.event_id, status, len(sets), time.time() - t0, per_page)
                counts[status] += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:  # one bad event shouldn't kill a long run
                log_result(ev.event_id, "error", 0, time.time() - t0,
                           getattr(e, "per_page", PER_PAGE), repr(e))
                counts["error"] += 1
                tqdm.write(f"  ✗ event {ev.event_id}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                ev_bar.close()

            # replace this event's estimate with the pages it actually took
            req_bar.total = max(req_bar.n, req_bar.total + pages_used - est)
            req_bar.set_postfix(events=f"{i:,}/{len(todo):,}", ok=counts["ok"],
                                empty=counts["empty"], err=counts["error"], refresh=False)
            req_bar.refresh()
    except KeyboardInterrupt:
        tqdm.write("\nStopped by user. The current event was not saved; re-run to resume.")
    finally:
        req_bar.close()

    print(f"\nDone: {counts}  ({client.n_requests:,} requests, {client.n_retries:,} retries)")
    if counts["error"]:
        print(f"Errors logged in {LOG_CSV}; re-run this script to retry them.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only pull the N largest remaining events")
    p.add_argument("--shard", default="0/1", help="worker i of n, e.g. 0/2 and 1/2")
    p.add_argument("--interval", type=float, default=None,
                   help="min seconds between requests for this worker (use 1.6 with 2 workers)")
    a = p.parse_args()
    i, n = (int(x) for x in a.shard.split("/"))
    if not 0 <= i < n:
        p.error("--shard must look like 0/2 or 1/2")
    main(a.limit, shard=i, n_shards=n, interval=a.interval)