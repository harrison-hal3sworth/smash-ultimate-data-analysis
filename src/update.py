"""
update.py  —  refresh the data and models in one command (built to run weekly, unattended).

What one run does:
  1. Re-pull the tournament list for every month since the data was last brought up to date
     (the last successful update, or step 03's pull log before the first one), so a gap of
     weeks or months is filled, not just the last week. Never less than the last 2 months,
     since organizers add or fix events late                                   -> step 01
  2. Rebuild the event list                                                     -> step 02
  3. Pull sets for every new event, and RE-pull recent events that were fetched while still
     in progress (fetched less than a week after they started) or came back empty -> step 03
  4. Rebuild the analysis tables                                                -> step 04
  5. Back up data/models, rebuild the models                                    -> step 05
  6. Run the data checks (pytest -m data). If anything fails, the old models are restored,
     so the dashboard never ends up on a broken build.
  7. Log a before/after summary to logs/ and a row to reports/update_history.csv.
     Optionally re-run the validation report (--validate) and commit + push the new models
     (--commit) so anyone running the dashboard just needs `git pull`.

Run from the project root:
    python src/update.py --dry-run      # show what would be refreshed, change nothing
    python src/update.py                # full update (about 20-40 min for a week of new data)
    python src/update.py --commit       # ... and commit + push data/models
    python src/update.py --skip-pull    # rebuild tables + models from data already on disk
    python src/update.py --since 2026-06  # force a re-pull of tournaments from June 2026 on

Schedule it weekly with scripts/update_weekly.bat (see README).
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOURN_DIR = ROOT / "data" / "raw" / "tournaments"
SETS_DIR = ROOT / "data" / "raw" / "sets"
STALE_DIR = ROOT / "data" / "raw" / "sets_stale"        # recent files set aside for a re-pull
EVENTS_CSV = ROOT / "data" / "interim" / "events.csv"
GAMES_CSV = ROOT / "data" / "processed" / "games.csv"
MODELS = ROOT / "data" / "models"
BACKUP = ROOT / "data" / "models_backup"
LOG_DIR = ROOT / "logs"
HISTORY = ROOT / "reports" / "update_history.csv"
PULL_LOG = ROOT / "data" / "interim" / "pull_log.csv"   # written by step 03
LOCK = ROOT / "data" / ".update.lock"

REFRESH_MONTHS = 2         # re-pull the tournament list for this month and the previous one(s)
REFRESH_DAYS = 60          # look for in-progress / empty event files from at least the last N days
SETTLE_DAYS = 7            # an event fetched less than this long after it started may be incomplete
LOCK_HOURS = 12            # a lock older than this is from a crashed run and is ignored

_log_file = None


def log(msg: str = "") -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}" if msg else ""
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


# ── pure helpers (unit-tested) ─────────────────────────────────────────

def months_to_refresh(today: date, existing: list[str], n_months: int = REFRESH_MONTHS,
                      last_pull: date | None = None) -> list[str]:
    """Month files ('YYYY-MM') to delete and re-pull.

    Covers every month since the last pull, however long ago that was: the month before the
    last pull (for events added late), the month of the last pull (it was still in progress),
    and every month up to today. Never fewer than the last `n_months`. Saved months after
    today (pulled early by an older version of step 01) are re-pulled too."""
    start_y, start_m = today.year, today.month
    for _ in range(n_months - 1):
        start_y, start_m = (start_y, start_m - 1) if start_m > 1 else (start_y - 1, 12)
    if last_pull is not None:
        y, m = (last_pull.year, last_pull.month - 1) if last_pull.month > 1 else (last_pull.year - 1, 12)
        start_y, start_m = min((start_y, start_m), (y, m))
    months = []
    y, m = start_y, start_m
    while (y, m) <= (today.year, today.month):
        months.append(f"{y}-{m:02d}")
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    current = f"{today.year}-{today.month:02d}"
    future = [e for e in existing if e > current]
    return sorted(set(months) | set(future))


def last_pull_from(history_rows: list[dict], pull_log_times: list[str]) -> date | None:
    """When was the data last brought up to date? The last successful update run, or, before
    the first update, the latest entry in step 03's pull log."""
    ok = [r["run_at"] for r in history_rows if r.get("status") == "ok" and r.get("run_at")]
    times = ok or [t for t in pull_log_times if t]
    if not times:
        return None
    return max(datetime.fromisoformat(t).date() for t in times)


def is_stale(fetched_at: str | None, event_date: date, n_sets: int, settle_days: int = SETTLE_DAYS) -> bool:
    """Should this event's set file be re-pulled? Yes if it came back empty, or if it was
    fetched less than `settle_days` after the event started (it may have been in progress)."""
    if n_sets == 0:
        return True
    if not fetched_at:
        return True
    fetched = datetime.fromisoformat(fetched_at).date()
    return fetched < event_date + timedelta(days=settle_days)


def lock_is_active(lock_path: Path, now: float | None = None, max_hours: float = LOCK_HOURS) -> bool:
    if not lock_path.exists():
        return False
    age_h = ((now or time.time()) - lock_path.stat().st_mtime) / 3600
    return age_h < max_hours


# ── steps ──────────────────────────────────────────────────────────────

def run_step(name: str, script: str, *args: str) -> None:
    """Run one pipeline script, streaming its output to the console and the log."""
    log(f"── {name}: python src/{script} {' '.join(args)}".rstrip())
    t0 = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([sys.executable, "-u", str(SRC / script), *args], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        line = line.rstrip("\n")
        if "\r" in line:                       # progress bars: keep only the final state
            line = line.split("\r")[-1]
        print("   " + line, flush=True)
        if _log_file:
            _log_file.write("   " + line + "\n")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed (exit code {proc.returncode})")
    log(f"   {name} done in {(time.time() - t0) / 60:.1f} min")


def stats() -> dict:
    """Headline counts for the before/after summary."""
    out = {"tournaments": 0, "events": 0, "event_files": 0, "games": 0, "top10": []}
    for f in TOURN_DIR.glob("*.json"):
        try:
            out["tournaments"] += len(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    if EVENTS_CSV.exists():
        out["events"] = sum(1 for _ in EVENTS_CSV.open(encoding="utf-8")) - 1
    if SETS_DIR.exists():
        out["event_files"] = sum(1 for n in os.listdir(SETS_DIR) if n.endswith(".json"))
    if GAMES_CSV.exists():
        with GAMES_CSV.open("rb") as fh:
            out["games"] = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b"")) - 1
    if (MODELS / "players.csv").exists():
        p = pd.read_csv(MODELS / "players.csv", usecols=["tag", "rank"], low_memory=False)
        out["top10"] = list(p.dropna(subset=["rank"]).sort_values("rank").tag.astype(str).head(10))
    return out


def last_pull() -> date | None:
    rows, times = [], []
    if HISTORY.exists():
        with HISTORY.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    if PULL_LOG.exists():
        times = pd.read_csv(PULL_LOG, usecols=["logged_at"]).logged_at.dropna().astype(str).tolist()
    return last_pull_from(rows, times)


def refresh_tournament_months(dry_run: bool, since: date | None) -> list[str]:
    existing = sorted(f.stem for f in TOURN_DIR.glob("*.json"))
    months = [m for m in months_to_refresh(date.today(), existing, last_pull=since)
              if m in existing or m <= date.today().strftime("%Y-%m")]
    for mth in months:
        f = TOURN_DIR / f"{mth}.json"
        if f.exists() and not dry_run:
            f.unlink()
    return months


def find_stale_events(dry_run: bool, since: date | None) -> list[int]:
    """Recent event files that may be incomplete; moved to sets_stale/ so step 03 re-pulls them."""
    if not EVENTS_CSV.exists():
        return []
    ev = pd.read_csv(EVENTS_CSV, usecols=["event_id", "date"])
    ev["date"] = pd.to_datetime(ev["date"]).dt.date
    days = REFRESH_DAYS
    if since is not None:                       # reach back past the last pull, however old
        days = max(days, (date.today() - since).days + 2 * SETTLE_DAYS)
    recent = ev[ev.date >= date.today() - timedelta(days=days)]
    stale = []
    for eid, d in zip(recent.event_id, recent.date):
        f = SETS_DIR / f"{int(eid)}.json"
        if not f.exists():
            continue                                    # new event: step 03 pulls it anyway
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stale.append(int(eid))
            continue
        if is_stale(payload.get("fetched_at"), d, int(payload.get("n_sets", len(payload.get("sets") or [])))):
            stale.append(int(eid))
    if stale and not dry_run:
        STALE_DIR.mkdir(parents=True, exist_ok=True)
        for eid in stale:
            os.replace(SETS_DIR / f"{eid}.json", STALE_DIR / f"{eid}.json")
    return stale


def restore_stale() -> None:
    """If the re-pull didn't produce a new file, put the old one back."""
    if not STALE_DIR.exists():
        return
    for f in STALE_DIR.glob("*.json"):
        target = SETS_DIR / f.name
        if target.exists():
            f.unlink()
        else:
            os.replace(f, target)


def backup_models() -> None:
    if BACKUP.exists():
        shutil.rmtree(BACKUP)
    shutil.copytree(MODELS, BACKUP)


def restore_models() -> None:
    if BACKUP.exists():
        shutil.rmtree(MODELS, ignore_errors=True)
        shutil.copytree(BACKUP, MODELS)
        log("   restored data/models from the backup: the dashboard keeps the previous build")


def run_data_checks() -> bool:
    log("── data checks: pytest -m data")
    r = subprocess.run([sys.executable, "-m", "pytest", "-m", "data", "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (r.stdout + r.stderr).strip().splitlines()[-15:]:
        log("   " + line)
    return r.returncode == 0


def git_commit(summary: str) -> None:
    log("── committing data/models")
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    git("add", "data/models", "reports/update_history.csv")
    if git("diff", "--cached", "--quiet").returncode == 0:
        log("   nothing changed in data/models; no commit")
        return
    c = git("commit", "-m", f"Data update {date.today():%Y-%m-%d}: {summary}")
    log("   " + (c.stdout or c.stderr).strip().splitlines()[0])
    p = git("push")
    log("   pushed" if p.returncode == 0 else f"   push failed: {(p.stderr or '').strip()[:300]}")


def append_history(row: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    new = not HISTORY.exists()
    with HISTORY.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ── main ───────────────────────────────────────────────────────────────

def main(args) -> int:
    global _log_file
    LOG_DIR.mkdir(exist_ok=True)
    _log_file = (LOG_DIR / f"update_{datetime.now():%Y-%m-%d_%H%M}.log").open("w", encoding="utf-8")

    if lock_is_active(LOCK):
        log(f"Another update started less than {LOCK_HOURS} h ago ({LOCK}). Exiting. "
            "Delete the lock file if that run crashed.")
        return 2
    if not args.dry_run:
        LOCK.write_text(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}")

    t0 = time.time()
    before = stats()
    log(f"Before: {before['tournaments']:,} tournaments, {before['events']:,} events, "
        f"{before['event_files']:,} event files, {before['games']:,} games")
    status, months, stale = "ok", [], []
    try:
        if not args.skip_pull:
            since = date.fromisoformat(args.since + "-01") if args.since else last_pull()
            log(f"Data last brought up to date: {since or 'unknown'}"
                + (" (from --since)" if args.since else ""))
            months = refresh_tournament_months(args.dry_run, since)
            log(f"Tournament months to re-pull: {', '.join(months) or 'none'}")
            if not args.dry_run:
                run_step("1/5 tournaments", "01_pull_tournaments.py")
                run_step("2/5 events", "02_filter_events.py")
            stale = find_stale_events(args.dry_run, since)
            log(f"Recent event files to re-pull (empty, or fetched while in progress): {len(stale)}")
            if args.dry_run:
                new = 0
                if EVENTS_CSV.exists():
                    ids = pd.read_csv(EVENTS_CSV, usecols=["event_id"]).event_id
                    new = sum(not (SETS_DIR / f"{int(e)}.json").exists() for e in ids)
                log(f"Events in events.csv without set data yet: {new:,} "
                    "(more appear after the tournament list is refreshed)")
                log("Dry run: nothing changed.")
                return 0
            try:
                run_step("3/5 sets", "03_pull_sets.py")
            finally:
                restore_stale()
        elif args.dry_run:
            log("Dry run with --skip-pull: would rebuild tables and models only.")
            return 0

        run_step("4/5 tables", "04_build_dataset.py")
        backup_models()
        try:
            run_step("5/5 models", "05_build_models.py")
        except RuntimeError:
            restore_models()
            raise
        if not run_data_checks():
            restore_models()
            raise RuntimeError("data checks failed")
        if args.validate:
            run_step("validation report", "07_validate.py")
    except Exception as e:                               # log, keep the old models, exit non-zero
        status = f"failed: {e}"
        log(f"UPDATE FAILED: {e}")
    finally:
        if not args.dry_run:
            LOCK.unlink(missing_ok=True)

    after = stats()
    entered = [t for t in after["top10"] if t not in before["top10"]]
    summary = (f"+{after['events'] - before['events']:,} events, "
               f"+{after['games'] - before['games']:,} games")
    log("")
    log(f"After:  {after['tournaments']:,} tournaments, {after['events']:,} events, "
        f"{after['event_files']:,} event files, {after['games']:,} games")
    log(f"Change: {summary}; {len(stale)} recent events re-pulled")
    log(f"Top 10: {', '.join(after['top10'])}" + (f"  (new: {', '.join(entered)})" if entered else ""))
    log(f"Status: {status}. Took {(time.time() - t0) / 60:.1f} min.")

    append_history({
        "run_at": datetime.now().isoformat(timespec="minutes"), "status": status,
        "months_refreshed": " ".join(months), "stale_repulled": len(stale),
        "events_before": before["events"], "events_after": after["events"],
        "games_before": before["games"], "games_after": after["games"],
        "top10": "; ".join(after["top10"]),
    })
    if status == "ok" and args.commit:
        git_commit(summary)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Refresh start.gg data and rebuild the models.")
    p.add_argument("--dry-run", action="store_true", help="show what would be refreshed; change nothing")
    p.add_argument("--since", metavar="YYYY-MM", help="re-pull tournaments from this month on "
                   "(default: worked out from the last update or pull)")
    p.add_argument("--skip-pull", action="store_true", help="no API calls; rebuild tables + models from disk")
    p.add_argument("--validate", action="store_true", help="also re-run the validation report (07)")
    p.add_argument("--commit", action="store_true", help="commit + push data/models when the update succeeds")
    sys.exit(main(p.parse_args()))
