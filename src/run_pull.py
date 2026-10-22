# Claude generated script that runs process to monitor and upkeep data pulling

"""
run_pull.py  —  supervisor for step 03

Starts the step-03 workers (each in its own window on Windows), keeps the
laptop awake, and restarts any worker that crashes, is closed, or stops
making progress. Exits when every event has been pulled.

Run from the project root:
    .\\.venv\\Scripts\\python.exe .\\src\\run_pull.py

Stop with Ctrl+C in this window (workers are stopped too; re-run to resume).
"""

from pathlib import Path
import os
import subprocess
import sys
import time

import pandas as pd

ROOT       = Path(__file__).resolve().parents[1]
EVENTS_CSV = ROOT / "data" / "interim" / "events.csv"
SETS_DIR   = ROOT / "data" / "raw" / "sets"
WORKER     = ROOT / "src" / "03_pull_sets.py"

N_WORKERS     = 1
INTERVAL      = 0.8     # seconds between requests (one worker measured ~1.5 s/request, no rate limits)
CHECK_EVERY   = 15      # seconds between health checks
STATUS_EVERY  = 300     # seconds between status lines
STALL_MINUTES = 30      # no new finished event for this long -> restart that worker
MAX_IDLE_RUNS = 3       # worker exits this many times in a row without progress -> give up on it

IS_WINDOWS = os.name == "nt"


# ── keep the computer awake ────────────────────────────────────────────

def keep_awake(on: bool) -> None:
    """Windows only: stop the PC from sleeping while this script runs.
    (Closing the lid can still sleep it unless 'lid close = Do nothing'.)"""
    if not IS_WINDOWS:
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


# ── progress bookkeeping ───────────────────────────────────────────────

def shard_ids(events: pd.DataFrame, shard: int) -> set[int]:
    return set(int(x) for x in events["event_id"] if int(x) % N_WORKERS == shard)


def done_ids() -> set[int]:
    ids = set()
    for name in os.listdir(SETS_DIR):
        if name.endswith(".json"):
            try:
                ids.add(int(name[:-5]))
            except ValueError:
                pass
    return ids


# ── worker processes ───────────────────────────────────────────────────

def start_worker(shard: int) -> subprocess.Popen:
    # same launch method that works reliably on this machine (runpy via -c)
    code = (
        "import sys, runpy; "
        f"sys.path.insert(0, r'{ROOT / 'src'}'); "
        f"sys.argv = ['03_pull_sets.py', '--shard', '{shard}/{N_WORKERS}', '--interval', '{INTERVAL}']; "
        f"runpy.run_path(r'{WORKER}', run_name='__main__')"
    )
    kwargs = {"cwd": ROOT}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE   # own window with its progress bars
    return subprocess.Popen([sys.executable, "-c", code], **kwargs)


def stop_worker(p: subprocess.Popen | None) -> None:
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


def log(msg: str) -> None:
    print(f"[{time.strftime('%a %H:%M:%S')}] {msg}", flush=True)


# ── main loop ──────────────────────────────────────────────────────────

def main() -> None:
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS_CSV)
    mine = {s: shard_ids(events, s) for s in range(N_WORKERS)}
    total = len(events)

    procs: dict[int, subprocess.Popen | None] = {s: None for s in range(N_WORKERS)}
    done_at_launch = {s: 0 for s in range(N_WORKERS)}
    last_progress = {s: time.time() for s in range(N_WORKERS)}
    last_count = {s: -1 for s in range(N_WORKERS)}
    idle_runs = {s: 0 for s in range(N_WORKERS)}
    restarts = {s: 0 for s in range(N_WORKERS)}
    finished: set[int] = set()
    gave_up: set[int] = set()

    keep_awake(True)
    log(f"Supervising {N_WORKERS} workers over {total:,} events. Ctrl+C here stops everything.")
    last_status = 0.0

    try:
        while True:
            done = done_ids()

            for s in range(N_WORKERS):
                if s in finished or s in gave_up:
                    continue
                n_done = len(mine[s] & done)
                remaining = len(mine[s]) - n_done

                if n_done != last_count[s]:
                    last_count[s] = n_done
                    last_progress[s] = time.time()

                p = procs[s]
                running = p is not None and p.poll() is None

                if remaining == 0:
                    stop_worker(p)
                    finished.add(s)
                    log(f"worker {s}: all {len(mine[s]):,} events done")
                    continue

                if running:
                    stalled = (time.time() - last_progress[s]) / 60
                    if stalled >= STALL_MINUTES:
                        log(f"worker {s}: no progress for {stalled:.0f} min -> restarting")
                        stop_worker(p)
                        procs[s] = start_worker(s)
                        restarts[s] += 1
                        last_progress[s] = time.time()
                    continue

                # not running: first launch, crash, closed window, or finished a pass with errors left
                if p is not None:
                    progressed = n_done > done_at_launch[s]
                    idle_runs[s] = 0 if progressed else idle_runs[s] + 1
                    log(f"worker {s}: exited (code {p.returncode}), {remaining:,} events left"
                        + ("" if progressed else f", no progress ({idle_runs[s]}/{MAX_IDLE_RUNS})"))
                    if idle_runs[s] >= MAX_IDLE_RUNS:
                        gave_up.add(s)
                        log_name = f"pull_log_shard{s}.csv" if N_WORKERS > 1 else "pull_log.csv"
                        log(f"worker {s}: giving up; {remaining:,} events keep failing "
                            f"(see data/interim/{log_name})")
                        continue
                    restarts[s] += 1
                    time.sleep(5)

                done_at_launch[s] = n_done
                last_progress[s] = time.time()
                procs[s] = start_worker(s)
                log(f"worker {s}: started ({remaining:,} events left)")

            if len(finished | gave_up) == N_WORKERS:
                break

            if time.time() - last_status >= STATUS_EVERY:
                last_status = time.time()
                n_all = len(set(events["event_id"].astype(int)) & done)
                alive = ", ".join(f"w{s}:{'up' if procs[s] and procs[s].poll() is None else 'down'}"
                                  f"(restarts {restarts[s]})" for s in range(N_WORKERS))
                log(f"progress {n_all:,}/{total:,} events ({n_all / max(total, 1):.1%}) | {alive}")

            time.sleep(CHECK_EVERY)

    except KeyboardInterrupt:
        log("Ctrl+C: stopping workers (the event each was on is not saved; re-run to resume)")
    finally:
        for p in procs.values():
            stop_worker(p)
        keep_awake(False)

    n_all = len(set(events["event_id"].astype(int)) & done_ids())
    log(f"Finished: {n_all:,}/{total:,} events saved.")
    if gave_up:
        log("Some events kept failing; check the pull log. Re-running this script retries them.")


if __name__ == "__main__":
    main()