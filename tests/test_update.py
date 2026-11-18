"""Unit tests for the pure helpers in src/update.py (no data or API needed)."""

from datetime import date
from pathlib import Path
import importlib.util
import os
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("update", ROOT / "src" / "update.py")
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)


# ── months_to_refresh ──────────────────────────────────────────────────

def test_refresh_current_and_previous_month():
    assert update.months_to_refresh(date(2026, 11, 18), ["2026-09", "2026-10", "2026-11"], 2) == ["2026-10", "2026-11"]


def test_refresh_wraps_year_boundary():
    assert update.months_to_refresh(date(2027, 1, 5), [], 2) == ["2026-12", "2027-01"]


def test_refresh_includes_future_months_pulled_early():
    out = update.months_to_refresh(date(2026, 11, 18), ["2026-10", "2026-11", "2026-12"], 2)
    assert out == ["2026-10", "2026-11", "2026-12"]


def test_refresh_more_months():
    assert update.months_to_refresh(date(2026, 3, 1), [], 4) == ["2025-12", "2026-01", "2026-02", "2026-03"]


# ── is_stale ───────────────────────────────────────────────────────────

def test_empty_event_is_stale():
    assert update.is_stale("2026-11-01T00:00:00+00:00", date(2026, 10, 1), 0)


def test_fetched_during_event_is_stale():
    assert update.is_stale("2026-11-02T10:00:00+00:00", date(2026, 11, 1), 120)


def test_fetched_after_settling_is_not_stale():
    assert not update.is_stale("2026-11-09T10:00:00+00:00", date(2026, 11, 1), 120)


def test_settle_boundary():
    # exactly settle_days after the start counts as settled
    assert not update.is_stale("2026-11-08T00:00:00+00:00", date(2026, 11, 1), 50, settle_days=7)
    assert update.is_stale("2026-11-07T23:59:00+00:00", date(2026, 11, 1), 50, settle_days=7)


def test_missing_fetched_at_is_stale():
    assert update.is_stale(None, date(2026, 11, 1), 50)


# ── lock ───────────────────────────────────────────────────────────────

def test_no_lock(tmp_path):
    assert not update.lock_is_active(tmp_path / ".lock")


def test_fresh_lock_is_active(tmp_path):
    lock = tmp_path / ".lock"
    lock.write_text("x")
    assert update.lock_is_active(lock)


def test_old_lock_is_ignored(tmp_path):
    lock = tmp_path / ".lock"
    lock.write_text("x")
    old = time.time() - 13 * 3600
    os.utime(lock, (old, old))
    assert not update.lock_is_active(lock, max_hours=12)


# ── catching up after a long gap ───────────────────────────────────────

def test_catch_up_covers_every_month_since_last_pull():
    # last pulled mid-June, updating in November: May (late additions) through November
    out = update.months_to_refresh(date(2026, 11, 18), [], 2, last_pull=date(2026, 6, 15))
    assert out == ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]


def test_recent_last_pull_keeps_minimum_window():
    out = update.months_to_refresh(date(2026, 11, 18), [], 2, last_pull=date(2026, 11, 11))
    assert out == ["2026-10", "2026-11"]


def test_catch_up_across_year_boundary():
    out = update.months_to_refresh(date(2027, 2, 1), [], 2, last_pull=date(2026, 12, 20))
    assert out == ["2026-11", "2026-12", "2027-01", "2027-02"]


def test_last_pull_prefers_successful_update_runs():
    rows = [{"run_at": "2026-11-02T03:00", "status": "ok"},
            {"run_at": "2026-11-09T03:00", "status": "failed: data checks failed"}]
    assert update.last_pull_from(rows, ["2026-10-21T22:03:00+00:00"]) == date(2026, 11, 2)


def test_last_pull_falls_back_to_pull_log():
    assert update.last_pull_from([], ["2026-10-20T00:36:41+00:00", "2026-10-21T22:03:00+00:00"]) == date(2026, 10, 21)


def test_last_pull_unknown():
    assert update.last_pull_from([], []) is None
