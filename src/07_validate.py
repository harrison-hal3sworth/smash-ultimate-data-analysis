"""
07_validate.py  —  validation and robustness report for the step-05 models.

Writes reports/validation.md (+ reports/figures/calibration.png and reports/tables/*.csv):

  1. Rating calibration      do predicted win chances match how often favourites win?
  2. Where ratings work      accuracy by year, event size and how much history players have
  3. Glicko-2 sensitivity    does the volatility setting (tau) change the results?
  4. Do characters help?     held-out log-loss gain from characters, with a bootstrap CI by event
  5. Character-strength stability
                             same strengths from two random halves of the events? before vs
                             after 2025? under a different penalty?

Run after step 05, from the project root:   python src/07_validate.py
Takes a few minutes on the full data (it refits Glicko-2 and the matchup model several times).
"""

from pathlib import Path
import importlib.util
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import clean  # noqa: E402

spec = importlib.util.spec_from_file_location("build_models", SRC / "05_build_models.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

ROOT = SRC.parent
REPORT = ROOT / "reports"
FIG, TAB = REPORT / "figures", REPORT / "tables"
TEST_FROM = pd.Timestamp(B.TEST_FROM)
N_BOOT = 500
SEED = 0

# thresholds for the pass / check flags in the summary
ECE_OK = 0.02            # mean calibration error (probability points)
STABILITY_OK = 0.80      # Spearman correlation of character strengths between splits


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:{floatfmt}}" if pd.notna(v) else "")
    head = "| " + " | ".join([df.index.name or ""] + list(map(str, df.columns))) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep, *rows])


def metric_rows(df, p_col="p", y_col="y"):
    y, p = df[y_col].to_numpy(), np.clip(df[p_col].to_numpy(), 1e-6, 1 - 1e-6)
    return pd.Series({"sets": len(df), "accuracy": ((p > 0.5) == (y == 1)).mean(),
                      "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()})


# ── 1 + 2: ratings ─────────────────────────────────────────────────────

def rating_checks(rated: pd.DataFrame, sets_c: pd.DataFrame, out: list, flags: list):
    rated = rated.merge(sets_c[["set_id", "num_entrants", "size_bucket"]], on="set_id", how="left")
    rated["p"] = B.expected(rated.p1_r, rated.p1_rd, rated.p2_r, rated.p2_rd)
    rated["y"] = rated.p1_won
    # recalibration: one scale factor k on the logit, p' = sigmoid(k * logit(p)), fitted on the two
    # years BEFORE the test period only (the same idea as the matchup model's learned beta)
    logit = np.log(np.clip(rated.p, 1e-6, 1 - 1e-6) / np.clip(1 - rated.p, 1e-6, 1))
    fit = (rated.date < TEST_FROM) & (rated.date >= TEST_FROM - pd.DateOffset(years=2))
    k = minimize_scalar(lambda k: np.logaddexp(0, k * logit[fit]).sum() - (rated.y[fit] * k * logit[fit]).sum(),
                        bounds=(0.2, 5), method="bounded").x
    rated["p_cal"] = 1 / (1 + np.exp(-k * logit))
    test = rated[rated.date >= TEST_FROM].copy()

    def calib(p):
        fav_p, fav_won = np.maximum(p, 1 - p), np.where(p >= 0.5, test.y, 1 - test.y)
        bins = pd.cut(fav_p, np.arange(0.5, 1.0001, 0.05), include_lowest=True)
        c = pd.DataFrame({"predicted": fav_p, "observed": fav_won}).groupby(bins, observed=True).agg(
            sets=("observed", "size"), predicted=("predicted", "mean"), observed=("observed", "mean"))
        c.index = [f"{iv.left:.2f}-{iv.right:.2f}" for iv in c.index]
        return c, float((c.sets * (c.predicted - c.observed).abs()).sum() / c.sets.sum())

    cal_raw, ece_raw = calib(test.p.to_numpy())
    cal_fix, ece_fix = calib(test.p_cal.to_numpy())
    cal = cal_raw.rename(columns={"predicted": "raw predicted", "observed": "raw observed"}).join(
        cal_fix[["predicted", "observed"]].rename(columns={"predicted": "recal. predicted", "observed": "recal. observed"}),
        how="outer")
    cal.index.name = "favourite's predicted chance"
    cal["sets"] = cal["sets"].fillna(0).astype(int)
    cal.to_csv(TAB / "calibration.csv")
    pd.Series({"k": k, "ece_raw": ece_raw, "ece_recalibrated": ece_fix}).to_csv(TAB / "calibration_factor.csv")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.5, 1], [0.5, 1], ls=":", color="grey", label="perfect calibration")
    ax.plot(cal_raw.predicted, cal_raw.observed, "o-", label=f"raw Glicko-2 (error {ece_raw:.3f})")
    ax.plot(cal_fix.predicted, cal_fix.observed, "s-", label=f"recalibrated, k={k:.2f} (error {ece_fix:.3f})")
    ax.set_xlabel("predicted chance the favourite wins")
    ax.set_ylabel("how often the favourite actually won")
    ax.set_title(f"Rating calibration, held-out sets from {TEST_FROM:%Y-%m-%d}")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "calibration.png", bbox_inches="tight", dpi=130)

    flags.append(("Raw Glicko-2 probabilities calibrated (mean error < %.2f)" % ECE_OK, ece_raw < ECE_OK, f"{ece_raw:.3f}"))
    flags.append(("Recalibrated probabilities calibrated (mean error < %.2f)" % ECE_OK, ece_fix < ECE_OK,
                  f"{ece_fix:.3f} (k = {k:.2f})"))
    direction = "underconfident (favourites win more often than predicted)" if k > 1 else "overconfident"
    out += ["## 1. Rating calibration", "",
            f"Held-out sets (from {TEST_FROM:%Y-%m-%d}) grouped by the favourite's predicted chance. "
            f"Raw Glicko-2 probabilities are **{direction}**: weighted mean error {ece_raw:.3f}. "
            f"Scaling the log-odds by k = {k:.2f}, fitted only on the two years before the test period, "
            f"brings the error to **{ece_fix:.3f}**. The dashboard is unaffected (its matchup model learns its own "
            "scale from the data); the auto-seeder should use the recalibrated probabilities.",
            "", md_table(cal), "", "![calibration](figures/calibration.png)", ""]

    # where the ratings work
    exp = np.minimum(rated.p1_prior_sets, rated.p2_prior_sets)
    rated["experience"] = pd.cut(exp, [-1, 4, 19, 99, np.inf], labels=["0-4", "5-19", "20-99", "100+"])
    by_year = rated.groupby(rated.date.dt.year).apply(metric_rows)
    by_year.index.name = "year"
    test = rated[rated.date >= TEST_FROM]
    by_size = test.groupby("size_bucket", observed=True).apply(metric_rows)
    by_exp = test.groupby("experience", observed=True).apply(metric_rows)
    by_exp.index.name = "prior sets (less experienced player)"
    for name, t in [("by_year", by_year), ("by_event_size", by_size), ("by_experience", by_exp)]:
        t.to_csv(TAB / f"rating_accuracy_{name}.csv")
    for t in (by_year, by_size, by_exp):
        t["sets"] = t["sets"].astype(int)
    out += ["## 2. Where the ratings work", "",
            "Every prediction uses only ratings from before that week, so each year is out-of-sample. "
            "Early years are weaker because nobody has a rating history yet.", "",
            "**By year (all sets):**", "", md_table(by_year), "",
            f"**By event size (sets from {TEST_FROM:%Y}):**", "", md_table(by_size), "",
            f"**By player experience (sets from {TEST_FROM:%Y}):**", "", md_table(by_exp), ""]


def tau_sensitivity(sets_c: pd.DataFrame, out: list, flags: list):
    rows = {}
    for tau in (0.3, 0.5, 0.8, 1.2):
        log(f"  Glicko-2 with tau={tau} ...")
        rated, _ = B.glicko2(sets_c, tau=tau)
        t = rated[rated.date >= TEST_FROM]
        rows[tau] = metric_rows(pd.DataFrame({"p": B.expected(t.p1_r, t.p1_rd, t.p2_r, t.p2_rd), "y": t.p1_won}))
    tab = pd.DataFrame(rows).T
    tab.index.name = "tau"
    tab["sets"] = tab["sets"].astype(int)
    tab.to_csv(TAB / "tau_sensitivity.csv")
    spread = tab.log_loss.max() - tab.log_loss.min()
    tab["log_loss"] = tab["log_loss"].map(lambda v: f"{v:.5f}")
    flags.append(("Results insensitive to Glicko tau (log-loss spread < 0.005)", spread < 0.005, f"{spread:.5f}"))
    out += ["## 3. Glicko-2 sensitivity", "",
            f"Held-out accuracy for different volatility settings (the model uses tau = {B.TAU}). "
            f"Log-loss spread across settings: **{spread:.5f}**.", "", md_table(tab, ".4f"), ""]


# ── 4 + 5: matchup model ───────────────────────────────────────────────

def fit_strengths(d, chars, pairs, lam_s, lam_i):
    w = B.fit_logistic(B.design(d, chars, pairs), d.y.to_numpy(), len(chars), lam_s, lam_i)
    return pd.Series(w[1:1 + len(chars)], index=chars), w


def character_checks(games: pd.DataFrame, rated: pd.DataFrame, model: dict, out: list, flags: list):
    d = B.prep_games(games, rated, seed=SEED)
    chars = model["characters"]
    pairs = sorted(tuple(k.split("|")) for k in model["interaction"])
    lam_s, lam_i = model["penalty"]["strength"], model["penalty"]["interaction"]
    tr, te = d[d.date < TEST_FROM], d[d.date >= TEST_FROM]

    # 4. does adding characters improve held-out predictions? bootstrap over events
    log("  value of characters (paired bootstrap by event) ...")
    X_tr, X_te = B.design(tr, chars, pairs), B.design(te, chars, pairs)
    w0 = B.fit_logistic(X_tr[:, :1], tr.y.to_numpy(), 0, 0, 0)
    w1 = B.fit_logistic(X_tr, tr.y.to_numpy(), len(chars), lam_s, lam_i)
    y = te.y.to_numpy()

    def ll(z):
        p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    gain = ll(X_te[:, :1] @ w0) - ll(X_te @ w1)              # > 0 means characters helped
    per_event = pd.DataFrame({"event": te.event_id.to_numpy(), "gain": gain}).groupby("event").gain.agg(["sum", "size"])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_event), size=(N_BOOT, len(per_event)))
    s, n = per_event["sum"].to_numpy(), per_event["size"].to_numpy()
    boot = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mean_gain = gain.mean()
    acc0 = ((X_te[:, :1] @ w0 > 0) == (y == 1)).mean()
    acc1 = ((X_te @ w1 > 0) == (y == 1)).mean()
    flags.append(("Characters improve held-out predictions (bootstrap CI above 0)", lo > 0,
                  f"{mean_gain:.4f} [{lo:.4f}, {hi:.4f}]"))
    out += ["## 4. Do characters add predictive value?", "",
            f"Fit on games before {TEST_FROM:%Y-%m-%d}, scored on {len(te):,} later games from "
            f"{len(per_event):,} events. Confidence interval from {N_BOOT} bootstrap resamples of whole events.", "",
            "| | log loss | accuracy |", "| --- | --- | --- |",
            f"| ratings only | {ll(X_te[:, :1] @ w0).mean():.4f} | {acc0:.3f} |",
            f"| ratings + characters | {ll(X_te @ w1).mean():.4f} | {acc1:.3f} |", "",
            f"Log-loss gain from characters: **{mean_gain:.4f}** (95% CI {lo:.4f} to {hi:.4f}). "
            + ("The gain is small but reliably above zero." if lo > 0 else
               "The interval includes zero: characters do not reliably improve predictions."), ""]

    # 5. stability of the character-strength ranking
    log("  character-strength stability (random halves, time split, penalty) ...")
    ev = d.event_id.unique()
    half = set(np.random.default_rng(SEED).permutation(ev)[: len(ev) // 2])
    in_half = d.event_id.isin(half)
    s_a, _ = fit_strengths(d[in_half], chars, pairs, lam_s, lam_i)
    s_b, _ = fit_strengths(d[~in_half], chars, pairs, lam_s, lam_i)
    cut = pd.Timestamp("2025-01-01")
    s_early, _ = fit_strengths(d[d.date < cut], chars, pairs, lam_s, lam_i)
    s_late, _ = fit_strengths(d[d.date >= cut], chars, pairs, lam_s, lam_i)
    s_full = pd.Series(model["strength"])
    s_pen, _ = fit_strengths(d, chars, pairs, lam_s * 10, lam_i * 10)

    def compare(x, y):
        top_x, top_y = set(x.nlargest(10).index), set(y.nlargest(10).index)
        return pd.Series({"spearman": x.corr(y, method="spearman"), "top10_overlap": len(top_x & top_y)})

    stab = pd.DataFrame({
        "random half A vs half B (by event)": compare(s_a, s_b),
        "before 2025 vs 2025 onward": compare(s_early, s_late),
        f"penalty {lam_s}/{lam_i} vs {lam_s * 10}/{lam_i * 10}": compare(s_full, s_pen),
    }).T
    stab.index.name = "comparison"
    stab["top10_overlap"] = stab["top10_overlap"].astype(int).astype(str) + " / 10"
    stab.to_csv(TAB / "strength_stability.csv")
    pd.DataFrame({"full": s_full, "half_a": s_a, "half_b": s_b, "before_2025": s_early,
                  "from_2025": s_late}).sort_values("full", ascending=False).to_csv(TAB / "strengths_by_split.csv")
    r_half = stab.iloc[0]["spearman"]
    flags.append((f"Character ranking reproducible across random halves (Spearman > {STABILITY_OK})",
                  r_half > STABILITY_OK, f"{r_half:.2f}"))
    out += ["## 5. Is the character-strength ranking stable?", "",
            "The skill-adjusted character strengths (the data-driven tier list) refitted on different "
            "slices of the data. High correlation between random halves means the ranking is not noise; "
            "a lower before/after-2025 correlation can reflect genuine meta change (patches ended in 2021, "
            "but play styles kept evolving).", "", md_table(stab, ".2f"), "",
            "Full per-split strengths: `reports/tables/strengths_by_split.csv`.", ""]


def main():
    for d in (REPORT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    out, flags = [], []

    log("loading data ...")
    data = clean.load_clean(ROOT)
    sets_c, games = data["sets"], data["games"]
    rated = pd.read_csv(ROOT / "data" / "models" / "sets_rated.csv.gz", parse_dates=["date"])
    model = __import__("json").loads((ROOT / "data" / "models" / "matchup_model.json").read_text())

    log("1-2. rating calibration and breakdowns ...")
    rating_checks(rated, sets_c, out, flags)
    log("3. Glicko-2 tau sensitivity ...")
    tau_sensitivity(sets_c, out, flags)
    log("4-5. matchup model checks ...")
    character_checks(games, rated, model, out, flags)

    summary = ["# Validation and robustness report", "",
               f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} from `data/processed` and `data/models`. "
               f"Held-out period: sets from {TEST_FROM:%Y-%m-%d}.", "",
               "| Check | Result | Value |", "| --- | --- | --- |"]
    summary += [f"| {name} | {'pass' if ok else '**check**'} | {val} |" for name, ok, val in flags]
    (REPORT / "validation.md").write_text("\n".join(summary + [""] + out), encoding="utf-8")
    print("\n".join(summary))
    log(f"wrote {REPORT / 'validation.md'}")


if __name__ == "__main__":
    main()
