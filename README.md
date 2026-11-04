# Smash Ultimate Data Analysis

*Claude generated README.md*

Do Super Smash Bros. Ultimate tier lists and matchup charts hold up against real tournament results once you account for player skill?

This project pulls tournament data from the [start.gg](https://start.gg) API, rates every player, models character matchups after controlling for skill, and wraps the results in an interactive dashboard.

**Data:** US offline singles events with 32+ entrants, January 2019 to October 2026. That's 26,238 events, 2.78M completed sets and 1.39M games with both characters reported.

---

## Run the dashboard

You need **Python 3.10 or newer** and **Git**. No API key or raw data is needed: the fitted models are included in the repo.

```bash
git clone https://github.com/harrison-hal3sworth/smash-ultimate-data-analysis.git
cd smash-ultimate-data-analysis

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run src/app.py
```

The app opens in your browser at `http://localhost:8501`. The first load takes a few seconds while it reads the player table. Character icons are loaded from start.gg, so they need an internet connection. The app works without them.

### Using it

1. **Search for two players** by their start.gg tag. Past tags work too. When several players share a tag, the dropdown separates them by state, rank and last event, e.g. `Sparg0 (CA, #3, last played 2026-08-06)`.
2. **See each player's character split**, e.g. Cloud 80%, Pyra & Mythra 10%, Other 9%, over the last 12 months.
3. **Pick a character for each side** (optional). The player's main character is listed first.
4. **Read the prediction**, shown as a set split (e.g. 57-43) with the per-game split underneath. Choose best-of-3 or best-of-5 at the top.

| You select | The prediction uses |
| --- | --- |
| Two players and two characters | Both players' ratings plus the character matchup |
| Two characters only | The character matchup alone, assuming equally skilled players |
| Two players only | Ratings only; the higher-ranked player is favoured |
| Two players, one character | The other player's most-used character is assumed |

**Rank (`#3`)** is the player's position among those with 10+ sets in the last 12 months and a settled Glicko-2 rating (rating deviation ≤ 100). Other players show as "unranked" but still have a rating that the predictions use. **State** is where the player has competed most often.

---

## How it works

- **Player ratings (Glicko-2):** every completed set, in date order, in weekly rating periods. Each prediction uses only ratings from before the match. Ratings pick the winner of **75.8%** of 2026 sets.
- **Matchup model:** a logistic regression on 1.37M games. It combines the players' rating gap, each character's overall strength, and a matchup-specific adjustment. Rarely played matchups are pulled toward the two characters' overall strengths.
- **Main finding so far:** skill explains most outcomes. Characters improve per-game accuracy only from 69.4% to 69.7%, a small but statistically reliable gain.

Full results: [`reports/validation.md`](reports/validation.md) and [`data/models/model_report.txt`](data/models/model_report.txt).

---

## Project structure

```
src/
  data_pull.py            start.gg API client (rate limiting, retries)
  01_pull_tournaments.py  US Ultimate tournaments, month by month
  02_filter_events.py     keep offline singles events with 32+ entrants
  03_pull_sets.py         sets and games for each event (resumable)
  run_pull.py             supervisor for step 03: restarts, keeps the PC awake
  04_build_dataset.py     raw JSON -> sets / games / player_games tables
  clean.py                cleaning rules + summary functions (shared)
  05_build_models.py      player table, Glicko-2 ratings, usage, matchup model
  06_fetch_icons.py       character icon links from start.gg
  07_validate.py          validation and robustness report
  predict.py              search, character split and win probability (used by the app)
  app.py                  Streamlit dashboard
EDA/cleaning.ipynb        exploratory analysis (figures in outputs/figures/)
tests/                    pytest suite
data/models/              fitted models used by the app (committed)
reports/                  validation report, figures and tables
```

`data/raw/`, `data/interim/` and `data/processed/` aren't in the repo, because they're large and can be regenerated with the steps below.

---

## Reproduce everything from scratch

You only need this to rebuild the data or the models. It requires a start.gg API token: create one under start.gg → Profile → Developer Settings, then save it as `secrets/key.txt` (git-ignored) or set the `STARTGG_TOKEN` environment variable.

```bash
python src/01_pull_tournaments.py   # ~1 hour
python src/02_filter_events.py      # seconds
python src/run_pull.py              # step 03 under a supervisor, ~50 hours for all events
python src/04_build_dataset.py      # a few minutes
python src/05_build_models.py       # ~5 minutes
python src/06_fetch_icons.py        # one API call
python src/07_validate.py           # a few minutes
```

Steps 01 and 03 can be stopped and restarted at any time, and they pick up where they left off. start.gg rate-limits each token, so run only one pull at a time.

---

## Tests

```bash
pytest -m "not data"   # 45 unit tests on small made-up data; no data needed, a few seconds
pytest -m data         # 10 checks on the real pipeline outputs (needs data/processed and data/models)
```

The unit tests cover the cleaning rules, the Glicko-2 implementation (checked against Glickman's published worked example), leakage, the matchup model and every dashboard case.

---

## Limitations

- Characters are reported for about 18% of sets, mostly from 2024 to 2026 and mid-size events, so results describe events that report characters.
- Predictions for players with fewer than 5 prior sets are much weaker (about 61% accuracy).
- Covers events held in the US, so players from other countries appear only through US events.
- start.gg records Pokémon Trainer and Pyra & Mythra as single picks.

## Credits

Tournament data from the [start.gg API](https://developer.start.gg). Character icons are hosted by start.gg. Super Smash Bros. characters and artwork are property of Nintendo. This is a non-commercial student project.
