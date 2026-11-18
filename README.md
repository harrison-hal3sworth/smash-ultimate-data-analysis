# Smash Ultimate Data Analysis

Do Super Smash Bros. Ultimate tier lists and matchup charts hold up against real tournament results once you account for player skill?

This project pulls tournament data from the [start.gg](https://start.gg) API, rates every player, models character matchups after controlling for skill, and wraps the results in an interactive dashboard.

**Data:** US offline singles events with 32+ entrants, January 2019 to October 2026. That's 26,238 events, 2.78M completed sets and 1.39M games with both characters reported. The data is refreshed with `src/update.py` (see [Keeping the data up to date](#keeping-the-data-up-to-date)); each refresh is logged in `reports/update_history.csv`.

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

### Auto-seeder tab

Paste a tournament's entrants, one start.gg tag per line, to get a seeding. Each player is assumed to use their most-used character, and players are ordered by their average chance of beating everyone else in the field. Unknown tags are seeded as new players (rating 1500), and tags shared by several active players are flagged for you to check. The seeding downloads as a CSV.

**Rank (`#3`)** is the player's position among those with 10+ sets in the last 12 months and a settled Glicko-2 rating (rating deviation ≤ 100). Other players show as "unranked" but still have a rating that the predictions use. **State** is where the player has competed most often.

---

## How it works

- **Player ratings (Glicko-2):** every completed set, in date order, in weekly rating periods. Each prediction uses only ratings from before the match. Ratings pick the winner of **75.8%** of 2026 sets.
- **Matchup model:** a logistic regression on 1.37M games. It combines the players' rating gap, each character's overall strength, and a matchup-specific adjustment. Rarely played matchups are pulled toward the two characters' overall strengths.
- **Player-on-character term:** how familiar each player is with the character they picked (the share of their earlier games on it), plus how much better or worse they do on that character than on their other characters. Picking a character a player rarely uses lowers their chance, and the dashboard says so. A separate player-level adjustment is fitted alongside, so these terms only describe differences *between a player's characters*; the dashboard leaves the player-level part out, so player strength is still the rating behind the rank.
- **Main finding so far:** skill explains most outcomes. On 2026 games, characters alone improve per-game accuracy from 69.4% to 69.7%; adding the player-on-character term raises it to 70.0% and cuts log loss by 4.6 times as much as the matchups alone did (0.0073 vs 0.0016). Both gains are statistically reliable.

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
  predict.py              search, character split, win probability and seeding (used by the app)
  seeder.py               auto-seeding + seeding evaluation metrics
  08_pull_seeding.py      organizer seeds + final placements for test events
  09_validate_seeding.py  auto-seeder vs organizer seeding report
  update.py               one-command data refresh (see below)
  app.py                  Streamlit dashboard
scripts/update_weekly.bat runs update.py from Windows Task Scheduler
EDA/cleaning.ipynb        exploratory analysis (figures in outputs/figures/)
tests/                    pytest suite
data/models/              fitted models used by the app (committed)
reports/                  validation reports, update history, figures and tables
logs/                     update logs (not committed)
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
python src/08_pull_seeding.py       # seeds + placements for 50 test events, ~5 minutes
python src/09_validate_seeding.py   # auto-seeder validation -> reports/seeding_validation.md
```

Steps 01 and 03 can be stopped and restarted at any time, and they pick up where they left off. start.gg rate-limits each token, so run only one pull at a time.

After this first build, you don't need to run these steps by hand: `src/update.py` (below) runs 01–05 for whatever is new.

---

## Keeping the data up to date

One command refreshes everything: it pulls tournaments and sets added since the last run, rebuilds the tables and models, and checks the result.

```bash
python src/update.py --dry-run   # show what would be refreshed; changes nothing
python src/update.py             # full update, about 20-40 minutes for a week of new events
python src/update.py --commit    # ... then commit + push data/models so others can `git pull`
```

Each run:

1. Re-pulls the tournament list for every month since the data was last brought up to date (the last successful update, or the last set pull before the first update), so a gap of weeks or months is filled in one run. It always covers at least the current and previous month, because organizers add and edit events late. To force a wider window, use `--since YYYY-MM`.
2. Pulls sets for every finished event that has no set data yet, whatever its date, and re-pulls events since the last update that were fetched while still in progress or came back empty.
3. Rebuilds `data/processed` and `data/models` (steps 04 and 05) from all of the data, so every rating, matchup and ranking reflects the new results.
4. Runs the data checks (`pytest -m data`). If the model build or a check fails, the previous models are restored, so the dashboard never shows a broken build.
5. Logs a before/after summary (new events, new games, top-10 changes) to `logs/` and adds a row to `reports/update_history.csv`.

Other options: `--skip-pull` rebuilds from data already on disk without calling the API, and `--validate` also re-runs the validation report. Only one update can run at a time; a crashed run's lock (`data/.update.lock`) is ignored after 12 hours.

**To get the latest data on another computer**, run `git pull` and restart the dashboard.

### Run it weekly (Windows)

`scripts/update_weekly.bat` runs `update.py --commit` with the project's virtual environment. To schedule it for 3 a.m. every Monday, open a terminal in the project folder and run:

```bat
schtasks /create /tn "Smash data update" /tr "\"%cd%\scripts\update_weekly.bat\"" /sc weekly /d MON /st 03:00
```

The computer must be on and awake at that time. To run it now as a test: `schtasks /run /tn "Smash data update"`. Output goes to `logs/scheduler.log`. For `--commit` to push, Git on that computer must already be signed in to GitHub.

---

## Tests

```bash
pytest -m "not data"   # 93 unit tests on small made-up data; no data needed, a few seconds
pytest -m data         # 10 checks on the real pipeline outputs (needs data/processed and data/models)
```

The unit tests cover the cleaning rules, the Glicko-2 implementation (checked against Glickman's published worked example), leakage, the matchup model, the auto-seeder, every dashboard case, the player-on-character term (no leakage from future games, player vs character effects kept apart) and the update workflow (which months to re-pull, catching up after a long gap, which events count as incomplete).

---

## Limitations

- Characters are reported for about 18% of sets, mostly from 2024 to 2026 and mid-size events, so results describe events that report characters.
- Predictions for players with fewer than 5 prior sets are much weaker (about 61% accuracy).
- Covers events held in the US, so players from other countries appear only through US events.
- start.gg records Pokémon Trainer and Pyra & Mythra as single picks.

## Credits

Tournament data from the [start.gg API](https://developer.start.gg). Character icons are hosted by start.gg. Super Smash Bros. characters and artwork are property of Nintendo. This is a non-commercial student project.
