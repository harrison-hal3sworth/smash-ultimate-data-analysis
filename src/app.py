"""
app.py  —  Smash Ultimate matchup dashboard (PM deliverables 1-5).

Run from the project root:
    pip install streamlit
    streamlit run src/app.py

Needs data/models/ from step 05 (python src/05_build_models.py).
Character icons: optional, from data/models/character_icons.csv (python src/06_fetch_icons.py).
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import SmashModel  # noqa: E402

NONE = "—"

st.set_page_config(page_title="Smash Ultimate Matchups", layout="wide")


@st.cache_resource
def load_model() -> SmashModel:
    return SmashModel()


@st.cache_data
def load_icons() -> dict:
    f = Path(__file__).resolve().parents[1] / "data" / "models" / "character_icons.csv"
    if not f.exists():
        return {}
    icons = pd.read_csv(f)
    return dict(zip(icons.char, icons.url))


m = load_model()
icons = load_icons()


def icon(char: str, width: int = 40):
    url = icons.get(char)
    if url:
        st.image(url, width=width)


def pct(p: float) -> str:
    return "<1%" if p < 0.005 else ">99%" if p > 0.995 else f"{p:.0%}"


def char_label(char: str, share: float | None = None) -> str:
    return f"{char} {share:.0%}" if share is not None else char


# ── header ─────────────────────────────────────────────────────────────

st.title("Smash Ultimate matchup predictor")
st.caption("US offline singles events with 32+ entrants, 2019 to 2026, from start.gg. "
           "Pick two players, two characters, or both.")

best_of = st.radio("Set length", [3, 5], format_func=lambda n: f"Best of {n}", horizontal=True)
st.divider()


# ── one column per side ────────────────────────────────────────────────

def side(col, key: str, title: str):
    """Player search -> character split -> character picker. Returns (player_id, character)."""
    with col:
        st.subheader(title)

        # 1. search by start.gg tag
        query = st.text_input("start.gg tag", key=f"q_{key}", placeholder="e.g. Sparg0")
        pid = None
        if query.strip():
            hits = m.search(query, limit=15)
            if hits.empty:
                st.warning(f"No player found for '{query}'.")
            else:
                options = [None] + list(hits.index)
                pid = st.selectbox(
                    "Which player?", options, key=f"p_{key}",
                    format_func=lambda p: NONE if p is None else m.label(p),
                    index=1 if len(hits) == 1 else 0,
                )

        # 2. character split for the chosen player
        if pid is not None:
            split = m.usage(pid)
            if split is None:
                st.info("No character data reported for this player.")
            else:
                window = m.usage_tbl.loc[m.usage_tbl.player_id == pid, "window"].iat[0]
                st.markdown(f"**Most-used characters** ({window})")
                for char, share in split:
                    c_icon, c_bar = st.columns([1, 6])
                    with c_icon:
                        if char != "Other":
                            icon(char, width=36)
                    with c_bar:
                        st.progress(min(max(share, 0.0), 1.0), text=char_label(char, share))

        # 3. character picker (main character first when known)
        main = m.main_char(pid) if pid is not None else None
        chars = sorted(m.characters)
        if main:
            chars = [main] + [c for c in chars if c != main]
        char = st.selectbox("Character", [NONE] + chars, key=f"c_{key}")
        char = None if char == NONE else char
        if char:
            icon(char, width=64)
        return pid, char


left, right = st.columns(2, gap="large")
pa, ca = side(left, "a", "Player 1")
pb, cb = side(right, "b", "Player 2")
st.divider()


# ── 4 + 5: prediction ──────────────────────────────────────────────────

def name(pid, char, fallback):
    if pid is not None:
        tag = m.players.at[pid, "tag"]
        return f"{tag} ({char})" if char else tag
    return char or fallback


both_players = pa is not None and pb is not None
both_chars = ca is not None and cb is not None

if both_players and pa == pb:
    st.warning("Pick two different players.")
elif not both_players and not both_chars:
    st.info("Select two players, two characters, or both to see a prediction.")
else:
    res = m.predict(pa if both_players else None, pb if both_players else None,
                    ca if both_chars or both_players else None,
                    cb if both_chars or both_players else None, best_of=best_of)
    # use the characters the model actually used (a missing pick is filled with that player's main)
    used_a, used_b = res["char_a"], res["char_b"]
    n1 = name(pa if both_players else None, used_a, "Player 1")
    n2 = name(pb if both_players else None, used_b, "Player 2")
    if both_players and (used_a, used_b) != (ca, cb) and used_a and used_b:
        st.caption("One character wasn't picked, so that player's most-used character is assumed.")

    st.subheader("Prediction")
    st.caption({"players + characters": "Players' ratings plus the character matchup.",
                "characters only": "Character matchup alone, assuming two equally skilled players.",
                "players only": "Players' ratings only (no characters selected)."}[res["mode"]])
    if "headline" in res:
        st.markdown(f"### {res['headline']}")

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.metric(n1, pct(res["p_set"]))
    with c2:
        st.markdown(f"<h2 style='text-align:center'>{res['set_split']}</h2>", unsafe_allow_html=True)
    with c3:
        st.metric(n2, pct(1 - res["p_set"]))
    st.progress(res["p_set"], text=f"Set win chance, best of {best_of}")
    st.caption(f"Per game: {res['game_split']}. The set figure assumes games are independent.")
    for note in res["notes"]:
        st.warning(note)

with st.expander("How this works"):
    st.markdown(
        "- **Ratings:** Glicko-2 over 2.8M completed sets. Ratings pick the winner of about 76% of 2026 sets.\n"
        "- **Rank:** among players with 10+ sets in the last 12 months and a settled rating (RD ≤ 100).\n"
        "- **Characters:** a matchup model fitted on 1.37M games with both characters reported, "
        "after accounting for player ratings. Matchups with few games are pulled toward each "
        "character's overall strength.\n"
        "- **Limits:** characters are reported for about 18% of sets, mostly 2024 to 2026; "
        "character choice adds only a little predictive power once skill is known."
    )
