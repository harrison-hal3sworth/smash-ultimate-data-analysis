# Contains Claude generated code

"""
predict.py  —  answers the dashboard's questions from the step-05 model files.

    from predict import SmashModel
    m = SmashModel()                              # loads data/models/
    m.search("sparg0")                            # candidate players (deliverable 1)
    m.usage(1234567)                              # character split, e.g. top 2 + Other (2)
    m.predict(a=..., b=..., char_a="Cloud", char_b="Steve")   # win % (4, 5a, 5b)

Every case in the PM's spec goes through predict():
    players + characters  -> rating gap + matchup effect + each player's skill on their character
    characters only       -> matchup effect alone (equal-skill players)
    players only          -> higher-ranked player favoured, probability from ratings
"""

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean  # noqa: E402

R0, RD0 = 1500.0, 350.0


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def set_prob(p: float, best_of: int = 3) -> float:
    """Chance of winning a best-of-N set from a per-game win chance p (games independent)."""
    need = best_of // 2 + 1
    q = 1 - p
    # win `need` games before losing `need`: sum over the number of games lost first
    return sum(math.comb(need - 1 + k, k) * p ** need * q ** k for k in range(need))


def split(p: float) -> str:
    """0.6 -> '60-40'."""
    a = round(100 * p)
    return f"{a}-{100 - a}"


class SmashModel:
    def __init__(self, root: Path | None = None):
        root = root or clean.find_root(Path(__file__).parent)
        d = root / "data" / "models"
        self.players = pd.read_csv(d / "players.csv", index_col="player_id", low_memory=False)
        self.players["tag"] = self.players["tag"].astype(str)
        self.players["other_tags"] = self.players["other_tags"].fillna("").astype(str)
        self.usage_tbl = pd.read_csv(d / "usage.csv")
        # player-on-character table (games per player and character + fitted adjustment);
        # older model folders without it simply skip the player-on-character term
        pc_path = d / "player_char.csv"
        if pc_path.exists():
            pc = pd.read_csv(pc_path)
            self.pc_games = pc.set_index(["player_id", "char"]).games.to_dict()
            self.pc_adj = pc[pc.adj != 0].set_index(["player_id", "char"]).adj.to_dict()
            self.player_games = pc.groupby("player_id").games.sum().to_dict()
        else:
            self.pc_games, self.pc_adj, self.player_games = {}, {}, {}
        self.model = json.loads((d / "matchup_model.json").read_text(encoding="utf-8"))
        self.characters = self.model["characters"]
        self.beta, self.scale = self.model["beta"], self.model["scale"]
        self.gamma = self.model.get("experience_coef", 0.0)
        self._tag_lower = self.players.tag.str.lower()
        self._other_lower = self.players.other_tags.str.lower()

    # ── deliverable 1: find players ──
    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Players whose current or past tag matches. Exact matches first, then
        prefix, then substring; ties broken by recent activity."""
        q = query.strip().lower()
        if not q:
            return self.players.head(0)
        exact = self._tag_lower == q
        prefix = self._tag_lower.str.startswith(q)
        contains = self._tag_lower.str.contains(q, regex=False)
        past = self._other_lower.str.contains(q, regex=False)
        hit = exact | prefix | contains | past
        res = self.players[hit].assign(
            _score=np.select([exact[hit], prefix[hit], contains[hit]], [3, 2, 1], 0))
        res = res.sort_values(["_score", "sets_last_12m", "sets"], ascending=False).head(limit)
        cols = ["tag", "other_tags", "state", "last_event", "last_date", "rank", "rating", "sets", "sets_last_12m"]
        return res[cols]

    def label(self, pid: int) -> str:
        """Display string that tells same-tag players apart."""
        p = self.players.loc[pid]
        rank = f"#{int(p['rank'])}" if pd.notna(p["rank"]) else "unranked"
        return f"{p.tag} ({p.state or '??'}, {rank}, last played {p.last_date})"

    # ── deliverable 2: character split ──
    def usage(self, pid: int, top: int = 2) -> list[tuple[str, float]] | None:
        """[(character, share), ..., ('Other', share)] or None if no character data."""
        u = self.usage_tbl[self.usage_tbl.player_id == pid]
        if u.empty:
            return None
        head = list(zip(u.char.head(top), u.share.head(top)))
        rest = 1 - sum(s for _, s in head)
        return head + ([("Other", rest)] if rest > 0.005 else [])

    def main_char(self, pid: int) -> str | None:
        u = self.usage_tbl[self.usage_tbl.player_id == pid]
        return None if u.empty else u.char.iat[0]

    # ── model pieces ──
    def rating(self, pid: int) -> tuple[float, float]:
        """(rating, RD); unknown or unrated players get the starting 1500 / 350."""
        if pid in self.players.index and pd.notna(self.players.at[pid, "rating"]):
            return float(self.players.at[pid, "rating"]), float(self.players.at[pid, "rd"])
        return R0, RD0

    def rating_gap(self, a: int, b: int) -> float:
        """Same RD-discounted gap the model was trained on."""
        (ra, da), (rb, db) = self.rating(a), self.rating(b)
        g = 1 / math.sqrt(1 + 3 * ((da ** 2 + db ** 2) / self.scale ** 2) / math.pi ** 2)
        return g * (ra - rb) / self.scale

    def matchup_logit(self, a: str, b: str) -> float:
        s, inter = self.model["strength"], self.model["interaction"]
        lo, hi = sorted((a, b))
        i = inter.get(f"{lo}|{hi}", 0.0)
        return s[a] - s[b] + (i if a == lo else -i)

    def player_char_logit(self, pid: int | None, char: str | None) -> float:
        """Player-on-character term: familiarity with the character (share of the player's
        games on it) plus the player's fitted over/under-performance on it. 0 for unknown players."""
        if pid is None or char is None:
            return 0.0
        n_p = self.player_games.get(pid, 0)
        if n_p == 0:                     # no character history: neutral, as in training
            return 0.0
        n_pc = self.pc_games.get((pid, char), 0)
        return self.gamma * math.log((1 + n_pc) / (1 + n_p)) + self.pc_adj.get((pid, char), 0.0)

    def matchup_games(self, a: str, b: str) -> int:
        lo, hi = sorted((a, b))
        return self.model["games"].get(f"{lo}|{hi}", 0)

    # ── deliverables 4 + 5: win percentage ──
    def predict(self, a: int | None = None, b: int | None = None,
                char_a: str | None = None, char_b: str | None = None, best_of: int = 3) -> dict:
        players = a is not None and b is not None
        chars = char_a is not None and char_b is not None

        # one character picked with both players: fill the other with that player's main
        if players and (char_a is None) != (char_b is None):
            char_a = char_a or self.main_char(a)
            char_b = char_b or self.main_char(b)
            chars = char_a is not None and char_b is not None

        if not players and not chars:
            raise ValueError("Pick two players, two characters, or both.")

        z, notes = 0.0, []
        if players:
            z += self.beta * self.rating_gap(a, b)
        if players and chars:
            for pid, c in ((a, char_a), (b, char_b)):
                z += (1 if pid == a else -1) * self.player_char_logit(pid, c)
                n_p = self.player_games.get(pid, 0)
                n_pc = self.pc_games.get((pid, c), 0)
                if n_p >= 20 and n_pc / n_p < 0.1:
                    notes.append(f"{self.players.at[pid, 'tag']} has played {c} in {n_pc} of {n_p} reported games; "
                                 "an unfamiliar character lowers their chance.")
        if chars:
            if char_a == char_b:
                notes.append("Mirror match: the characters cancel out.")
            else:
                z += self.matchup_logit(char_a, char_b)
                n = self.matchup_games(char_a, char_b)
                if n < 100:
                    notes.append(f"Only {n} games of {char_a} vs {char_b} in the data; "
                                 "this matchup is estimated mostly from each character's overall strength.")

        p_game = _sigmoid(z)
        out = {
            "mode": "players + characters" if players and chars else "players only" if players else "characters only",
            "p_game": p_game, "p_set": set_prob(p_game, best_of), "best_of": best_of,
            "game_split": split(p_game), "set_split": split(set_prob(p_game, best_of)),
            "notes": notes,
            "char_a": char_a if chars else None, "char_b": char_b if chars else None,
        }
        if players and not chars:
            ra, rb = self.players.at[a, "rank"], self.players.at[b, "rank"]
            if pd.notna(ra) and pd.notna(rb):
                fav = a if ra < rb else b
            else:
                fav = a if self.rating(a)[0] >= self.rating(b)[0] else b
            out["favoured"] = fav
            out["headline"] = f"{self.players.at[fav, 'tag']} is the higher-ranked player and is favoured."
        return out


    # ── deliverable 6: auto-seeding ──
    def seed(self, player_ids: list, names: list | None = None, use_chars: bool = True,
             best_of: int = 3) -> pd.DataFrame:
        """Seed a field by each player's average chance of beating everyone else.
        player_ids may contain None for players not in the data (rated 1500 / 350, no character)."""
        import seeder
        names = names or [None] * len(player_ids)
        ratings = [self.rating(p) if p is not None else (R0, RD0) for p in player_ids]
        chars = [self.main_char(p) if (p is not None and use_chars) else None for p in player_ids]
        offsets = [self.player_char_logit(p, c) for p, c in zip(player_ids, chars)] if use_chars else None
        P = seeder.pairwise_set_probs(self.model, [r for r, _ in ratings], [d for _, d in ratings],
                                      chars if use_chars else None, best_of, offsets)
        share = seeder.expected_win_share(P)
        rows = []
        for i, pid in enumerate(player_ids):
            known = pid is not None
            rows.append({
                "player": self.players.at[pid, "tag"] if known else (names[i] or "unknown"),
                "player_id": pid,
                "state": self.players.at[pid, "state"] if known else None,
                "rank": self.players.at[pid, "rank"] if known else None,
                "rating": round(ratings[i][0]),
                "main": chars[i],
                "avg_win_chance": share[i],
            })
        out = pd.DataFrame(rows).iloc[seeder.seed_order(P)].reset_index(drop=True)
        out.insert(0, "seed", range(1, len(out) + 1))
        return out


if __name__ == "__main__":
    # quick smoke test from the command line: python src/predict.py "tagA" "tagB" [charA charB]
    m = SmashModel()
    args = sys.argv[1:]
    if len(args) >= 2:
        pa, pb = m.search(args[0]).index[0], m.search(args[1]).index[0]
        ca, cb = (args[2], args[3]) if len(args) >= 4 else (None, None)
        print(m.label(pa), "vs", m.label(pb))
        print("  usage:", m.usage(pa), "|", m.usage(pb))
        print("  ", m.predict(pa, pb, ca, cb))
