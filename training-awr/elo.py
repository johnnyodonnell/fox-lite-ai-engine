"""Elo ladder bookkeeping for snapshot evaluation.

`random` is anchored at 0 so ratings are comparable across the whole run. A new
candidate is rated against the (fixed) ratings of its opponents by finding the
Elo that matches its observed total score (logistic MLE via bisection). Older
ratings are not refit, giving a stable anchored ladder.
"""

import json
from pathlib import Path


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def fit_rating(results, ratings, default_opp=0.0) -> float:
    total_games = sum(r["games"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    if total_games == 0:
        return 0.0
    score = total_wins / total_games
    eps = 1.0 / (2 * total_games)  # keep rating finite at 0%/100%
    score = min(max(score, eps), 1 - eps)
    lo, hi = -2000.0, 4000.0
    for _ in range(80):
        r = (lo + hi) / 2
        exp = sum(
            x["games"] * expected(r, ratings.get(x["opponent"], default_opp))
            for x in results
        ) / total_games
        if exp < score:
            lo = r
        else:
            hi = r
    return (lo + hi) / 2


def load_ladder(run_dir) -> dict:
    p = Path(run_dir) / "elo.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.setdefault("ratings", {})
    d["ratings"].setdefault("random", 0.0)
    d.setdefault("history", [])
    return d


def update_ladder(run_dir, candidate: str, results, t=None) -> float:
    d = load_ladder(run_dir)
    rating = fit_rating(results, d["ratings"])
    d["ratings"][candidate] = rating
    d["history"].append({"candidate": candidate, "elo": rating, "t": t, "results": results})
    (Path(run_dir) / "elo.json").write_text(json.dumps(d, indent=2))
    return rating
