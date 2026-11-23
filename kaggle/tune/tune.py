"""
Part 4b on Kaggle: tuning the chosen kNN model and the app's scoring settings, on kaggle/build's output.

Similarity. The kNN of Part 3 is cosine on log weighted listens. Two changes are tried on top of it:
- popularity penalty (BETA): sim(i, j) = dot(i, j) / (|i|^(2(1 − BETA)) · |j|^(2 BETA)), the asymmetric cosine.
  BETA 0.5 is cosine, a higher BETA pushes widely listened candidates j down, a lower one pulls them up. A check
  after the pool is built stops the run if BETA 0.5 does not give back implicit's own cosine neighbours.
- shrinkage (SHRINK): the denominator gets + SHRINK, so a similarity resting on very few shared listeners counts less.
implicit has neither, so its cosine model gives every artist a pool of its POOL closest artists, and these are
scored again: the cosine score times |i|·|j| is exactly dot(i, j). Only the pool itself is approximate.

Scoring. The app's score() (playlist_tracker/music/recommend.py) has three settings: SEEDS (how many of the most
played artists suggestions start from), RELATED (how many similar artists each seed lists) and SEED_DECAY.

Stages, all on TUNE_USERS validation users and judged by NDCG@12 (the app shows 12):
  A. every similarity setting with the app's current scoring (10, 20, 0.5)
  B. every scoring setting with the best similarity of A
  C. the best few similarities of A again with the best scoring of B
Then the current setup (cosine, 10, 20, 0.5) and the best one are scored once on every test user.

Outputs in OUT: results.json (every setting tried), neighbours_knn_tuned.npz (EXPORT_N neighbours per artist for
the best similarity), summary.json (best settings and the test scores).
"""

import glob
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
import time

if not os.environ.get("TUNE_INPUT"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "implicit"], check=True)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from implicit.nearest_neighbours import CosineRecommender  # noqa: E402

KS = (12, 20)
TUNE_USERS = int(os.environ.get("TUNE_USERS", 2000))
POOL = int(os.environ.get("TUNE_POOL", 400))
EXPORT_N = 50
SEED = 42
CURRENT = {"beta": 0.5, "shrink": 0.0, "seeds": 10, "related": 20, "decay": 0.5}

BETAS = (0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8)
# The denominator at BETA 0.5 is |i|·|j|: from about 30 for two rare artists to 10^5 for two popular ones
SHRINKS = (0.0, 20.0, 100.0, 500.0)
SEEDS_GRID = (5, 10, 15, 20, 30)
RELATED_GRID = (10, 20, 30, 50)
DECAY_GRID = (0.25, 0.5, 0.75)
STAGE_C = 3

# TUNE_INPUT, TUNE_OUT and TUNE_QUICK let the script run locally on a small sample
INPUT = os.environ.get("TUNE_INPUT") or os.path.dirname(
    glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0])
OUT = os.environ.get("TUNE_OUT") or "/kaggle/working"
QUICK = bool(os.environ.get("TUNE_QUICK"))
if QUICK:
    BETAS, SHRINKS = (0.5, 0.7), (0.0, 20.0)
    SEEDS_GRID, RELATED_GRID, DECAY_GRID = (5, 10), (10, 20), (0.5,)
    STAGE_C = 2
os.makedirs(OUT, exist_ok=True)
results = []


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def save_results():
    with open(f"{OUT}/results.json", "w") as out:
        json.dump(results, out, indent=1)


# Data
artists = pd.read_parquet(f"{INPUT}/artists.parquet")
users = pd.read_parquet(f"{INPUT}/users.parquet")
train = pd.read_parquet(f"{INPUT}/train.parquet")
holdout = pd.read_parquet(f"{INPUT}/holdout.parquet")
n_users, n_artists = len(users), len(artists)
listens = sp.csr_matrix((train["listens"].to_numpy(np.float32), (train["u"].to_numpy(), train["a"].to_numpy())),
                        shape=(n_users, n_artists))
popularity = np.asarray((listens > 0).sum(axis=0)).ravel()
weighted = listens.copy()
weighted.data = np.log1p(weighted.data)
norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=0)).ravel()).astype(np.float32)
log(f"{n_users:,} users, {n_artists:,} artists, {listens.nnz:,} training pairs")


def eval_users(split, limit):
    """[(seed ids, seed weights, known ids, hidden ids)] for the users of a split, at most limit of them. The seeds
    are the user's most listened artists, as many as the largest SEEDS tried, and are cut per setting."""
    ids = users.loc[users["split"] == split, "u"].to_numpy()
    if limit and len(ids) > limit:
        ids = np.random.default_rng(SEED).choice(ids, limit, replace=False)
    hidden = holdout[holdout["split"] == split].groupby("u")["a"].apply(np.asarray)
    out = []
    for u in ids:
        row = listens.getrow(u)
        if row.nnz == 0 or u not in hidden.index:
            continue
        # The same sort as kaggle/train/train.py, so equal listen counts pick the same seeds
        order = np.argsort(-row.data)[:max(SEEDS_GRID)]
        out.append((row.indices[order], row.data[order], frozenset(row.indices.tolist()), hidden[u]))
    return out


# The pool: every artist's POOL closest artists by cosine, and the dot products behind them
start = time.time()
cosine = CosineRecommender(K=POOL)
cosine.fit(weighted, show_progress=False)
pool_ids = np.full((n_artists, POOL), -1, dtype=np.int32)
pool_dot = np.zeros((n_artists, POOL), dtype=np.float32)
for first in range(0, n_artists, 20000):
    chunk = np.arange(first, min(first + 20000, n_artists))
    ids, scores = cosine.similar_items(chunk, N=POOL + 1)
    for row_i, (i, row, values) in enumerate(zip(chunk, ids, scores)):
        mask = (row != i) & (row >= 0) & (values > 0)
        kept = row[mask][:POOL]
        pool_ids[i, :len(kept)] = kept
        pool_dot[i, :len(kept)] = values[mask][:POOL] * norms[i] * norms[kept]
del cosine
log(f"pool of {POOL} fitted in {time.time() - start:.0f}s")


def neighbours(beta, shrink, n=EXPORT_N):
    """Every artist's n most similar artists from its pool, best first, -1 where the pool runs out."""
    valid = pool_ids >= 0
    candidate_norms = norms[np.where(valid, pool_ids, 0)]
    score = pool_dot / (norms[:, None] ** (2 * (1 - beta)) * candidate_norms ** (2 * beta) + shrink + 1e-9)
    score[~valid] = -np.inf
    n = min(n, POOL)
    top = np.argpartition(-score, n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(score, top, axis=1), axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    out = np.take_along_axis(pool_ids, top, axis=1)
    out[~np.take_along_axis(valid, top, axis=1)] = -1
    return out


# BETA 0.5 without shrinkage must be implicit's cosine: its first 20 neighbours, compared on a sample of artists
_sample = np.random.default_rng(SEED).choice(n_artists, min(2000, n_artists), replace=False)
_ours = neighbours(0.5, 0.0, 20)[_sample]
_theirs = pool_ids[_sample, :20]
_same = float(np.mean([set(a[a >= 0].tolist()) == set(b[b >= 0].tolist()) for a, b in zip(_ours, _theirs)]))
log(f"check: beta 0.5 gives implicit's cosine neighbours for {_same:.1%} of the sample")
assert _same > 0.98, "the rescoring does not reproduce cosine"


def app_rank(seeds, weights, related, known, related_n, decay, k):
    """The app's score(): shares by place, then greedy picks, each pick multiplying its seeds' shares by decay."""
    shares = {}
    for s, w in zip(seeds, weights):
        candidates = [a for a in related[s][:related_n] if a >= 0]
        for place, a in enumerate(candidates):
            if a in known:
                continue
            entry = shares.setdefault(a, {})
            entry[s] = entry.get(s, 0.0) + w * (1 - place / max(len(candidates), related_n))
    used = {int(s): 0 for s in seeds}
    ranked = []
    while shares and len(ranked) < k:
        best = max(shares, key=lambda a: (sum(v * decay ** used[int(s)] for s, v in shares[a].items()),
                                          popularity[a]))
        for s in shares.pop(best):
            used[int(s)] += 1
        ranked.append(best)
    return ranked


def metrics(rankings, group):
    out = {}
    for k in KS:
        recall, hit, ndcg = [], [], []
        for ranked, (_, _, _, hidden) in zip(rankings, group):
            hidden_set = set(hidden.tolist())
            gains = [1.0 if a in hidden_set else 0.0 for a in ranked[:k]]
            found = sum(gains)
            recall.append(found / len(hidden_set))
            hit.append(1.0 if found else 0.0)
            ideal = sum(1 / np.log2(i + 2) for i in range(min(len(hidden_set), k)))
            ndcg.append(sum(g / np.log2(i + 2) for i, g in enumerate(gains)) / ideal)
        out[f"recall@{k}"] = round(float(np.mean(recall)), 4)
        out[f"hit@{k}"] = round(float(np.mean(hit)), 4)
        out[f"ndcg@{k}"] = round(float(np.mean(ndcg)), 4)
    return out


# Workers are forked, so they share the neighbour table and the users without copying them
_job = {}


def _rank_one(user):
    seeds, weights, known, _ = user
    p = _job["params"]
    n = p["seeds"]
    return app_rank(seeds[:n], weights[:n], _job["related"], known, p["related"], p["decay"], max(KS))


def evaluate(params, related, group, split, stage):
    start = time.time()
    _job.update(params=params, related=related)
    if "fork" in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context("fork").Pool(os.cpu_count()) as pool:
            rankings = pool.map(_rank_one, group, chunksize=64)
    else:  # Windows, for the local test only
        rankings = [_rank_one(user) for user in group]
    row = {"stage": stage, "split": split, "users": len(group), "params": params, **metrics(rankings, group),
           "eval_seconds": round(time.time() - start)}
    results.append(row)
    save_results()
    log(json.dumps(row))
    return row


def score_key(row):
    return (row["ndcg@12"], row["hit@12"])


val = eval_users("val", TUNE_USERS)
log(f"validation users: {len(val):,}")
tables = {}


def table(beta, shrink):
    if (beta, shrink) not in tables:
        tables[(beta, shrink)] = neighbours(beta, shrink)
    return tables[(beta, shrink)]


# A. similarity, with the app's current scoring
stage_a = []
for beta, shrink in itertools.product(BETAS, SHRINKS):
    params = {**CURRENT, "beta": beta, "shrink": shrink}
    stage_a.append(evaluate(params, table(beta, shrink), val, "val", "A"))
stage_a.sort(key=score_key, reverse=True)
best_sim = stage_a[0]["params"]

# B. scoring, with the best similarity
stage_b = []
for seeds, related_n, decay in itertools.product(SEEDS_GRID, RELATED_GRID, DECAY_GRID):
    params = {**best_sim, "seeds": seeds, "related": related_n, "decay": decay}
    stage_b.append(evaluate(params, table(best_sim["beta"], best_sim["shrink"]), val, "val", "B"))
stage_b.sort(key=score_key, reverse=True)
best_scoring = {k: stage_b[0]["params"][k] for k in ("seeds", "related", "decay")}

# C. the runners up of A with the best scoring, in case the two interact
stage_c = [stage_b[0]]
for row in stage_a[1:STAGE_C]:
    params = {**row["params"], **best_scoring}
    stage_c.append(evaluate(params, table(params["beta"], params["shrink"]), val, "val", "C"))
stage_c.sort(key=score_key, reverse=True)
best = stage_c[0]["params"]
log(f"best on validation: {best}")

# Test: the current setup against the best, once each, on every test user
test = eval_users("test", 0)
log(f"test users: {len(test):,}")
current_test = evaluate(CURRENT, table(CURRENT["beta"], CURRENT["shrink"]), test, "test", "test")
best_test = evaluate(best, table(best["beta"], best["shrink"]), test, "test", "test")

ids = table(best["beta"], best["shrink"])
np.savez_compressed(f"{OUT}/neighbours_knn_tuned.npz", ids=ids)
artists.to_parquet(f"{OUT}/artists.parquet", index=False)
summary = {"current": CURRENT, "best": best, "test_current": current_test, "test_best": best_test,
           "pool": POOL, "top_stage_a": stage_a[:5], "top_stage_b": stage_b[:5]}
with open(f"{OUT}/summary.json", "w") as out:
    json.dump(summary, out, indent=1)
log("done")
