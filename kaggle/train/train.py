"""
Part 3 on Kaggle: the artist similarity models, trained on kaggle/build's output and scored the way the app uses them.

Scoring copies the app (playlist_tracker/music/recommend.py): a user's SEEDS most played known artists are the
seeds, each seed's RELATED most similar artists are read, an artist adds seed weight × (1 − place / RELATED) for
each seed listing it, artists are picked one at a time and every pick multiplies what its seeds add to the rest by
SEED_DECAY. Known artists are left out. The hidden artists of a user are what should come up.

Metrics at 12 (what the app shows) and 20: recall (hidden found / hidden), hit rate (at least one found) and NDCG.
Hyperparameters are chosen on TUNE_USERS validation users, then the best setting of each family is scored once
on all test users and its neighbour lists are exported.

Outputs in OUT: results.json (every setting tried), neighbours_<model>.npz (for every artist its EXPORT_N most
similar artists and their scores, Part 4 reads these), als_factors.npy and item2vec_vectors.npy, summary.json.
"""

import glob
import json
import os
import subprocess
import sys
import time

if not os.environ.get("TRAIN_INPUT"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "implicit", "gensim"], check=True)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from implicit.als import AlternatingLeastSquares  # noqa: E402
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, bm25_weight  # noqa: E402

SEEDS = 10
RELATED = 20
SEED_DECAY = 0.5
KS = (12, 20)
TUNE_USERS = int(os.environ.get("TUNE_USERS", 2000))
EXPORT_N = 50
SEED = 42

# TRAIN_INPUT, TRAIN_OUT and TRAIN_QUICK let the script run locally on a small sample
INPUT = os.environ.get("TRAIN_INPUT") or os.path.dirname(
    glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0])
OUT = os.environ.get("TRAIN_OUT") or "/kaggle/working"
QUICK = bool(os.environ.get("TRAIN_QUICK"))
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
log(f"{n_users:,} users, {n_artists:,} artists, {listens.nnz:,} training pairs")


def weighted(kind):
    """The user × artist matrix with listen counts weighted: binary, log(1 + listens) or BM25."""
    if kind == "bin":
        out = listens.copy()
        out.data[:] = 1.0
        return out
    if kind == "log":
        out = listens.copy()
        out.data = np.log1p(out.data)
        return out
    return bm25_weight(listens, K1=100, B=0.8).tocsr().astype(np.float32)


def eval_users(split, limit):
    """[(seed ids, seed weights, known ids, hidden ids)] for the users of a split, at most limit of them."""
    ids = users.loc[users["split"] == split, "u"].to_numpy()
    if limit and len(ids) > limit:
        ids = np.random.default_rng(SEED).choice(ids, limit, replace=False)
    hidden = holdout[holdout["split"] == split].groupby("u")["a"].apply(np.asarray)
    out = []
    for u in ids:
        row = listens.getrow(u)
        if row.nnz == 0 or u not in hidden.index:
            continue
        order = np.argsort(-row.data)[:SEEDS]
        out.append((row.indices[order], row.data[order], row.indices, hidden[u]))
    return out


def app_rank(seeds, weights, related, known, k):
    """The app's score(): shares by place, then greedy picks with SEED_DECAY."""
    shares = {}
    for s, w in zip(seeds, weights):
        candidates = related.get(int(s), ())
        for place, a in enumerate(candidates):
            if a in known:
                continue
            entry = shares.setdefault(a, {})
            entry[s] = entry.get(s, 0.0) + w * (1 - place / max(len(candidates), RELATED))
    used = {int(s): 0 for s in seeds}
    ranked = []
    while shares and len(ranked) < k:
        best = max(shares, key=lambda a: (sum(v * SEED_DECAY ** used[int(s)] for s, v in shares[a].items()),
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


def evaluate(name, params, neighbours_of, group, split):
    """Scores a model given by neighbours_of(seed ids) -> {seed: [RELATED artists, best first]}."""
    start = time.time()
    seeds = np.unique(np.concatenate([g[0] for g in group]))
    related = neighbours_of(seeds)
    rankings = [app_rank(s, w, related, set(known.tolist()), max(KS)) for s, w, known, _ in group]
    found = metrics(rankings, group)
    coverage = float(np.mean([len(related.get(int(s), ())) > 0 for s in seeds]))
    row = {"model": name, "params": params, "split": split, "users": len(group), **found,
           "seed_coverage": round(coverage, 4), "eval_seconds": round(time.time() - start)}
    results.append(row)
    save_results()
    log(json.dumps(row))
    return row


def popular_neighbours(seeds):
    """Popularity as a model: every artist's related list is the most played artists."""
    top = np.argsort(-popularity)[:RELATED + SEEDS].tolist()
    return {int(s): [a for a in top if a != s][:RELATED] for s in seeds}


def knn_neighbours(model):
    def neighbours_of(seeds):
        ids, scores = model.similar_items(np.asarray(seeds), N=RELATED + 1)
        return {int(s): [int(a) for a, v in zip(row, values) if a != s and a >= 0 and v > 0][:RELATED]
                for s, row, values in zip(seeds, ids, scores)}
    return neighbours_of


def cosine_top(vectors, ids, n, batch=512):
    """For each id, the n artists with the closest vectors (cosine), itself left out, and their scores."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = (vectors / np.maximum(norms, 1e-9)).astype(np.float32)
    known = norms.ravel() > 0
    out_ids = np.full((len(ids), n), -1, dtype=np.int32)
    out_scores = np.zeros((len(ids), n), dtype=np.float32)
    for start in range(0, len(ids), batch):
        chunk = np.asarray(ids[start:start + batch])
        # implicit wants BLAS on one thread for its own work, but this product needs every core
        with threadpool_limits(os.cpu_count()):
            scores = unit[chunk] @ unit.T
        scores[:, ~known] = -np.inf
        scores[np.arange(len(chunk)), chunk] = -np.inf
        top = np.argpartition(-scores, n, axis=1)[:, :n]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        out_ids[start:start + len(chunk)] = np.take_along_axis(top, order, axis=1)
        out_scores[start:start + len(chunk)] = np.take_along_axis(top_scores, order, axis=1)
        out_ids[start:start + len(chunk)][~known[chunk]] = -1
    return out_ids, out_scores


def vector_neighbours(vectors):
    def neighbours_of(seeds):
        ids, _ = cosine_top(vectors, seeds, RELATED)
        return {int(s): [int(a) for a in row if a >= 0] for s, row in zip(seeds, ids)}
    return neighbours_of


def fit_knn(kind, weighting, k=100):
    model = (CosineRecommender(K=k) if kind == "cosine" else BM25Recommender(K=k, K1=1.2, B=0.75))
    model.fit(weighted(weighting), show_progress=False)
    return model


def fit_als(factors, regularization, alpha, weighting, iterations=15):
    model = AlternatingLeastSquares(factors=factors, regularization=regularization, alpha=alpha,
                                    iterations=iterations, random_state=SEED)
    model.fit(weighted(weighting), show_progress=False)
    factors_out = model.item_factors
    return np.asarray(factors_out.to_numpy() if hasattr(factors_out, "to_numpy") else factors_out)


def fit_item2vec(window, ns_exponent, size=64, epochs=10):
    items = np.load(f"{INPUT}/sessions_items.npy")
    offsets = np.load(f"{INPUT}/sessions_offsets.npy")
    tokens = items.astype(str)
    sentences = [tokens[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    model = Word2Vec(sentences, vector_size=size, window=window, min_count=5, sg=1, negative=10,
                     ns_exponent=ns_exponent, sample=1e-4, epochs=epochs, workers=os.cpu_count(), seed=SEED)
    vectors = np.zeros((n_artists, size), dtype=np.float32)
    for token, index in model.wv.key_to_index.items():
        vectors[int(token)] = model.wv.vectors[index]
    return vectors


# Settings tried on the validation users. QUICK keeps one of each for the local test
GRID = {
    "knn": [("cosine", "bin"), ("cosine", "log"), ("bm25", "log"), ("bm25", "bin")],
    "als": [(64, 0.05, 1.0, "bm25"), (128, 0.05, 1.0, "bm25"), (128, 0.5, 1.0, "bm25"),
            (64, 0.05, 10.0, "log"), (128, 0.05, 10.0, "log"), (256, 0.05, 1.0, "bm25")],
    "item2vec": [(5, 0.75), (5, -0.5), (20, -0.5)],
}
if QUICK:
    # The local sample has only a few hundred artists, where 64 factors overfit
    GRID = {"knn": GRID["knn"][:1], "als": [(8, 0.05, 1.0, "log")], "item2vec": GRID["item2vec"][:1]}

val = eval_users("val", TUNE_USERS)
log(f"validation users: {len(val):,}")
evaluate("popularity", {}, popular_neighbours, val, "val")

best = {}
for kind, weighting in GRID["knn"]:
    start = time.time()
    model = fit_knn(kind, weighting)
    log(f"knn {kind}/{weighting} fitted in {time.time() - start:.0f}s")
    row = evaluate("knn", {"kind": kind, "weighting": weighting}, knn_neighbours(model), val, "val")
    if row["ndcg@20"] > best.get("knn", ({}, -1))[1]:
        best["knn"] = ({"kind": kind, "weighting": weighting}, row["ndcg@20"])

for factors, regularization, alpha, weighting in GRID["als"]:
    start = time.time()
    vectors = fit_als(factors, regularization, alpha, weighting)
    params = {"factors": factors, "regularization": regularization, "alpha": alpha, "weighting": weighting}
    log(f"als {params} fitted in {time.time() - start:.0f}s")
    row = evaluate("als", params, vector_neighbours(vectors), val, "val")
    if row["ndcg@20"] > best.get("als", ({}, -1))[1]:
        best["als"] = (params, row["ndcg@20"])

for window, ns_exponent in GRID["item2vec"]:
    start = time.time()
    vectors = fit_item2vec(window, ns_exponent)
    params = {"window": window, "ns_exponent": ns_exponent}
    log(f"item2vec {params} fitted in {time.time() - start:.0f}s")
    row = evaluate("item2vec", params, vector_neighbours(vectors), val, "val")
    if row["ndcg@20"] > best.get("item2vec", ({}, -1))[1]:
        best["item2vec"] = (params, row["ndcg@20"])

# Best of each family: refit once, score on every test user, export neighbour lists for every artist
test = eval_users("test", 0)
log(f"test users: {len(test):,}")
evaluate("popularity", {}, popular_neighbours, test, "test")
all_ids = np.arange(n_artists)
summary = {"best_on_validation": {family: params for family, (params, _) in best.items()}}

params = best["knn"][0]
model = fit_knn(params["kind"], params["weighting"])
evaluate("knn", params, knn_neighbours(model), test, "test")
found_ids, found_scores = model.similar_items(all_ids, N=EXPORT_N + 1)
# Rare artists may have fewer neighbours than EXPORT_N: their rows end in -1
ids = np.full((n_artists, EXPORT_N), -1, dtype=np.int32)
scores = np.zeros((n_artists, EXPORT_N), dtype=np.float16)
for i, (row, values) in enumerate(zip(found_ids, found_scores)):
    mask = (row != i) & (row >= 0) & (values > 0)
    kept = row[mask][:EXPORT_N]
    ids[i, :len(kept)] = kept
    scores[i, :len(kept)] = values[mask][:EXPORT_N]
np.savez_compressed(f"{OUT}/neighbours_knn.npz", ids=ids, scores=scores)
del model

params = best["als"][0]
vectors = fit_als(params["factors"], params["regularization"], params["alpha"], params["weighting"])
evaluate("als", params, vector_neighbours(vectors), test, "test")
np.save(f"{OUT}/als_factors.npy", vectors.astype(np.float16))
ids, scores = cosine_top(vectors, all_ids, EXPORT_N)
np.savez_compressed(f"{OUT}/neighbours_als.npz", ids=ids, scores=scores.astype(np.float16))

params = best["item2vec"][0]
vectors = fit_item2vec(params["window"], params["ns_exponent"])
evaluate("item2vec", params, vector_neighbours(vectors), test, "test")
np.save(f"{OUT}/item2vec_vectors.npy", vectors.astype(np.float16))
ids, scores = cosine_top(vectors, all_ids, EXPORT_N)
np.savez_compressed(f"{OUT}/neighbours_item2vec.npz", ids=ids, scores=scores.astype(np.float16))

artists.to_parquet(f"{OUT}/artists.parquet", index=False)
summary["test"] = [r for r in results if r["split"] == "test"]
with open(f"{OUT}/summary.json", "w") as out:
    json.dump(summary, out, indent=1)
log("done")
