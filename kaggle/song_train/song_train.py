"""
Songs, step 3b on Kaggle: song recommendation models against the app's current way of suggesting songs.

Data. A user's songs are their all time songs from the statistics dump (kaggle/song_stats) together with their
15 days of daily listens (kaggle/songs), the larger count kept when both have a pair. Songs need MIN_SONG_USERS
listeners, users MIN_USER_SONGS songs, and each user keeps at most CAP of their most played songs. VAL_USERS and
TEST_USERS users are set aside and HIDE_SHARE of each one's songs is hidden.

Two tasks, both scored at 12 (what the app shows) and 20 with recall, hit rate and NDCG:
- all: find any hidden song. Songs the user already has are never suggested.
- discovery: find hidden songs by artists the user has never played, which is what "Senin için" is for. Songs by
  the user's artists are not suggested and users without such a hidden song are left out.

Song models give each song a list of similar songs and are scored the app's way (the user's most played songs are
the seeds, see kaggle/tune): kNN (cosine on log weights, also the asymmetric cosine and shrinkage of kaggle/tune)
and item2vec over listening sessions of the daily dumps.

Baselines:
- popularity: the most listened songs.
- artists' top songs (all only): the most listened songs the user has not played by the user's own artists.
- the app today: the tuned artist kNN (kaggle/tune, B settings) suggests new artists, each with its most listened
  songs: 3 per artist as the app shows them, or 1 per artist.

Outputs in OUT: results.json, summary.json (with the decision rule: a song model must beat the best baseline's
recall@12 by MARGIN on test users), neighbours_song_knn.npz and songs_model.parquet (the songs the model knows).
"""

import glob
import json
import multiprocessing
import os
import subprocess
import sys
import time
import unicodedata

if not os.environ.get("ST_SONGS"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "implicit", "gensim"], check=True)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from implicit.nearest_neighbours import CosineRecommender  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

QUICK = bool(os.environ.get("ST_QUICK"))
MIN_SONG_USERS = 5 if QUICK else 20
MIN_USER_SONGS = 10
CAP = 500
VAL_USERS = 200 if QUICK else 2000
TEST_USERS = 300 if QUICK else 5000
HIDE_SHARE = 0.2
POOL = 60 if QUICK else 200
RELATED_MAX = 100
EXPORT_N = 50
KS = (12, 20)
SEED = 42
MARGIN = 0.2
SESSION_GAP = 30 * 60
MAX_TOKENS = 60_000_000
KNN_GRID = [(0.5, 0.0), (0.5, 100.0), (0.4, 0.0), (0.4, 100.0)]
SCORINGS = [(30, 50, 0.75), (10, 20, 0.5)]
ARTIST_SCORING = (30, 50, 0.75)

OUT = os.environ.get("ST_OUT") or "/kaggle/working"
os.makedirs(OUT, exist_ok=True)


def find(name, env, hint):
    """A file of an input kernel: under ST_<env> locally, else the one under /kaggle/input whose path has hint."""
    if os.environ.get(env):
        return os.path.join(os.environ[env], name)
    paths = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return next(p for p in paths if hint in p)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def fold(text):
    text = (text or "").replace("ı", "i").replace("İ", "i")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


results = []
summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}


def save():
    with open(f"{OUT}/results.json", "w") as out:
        json.dump(results, out, indent=1)
    with open(f"{OUT}/summary.json", "w") as out:
        json.dump(summary, out, indent=1)


# ---------------------------------------------------------------- data
start = time.time()
songs_all = pd.read_parquet(find("songs_all.parquet", "ST_STATS", "song-stats"))
stats = pd.read_parquet(find("user_songs.parquet", "ST_STATS", "song-stats"))
listen_dir = os.path.dirname(find("songs.parquet", "ST_SONGS", "listenbrainz-songs"))
listen_files = sorted(glob.glob(f"{listen_dir}/listens/*.parquet"))
daily = pd.concat([pd.read_parquet(f) for f in listen_files], ignore_index=True)
log(f"read {len(stats):,} statistics pairs and {len(daily):,} daily listens in {time.time() - start:.0f}s")

pair_key = lambda users, songs: (users.astype(np.int64) << 32) | songs.astype(np.int64)  # noqa: E731
daily_keys, daily_counts = np.unique(pair_key(daily["user_id"].to_numpy(), daily["song"].to_numpy()),
                                     return_counts=True)
keys = np.concatenate([pair_key(stats["user_id"].to_numpy(), stats["song"].to_numpy()), daily_keys])
counts = np.concatenate([stats["listens"].to_numpy().astype(np.int64), daily_counts])
order = np.argsort(keys, kind="stable")
keys, counts = keys[order], counts[order]
first = np.concatenate([[True], keys[1:] != keys[:-1]])
starts = np.flatnonzero(first)
pairs = pd.DataFrame({"user_id": (keys[starts] >> 32).astype(np.int64), "song": (keys[starts] & 0xFFFFFFFF),
                      "listens": np.maximum.reduceat(counts, starts)})
del keys, counts, order, first, starts, daily_keys, daily_counts, stats
summary["pairs_before_filter"] = len(pairs)
summary["users_before_filter"] = int(pairs.user_id.nunique())

song_users = pairs.groupby("song").size()
pairs = pairs[pairs.song.isin(song_users.index[song_users >= MIN_SONG_USERS])]
user_songs = pairs.groupby("user_id").size()
pairs = pairs[pairs.user_id.isin(user_songs.index[user_songs >= MIN_USER_SONGS])]
pairs = pairs.sort_values(["user_id", "listens", "song"], ascending=[True, False, True], kind="stable")
pairs = pairs[pairs.groupby("user_id").cumcount() < CAP]

song_ids, s_of = np.unique(pairs["song"].to_numpy(), return_inverse=True)
user_ids, u_of = np.unique(pairs["user_id"].to_numpy(), return_inverse=True)
n_songs, n_users = len(song_ids), len(user_ids)
meta = songs_all.set_index("song").loc[song_ids].reset_index()
artist_names = meta["artist"].map(fold)
artist_code, artist_keys = pd.factorize(artist_names)
artist_code = artist_code.astype(np.int32)
summary["data"] = {"users": n_users, "songs": n_songs, "pairs": len(pairs), "artists": len(artist_keys),
                   "seconds": round(time.time() - start)}
log(json.dumps(summary["data"]))

# Split and hide
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_users)
split = np.zeros(n_users, dtype=np.int8)  # 0 train, 1 val, 2 test
split[perm[:VAL_USERS]] = 1
split[perm[VAL_USERS:VAL_USERS + TEST_USERS]] = 2
listens_v = pairs["listens"].to_numpy().astype(np.float32)
hidden_mask = np.zeros(len(pairs), dtype=bool)
user_rows = np.split(np.arange(len(pairs)), np.flatnonzero(np.diff(u_of)) + 1)
for u, rows in zip(np.unique(u_of), user_rows):
    if split[u]:
        n_hide = max(1, int(round(len(rows) * HIDE_SHARE)))
        hidden_mask[rng.choice(rows, n_hide, replace=False)] = True
train = sp.csr_matrix((listens_v[~hidden_mask], (u_of[~hidden_mask], s_of[~hidden_mask])),
                      shape=(n_users, n_songs))
hidden_by_user = pd.Series(s_of[hidden_mask]).groupby(u_of[hidden_mask]).apply(np.asarray)
popularity = np.asarray((train > 0).sum(axis=0)).ravel()
weighted = train.copy()
weighted.data = np.log1p(weighted.data)
norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=0)).ravel()).astype(np.float32)
del pairs

# Each artist's songs, most listened first, for the baselines
by_artist = pd.Series(np.arange(n_songs)).groupby(artist_code).apply(
    lambda s: s.to_numpy()[np.argsort(-popularity[s.to_numpy()], kind="stable")])
popular_order = np.argsort(-popularity, kind="stable")

# The tuned artist model and its artists matched to the song artists by folded name
artist_table = pd.read_parquet(find("artists.parquet", "ST_TUNE", "listenbrainz-tune"))
artist_neighbours = np.load(find("neighbours_knn_tuned.npz", "ST_TUNE", "listenbrainz-tune"))["ids"]
model_of_name = {}
for a, name, users in sorted(zip(artist_table["a"], artist_table["name"], artist_table["users"]), key=lambda x: x[2]):
    model_of_name[fold(name)] = int(a)
model_of_code = np.array([model_of_name.get(k, -1) for k in artist_keys], dtype=np.int64)
code_of_model = {int(m): c for c, m in enumerate(model_of_code) if m >= 0}
summary["artists_matched_to_artist_model"] = round(float(np.mean(model_of_code >= 0)), 4)
log(f"song artists in the artist model: {summary['artists_matched_to_artist_model']:.1%}")
save()


def eval_users(which):
    """Per user: seed songs, their weights, known songs, known artists, artist seeds and weights, hidden songs."""
    out = []
    for u in np.flatnonzero(split == which):
        row = train.getrow(u)
        if row.nnz == 0 or u not in hidden_by_user.index:
            continue
        order = np.argsort(-row.data)
        known = frozenset(row.indices.tolist())
        codes = artist_code[row.indices]
        artist_weight = pd.Series(row.data).groupby(codes).sum().sort_values(ascending=False, kind="stable")
        out.append({"seeds": row.indices[order][:max(s for s, _, _ in SCORINGS)], "weights": row.data[order],
                    "known": known, "known_artists": frozenset(codes.tolist()),
                    "artist_seeds": artist_weight.index.to_numpy(), "artist_weights": artist_weight.to_numpy(),
                    "hidden": hidden_by_user[u]})
    return out


def targets(user, mode):
    hidden = user["hidden"]
    if mode == "discovery":
        hidden = hidden[~np.isin(artist_code[hidden], list(user["known_artists"]))]
    return set(hidden.tolist())


def allowed(user, mode):
    known, known_artists = user["known"], user["known_artists"]
    if mode == "discovery":
        return lambda s: s not in known and artist_code[s] not in known_artists
    return lambda s: s not in known


def app_rank(seeds, weights, related, ok, related_n, decay, k, tie):
    """The app's score(): shares by place, then greedy picks with decay. Unwanted candidates are dropped first."""
    shares = {}
    for s, w in zip(seeds, weights):
        candidates = [int(a) for a in related[s] if a >= 0 and ok(int(a))][:related_n]
        for place, a in enumerate(candidates):
            entry = shares.setdefault(a, {})
            entry[int(s)] = entry.get(int(s), 0.0) + float(w) * (1 - place / max(len(candidates), related_n))
    used = {int(s): 0 for s in seeds}
    ranked = []
    while shares and len(ranked) < k:
        best = max(shares, key=lambda a: (sum(v * decay ** used[s] for s, v in shares[a].items()), tie[a]))
        for s in shares.pop(best):
            used[s] += 1
        ranked.append(best)
    return ranked


# Rankers: user, mode -> songs, best first. Workers are forked and read these globals
_job = {}


def song_model_ranker(user, mode):
    related, (n_seeds, related_n, decay) = _job["related"], _job["scoring"]
    return app_rank(user["seeds"][:n_seeds], user["weights"][:n_seeds], related, allowed(user, mode), related_n,
                    decay, max(KS), popularity)


def popularity_ranker(user, mode):
    ok = allowed(user, mode)
    out = []
    for s in popular_order:
        if ok(int(s)):
            out.append(int(s))
            if len(out) == max(KS):
                break
    return out


def own_artists_ranker(user, mode):
    """The most listened songs the user has not played, by the user's own artists in turn."""
    lists = [iter(by_artist[c]) for c in user["artist_seeds"][:30]]
    out = []
    while lists and len(out) < max(KS):
        for it in list(lists):
            s = next((int(x) for x in it if int(x) not in user["known"]), None)
            if s is None:
                lists.remove(it)
            else:
                out.append(s)
        out = out[:max(KS)]
    return out


def app_today_ranker(user, mode):
    """The tuned artist kNN suggests artists the user has not played, each with its most listened songs."""
    per_artist = _job["per_artist"]
    n_seeds, related_n, decay = ARTIST_SCORING
    seeds = [(int(model_of_code[c]), w) for c, w in zip(user["artist_seeds"], user["artist_weights"])
             if model_of_code[c] >= 0][:n_seeds]
    known_models = {int(model_of_code[c]) for c in user["known_artists"] if model_of_code[c] >= 0}
    artists = app_rank([s for s, _ in seeds], [w for _, w in seeds], artist_neighbours,
                       lambda a: a not in known_models and a in code_of_model, related_n, decay, max(KS),
                       np.zeros(len(artist_neighbours)))
    out = []
    for a in artists:
        out.extend(int(s) for s in by_artist[code_of_model[a]][:per_artist] if int(s) not in user["known"])
        if len(out) >= max(KS):
            break
    return out[:max(KS)]


def _rank(user):
    return _job["ranker"](user, _job["mode"])


def metrics(rankings, group, mode):
    out = {}
    rows = [(r, targets(u, mode)) for r, u in zip(rankings, group)]
    rows = [(r, t) for r, t in rows if t]
    for k in KS:
        recall, hit, ndcg = [], [], []
        for ranked, wanted in rows:
            gains = [1.0 if s in wanted else 0.0 for s in ranked[:k]]
            found = sum(gains)
            recall.append(found / len(wanted))
            hit.append(1.0 if found else 0.0)
            ideal = sum(1 / np.log2(i + 2) for i in range(min(len(wanted), k)))
            ndcg.append(sum(g / np.log2(i + 2) for i, g in enumerate(gains)) / ideal)
        out[f"recall@{k}"] = round(float(np.mean(recall)), 4)
        out[f"hit@{k}"] = round(float(np.mean(hit)), 4)
        out[f"ndcg@{k}"] = round(float(np.mean(ndcg)), 4)
    out["users"] = len(rows)
    return out


def evaluate(name, params, ranker, group, split_name, mode, **job):
    start = time.time()
    _job.update(ranker=ranker, mode=mode, **job)
    if "fork" in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context("fork").Pool(os.cpu_count()) as pool:
            rankings = pool.map(_rank, group, chunksize=32)
    else:
        rankings = [_rank(u) for u in group]
    row = {"model": name, "params": params, "split": split_name, "mode": mode, **metrics(rankings, group, mode),
           "eval_seconds": round(time.time() - start)}
    results.append(row)
    save()
    log(json.dumps(row))
    return row


val, test = eval_users(1), eval_users(2)
log(f"validation users {len(val):,}, test users {len(test):,}")

# ---------------------------------------------------------------- kNN
start = time.time()
cosine = CosineRecommender(K=POOL)
cosine.fit(weighted, show_progress=False)
pool_ids = np.full((n_songs, POOL), -1, dtype=np.int32)
pool_dot = np.zeros((n_songs, POOL), dtype=np.float32)
for lo in range(0, n_songs, 20000):
    chunk = np.arange(lo, min(lo + 20000, n_songs))
    ids, scores = cosine.similar_items(chunk, N=POOL + 1)
    for i, row, values in zip(chunk, ids, scores):
        mask = (row != i) & (row >= 0) & (values > 0)
        kept = row[mask][:POOL]
        pool_ids[i, :len(kept)] = kept
        pool_dot[i, :len(kept)] = values[mask][:POOL] * norms[i] * norms[kept]
del cosine
summary["knn_pool_seconds"] = round(time.time() - start)
log(f"pool of {POOL} fitted in {summary['knn_pool_seconds']}s")


def neighbours(beta, shrink, n=RELATED_MAX):
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


sample = np.random.default_rng(SEED).choice(n_songs, min(2000, n_songs), replace=False)
ours, theirs = neighbours(0.5, 0.0, 20)[sample], pool_ids[sample, :20]
same = float(np.mean([set(a[a >= 0].tolist()) == set(b[b >= 0].tolist()) for a, b in zip(ours, theirs)]))
log(f"check: beta 0.5 gives implicit's cosine neighbours for {same:.1%} of the sample")
assert same > 0.98, "the rescoring does not reproduce cosine"

tables = {}
best = {}
for beta, shrink in KNN_GRID:
    tables[(beta, shrink)] = neighbours(beta, shrink)
    for scoring in SCORINGS:
        for mode in ("all", "discovery"):
            params = {"beta": beta, "shrink": shrink, "scoring": scoring}
            row = evaluate("song knn", params, song_model_ranker, val, "val", mode,
                           related=tables[(beta, shrink)], scoring=scoring)
            if row["ndcg@12"] > best.get(mode, ({}, -1))[1]:
                best[mode] = (params, row["ndcg@12"])
summary["best_knn"] = {mode: params for mode, (params, _) in best.items()}
save()

# ---------------------------------------------------------------- item2vec
start = time.time()
song_index = pd.Series(np.arange(n_songs), index=song_ids)
train_users = set(user_ids[split == 0].tolist())
daily = daily[daily["user_id"].isin(train_users) & daily["song"].isin(song_ids)]
daily = daily.sort_values(["user_id", "ts"], kind="stable")
items = song_index.loc[daily["song"].to_numpy()].to_numpy()
user_col, ts = daily["user_id"].to_numpy(), daily["ts"].to_numpy()
breaks = np.flatnonzero((user_col[1:] != user_col[:-1]) | (ts[1:] - ts[:-1] > SESSION_GAP)) + 1
sessions = []
for chunk in np.split(items, breaks):
    chunk = chunk[np.concatenate([[True], chunk[1:] != chunk[:-1]])]
    if len(chunk) >= 2:
        sessions.append(chunk)
del daily, items, user_col, ts
total = sum(len(s) for s in sessions)
if total > MAX_TOKENS:
    keep = np.random.default_rng(SEED).random(len(sessions)) < MAX_TOKENS / total
    sessions = [s for s, k in zip(sessions, keep) if k]
summary["item2vec_sessions"] = {"sessions": len(sessions), "tokens": int(sum(len(s) for s in sessions)),
                                "tokens_before_sampling": int(total)}
log(json.dumps(summary["item2vec_sessions"]))
model = Word2Vec([s.astype(str).tolist() for s in sessions], vector_size=64, window=10, min_count=5, sg=1,
                 negative=10, ns_exponent=-0.5, sample=1e-4, epochs=5, workers=os.cpu_count(), seed=SEED)
vectors = np.zeros((n_songs, 64), dtype=np.float32)
for token, index in model.wv.key_to_index.items():
    vectors[int(token)] = model.wv.vectors[index]
del model, sessions
summary["item2vec_seconds"] = round(time.time() - start)
log(f"item2vec trained in {summary['item2vec_seconds']}s, {int((np.abs(vectors).sum(1) > 0).sum()):,} songs known")


def vector_neighbours(vectors, ids, n=RELATED_MAX, batch=128):
    """The n closest songs of the given songs by cosine, -1 for songs without a vector, as a full table."""
    norms_v = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = (vectors / np.maximum(norms_v, 1e-9)).astype(np.float32)
    has = norms_v.ravel() > 0
    out = np.full((n_songs, n), -1, dtype=np.int32)
    ids = np.asarray([i for i in ids if has[i]])
    for lo in range(0, len(ids), batch):
        chunk = ids[lo:lo + batch]
        with threadpool_limits(os.cpu_count()):
            scores = unit[chunk] @ unit.T
        scores[:, ~has] = -np.inf
        scores[np.arange(len(chunk)), chunk] = -np.inf
        top = np.argpartition(-scores, n, axis=1)[:, :n]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        out[chunk] = np.take_along_axis(top, order, axis=1)
    return out


seed_ids = np.unique(np.concatenate([u["seeds"] for u in val + test]))
item2vec_table = vector_neighbours(vectors, seed_ids)
del vectors
for mode in ("all", "discovery"):
    evaluate("song item2vec", {"scoring": best[mode][0]["scoring"]}, song_model_ranker, val, "val", mode,
             related=item2vec_table, scoring=best[mode][0]["scoring"])

# ---------------------------------------------------------------- baselines on validation
for mode in ("all", "discovery"):
    evaluate("popularity", {}, popularity_ranker, val, "val", mode)
    evaluate("app today: artist knn + top songs", {"per_artist": 3}, app_today_ranker, val, "val", mode,
             per_artist=3)
    evaluate("app today: artist knn + top songs", {"per_artist": 1}, app_today_ranker, val, "val", mode,
             per_artist=1)
evaluate("own artists' top songs", {}, own_artists_ranker, val, "val", "all")

# ---------------------------------------------------------------- test, once each
summary["test"] = {}
for mode in ("all", "discovery"):
    params = best[mode][0]
    rows = [
        evaluate("song knn", params, song_model_ranker, test, "test", mode,
                 related=tables[(params["beta"], params["shrink"])], scoring=params["scoring"]),
        evaluate("song item2vec", {"scoring": params["scoring"]}, song_model_ranker, test, "test", mode,
                 related=item2vec_table, scoring=params["scoring"]),
        evaluate("popularity", {}, popularity_ranker, test, "test", mode),
        evaluate("app today: artist knn + top songs", {"per_artist": 3}, app_today_ranker, test, "test", mode,
                 per_artist=3),
        evaluate("app today: artist knn + top songs", {"per_artist": 1}, app_today_ranker, test, "test", mode,
                 per_artist=1),
    ]
    if mode == "all":
        rows.append(evaluate("own artists' top songs", {}, own_artists_ranker, test, "test", mode))
    models = [r for r in rows if r["model"].startswith("song ")]
    baselines = [r for r in rows if not r["model"].startswith("song ")]
    best_model = max(models, key=lambda r: r["recall@12"])
    best_baseline = max(baselines, key=lambda r: r["recall@12"])
    gain = best_model["recall@12"] / best_baseline["recall@12"] - 1 if best_baseline["recall@12"] else float("inf")
    summary["test"][mode] = {"rows": rows, "best_model": best_model["model"], "best_baseline": best_baseline["model"],
                             "gain_recall@12": round(gain, 4), "passes": gain >= MARGIN}
    save()

# ---------------------------------------------------------------- export
params = best["all"][0]
np.savez_compressed(f"{OUT}/neighbours_song_knn.npz",
                    ids=tables[(params["beta"], params["shrink"])][:, :EXPORT_N].astype(np.int32))
meta.assign(s=np.arange(n_songs, dtype=np.int32), users=popularity.astype(np.int32))[
    ["s", "song", "key", "artist", "title", "mbid", "users"]].to_parquet(f"{OUT}/songs_model.parquet", index=False)
summary["exported"] = {"knn": params, "songs": n_songs}
summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("done")
