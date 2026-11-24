"""
Songs, step 4 on Kaggle: can the song recommendations get better? Everything is measured on one data set and one
set of test users, against the song kNN of kaggle/song_train (the reference).

More data: songs need MIN_SONG_USERS listeners (20 before) and users keep up to CAP songs (500 before). A kNN on a
copy of the training matrix cut to 500 songs per user shows how much the cap alone brings.

Candidates, each a list of similar songs per song, scored the app's way (SEEDS, RELATED, SEED_DECAY of the B
settings):
- knn: cosine with shrinkage 100 on log listens. "reference" reads 100 neighbours as before, "big pool" 400, which
  matters for discovery, where the many neighbours by the user's own artists are dropped.
- RP3beta: two step random walk user → song → user → song on the binary matrix, the walk through a user damped by
  their song count^ALPHA and the arrival song by its popularity^BETA. Computed as a co-occurrence pool and scored
  again, as the asymmetric cosine of kaggle/tune.
- EASE: the closed form linear model on the EASE_SONGS most listened songs.
- item2vec over the daily dumps' listening sessions.
Then two combinations:
- artist then song: the tuned artist kNN picks new artists as the app does today, and the song kNN picks each
  artist's song (instead of its most listened one).
- reciprocal rank fusion (RRF) of the candidates' ranked lists.

Two tasks as in kaggle/song_train: all (any hidden song) and discovery (hidden songs by artists the user has never
played). Everything is chosen on VAL_USERS by NDCG@12 and scored once on TEST_USERS.
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
from implicit.nearest_neighbours import CosineRecommender, ItemItemRecommender  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

QUICK = bool(os.environ.get("ST_QUICK"))
MIN_SONG_USERS = 5 if QUICK else 10
MIN_USER_SONGS = 10
CAP = 1000
OLD_CAP = 500
VAL_USERS = 200 if QUICK else 2000
TEST_USERS = 300 if QUICK else 5000
HIDE_SHARE = 0.2
POOL = 60 if QUICK else 400
REFERENCE_N = 30 if QUICK else 100
SMALL = 40 if QUICK else 200
EASE_SONGS = 200 if QUICK else 30000
EASE_LAMBDA = 400.0
RP3_ALPHAS = (0.5, 1.0)
RP3_BETAS = (0.0, 0.3, 0.6)
SCORING = (30, 50, 0.75)
K_OUT = 50
KS = (12, 20)
SEED = 42
RRF_K = 60
SESSION_GAP = 30 * 60
MAX_TOKENS = 60_000_000
EXPORT_N = 50

OUT = os.environ.get("ST_OUT") or "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
MODES = ("all", "discovery")


def find(name, env, hint):
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
summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": {}}


def save():
    with open(f"{OUT}/results.json", "w") as out:
        json.dump(results, out, indent=1)
    with open(f"{OUT}/summary.json", "w") as out:
        json.dump(summary, out, indent=1)


def timed(name, since):
    summary["seconds"][name] = round(time.time() - since)
    log(f"{name}: {summary['seconds'][name]}s")


# ---------------------------------------------------------------- data (as kaggle/song_train, more of it)
start = time.time()
songs_all = pd.read_parquet(find("songs_all.parquet", "ST_STATS", "song-stats"))
stats = pd.read_parquet(find("user_songs.parquet", "ST_STATS", "song-stats"))
listen_dir = os.path.dirname(find("songs.parquet", "ST_SONGS", "listenbrainz-songs"))
daily = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{listen_dir}/listens/*.parquet"))],
                  ignore_index=True)

pair_key = lambda users, songs: (users.astype(np.int64) << 32) | songs.astype(np.int64)  # noqa: E731
daily_keys, daily_counts = np.unique(pair_key(daily["user_id"].to_numpy(), daily["song"].to_numpy()),
                                     return_counts=True)
keys = np.concatenate([pair_key(stats["user_id"].to_numpy(), stats["song"].to_numpy()), daily_keys])
counts = np.concatenate([stats["listens"].to_numpy().astype(np.int64), daily_counts])
order = np.argsort(keys, kind="stable")
keys, counts = keys[order], counts[order]
starts = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]]))
pairs = pd.DataFrame({"user_id": (keys[starts] >> 32).astype(np.int64), "song": (keys[starts] & 0xFFFFFFFF),
                      "listens": np.maximum.reduceat(counts, starts)})
del keys, counts, order, starts, daily_keys, daily_counts, stats

song_users = pairs.groupby("song").size()
pairs = pairs[pairs.song.isin(song_users.index[song_users >= MIN_SONG_USERS])]
user_songs = pairs.groupby("user_id").size()
pairs = pairs[pairs.user_id.isin(user_songs.index[user_songs >= MIN_USER_SONGS])]
pairs = pairs.sort_values(["user_id", "listens", "song"], ascending=[True, False, True], kind="stable")
rank_in_user = pairs.groupby("user_id").cumcount().to_numpy()
pairs = pairs[rank_in_user < CAP]
rank_in_user = rank_in_user[rank_in_user < CAP]

song_ids, s_of = np.unique(pairs["song"].to_numpy(), return_inverse=True)
user_ids, u_of = np.unique(pairs["user_id"].to_numpy(), return_inverse=True)
n_songs, n_users = len(song_ids), len(user_ids)
meta = songs_all.set_index("song").loc[song_ids].reset_index()
artist_code, artist_keys = pd.factorize(meta["artist"].map(fold))
artist_code = artist_code.astype(np.int32)
summary["data"] = {"users": n_users, "songs": n_songs, "pairs": len(pairs), "artists": len(artist_keys),
                   "min_song_users": MIN_SONG_USERS, "cap": CAP}
log(json.dumps(summary["data"]))

rng = np.random.default_rng(SEED)
perm = rng.permutation(n_users)
split = np.zeros(n_users, dtype=np.int8)
split[perm[:VAL_USERS]] = 1
split[perm[VAL_USERS:VAL_USERS + TEST_USERS]] = 2
listens_v = pairs["listens"].to_numpy().astype(np.float32)
hidden_mask = np.zeros(len(pairs), dtype=bool)
for u, rows in zip(np.unique(u_of), np.split(np.arange(len(pairs)), np.flatnonzero(np.diff(u_of)) + 1)):
    if split[u]:
        hidden_mask[rng.choice(rows, max(1, int(round(len(rows) * HIDE_SHARE))), replace=False)] = True
keep = ~hidden_mask
train = sp.csr_matrix((listens_v[keep], (u_of[keep], s_of[keep])), shape=(n_users, n_songs))
# The same users with at most OLD_CAP songs each, for the effect of the larger cap
old_keep = keep & (rank_in_user < OLD_CAP)
train_old_cap = sp.csr_matrix((listens_v[old_keep], (u_of[old_keep], s_of[old_keep])), shape=(n_users, n_songs))
hidden_by_user = pd.Series(s_of[hidden_mask]).groupby(u_of[hidden_mask]).apply(np.asarray)
popularity = np.asarray((train > 0).sum(axis=0)).ravel()
del pairs, rank_in_user

by_artist = pd.Series(np.arange(n_songs)).groupby(artist_code).apply(
    lambda s: s.to_numpy()[np.argsort(-popularity[s.to_numpy()], kind="stable")])
popular_order = np.argsort(-popularity, kind="stable")

artist_table = pd.read_parquet(find("artists.parquet", "ST_TUNE", "listenbrainz-tune"))
artist_neighbours = np.load(find("neighbours_knn_tuned.npz", "ST_TUNE", "listenbrainz-tune"))["ids"]
model_of_name = {}
for a, name, users in sorted(zip(artist_table["a"], artist_table["name"], artist_table["users"]), key=lambda x: x[2]):
    model_of_name[fold(name)] = int(a)
model_of_code = np.array([model_of_name.get(k, -1) for k in artist_keys], dtype=np.int64)
code_of_model = {int(m): c for c, m in enumerate(model_of_code) if m >= 0}
timed("data", start)
save()


# ---------------------------------------------------------------- evaluation
def eval_users(which):
    out = []
    for u in np.flatnonzero(split == which):
        row = train.getrow(u)
        if row.nnz == 0 or u not in hidden_by_user.index:
            continue
        order = np.argsort(-row.data)
        codes = artist_code[row.indices]
        artist_weight = pd.Series(row.data).groupby(codes).sum().sort_values(ascending=False, kind="stable")
        out.append({"seeds": row.indices[order][:SCORING[0]], "weights": row.data[order][:SCORING[0]],
                    "known": frozenset(row.indices.tolist()), "known_artists": frozenset(codes.tolist()),
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


def shares_of(seeds, weights, related, ok, related_n):
    shares = {}
    for s, w in zip(seeds, weights):
        candidates = [int(a) for a in related[s] if a >= 0 and ok(int(a))][:related_n]
        for place, a in enumerate(candidates):
            entry = shares.setdefault(a, {})
            entry[int(s)] = entry.get(int(s), 0.0) + float(w) * (1 - place / max(len(candidates), related_n))
    return shares


def app_rank(seeds, weights, related, ok, related_n, decay, k, tie):
    shares = shares_of(seeds, weights, related, ok, related_n)
    used = {int(s): 0 for s in seeds}
    ranked = []
    while shares and len(ranked) < k:
        best = max(shares, key=lambda a: (sum(v * decay ** used[s] for s, v in shares[a].items()), tie[a]))
        for s in shares.pop(best):
            used[s] += 1
        ranked.append(best)
    return ranked


_job = {}


def song_model_ranker(user, mode):
    n_seeds, related_n, decay = SCORING
    return app_rank(user["seeds"], user["weights"], _job["related"], allowed(user, mode), related_n, decay, K_OUT,
                    popularity)


def popularity_ranker(user, mode):
    ok = allowed(user, mode)
    out = []
    for s in popular_order:
        if ok(int(s)):
            out.append(int(s))
            if len(out) == K_OUT:
                break
    return out


def new_artists(user):
    """The tuned artist kNN's new artists for a user, best first, as song artist codes."""
    n_seeds, related_n, decay = SCORING
    seeds = [(int(model_of_code[c]), w) for c, w in zip(user["artist_seeds"], user["artist_weights"])
             if model_of_code[c] >= 0][:n_seeds]
    known_models = {int(model_of_code[c]) for c in user["known_artists"] if model_of_code[c] >= 0}
    ranked = app_rank([s for s, _ in seeds], [w for _, w in seeds], artist_neighbours,
                      lambda a: a not in known_models and a in code_of_model, related_n, decay, K_OUT,
                      np.zeros(len(artist_neighbours)))
    return [code_of_model[a] for a in ranked]


def app_today_ranker(user, mode):
    out = []
    for code in new_artists(user):
        out.extend(int(s) for s in by_artist[code][:_job["per_artist"]] if int(s) not in user["known"])
        if len(out) >= K_OUT:
            break
    return out[:K_OUT]


def artist_then_song_ranker(user, mode):
    """New artists from the artist kNN, and for each the song the song kNN likes best for this user, in rounds."""
    shares = shares_of(user["seeds"], user["weights"], _job["related"], lambda s: s not in user["known"], POOL)
    score = {s: sum(v.values()) for s, v in shares.items()}
    lists = []
    for code in new_artists(user):
        songs = [int(s) for s in by_artist[code] if int(s) not in user["known"]]
        lists.append(sorted(songs, key=lambda s: (-score.get(s, 0.0), -popularity[s])))
    out = []
    for depth in range(3):
        out.extend(songs[depth] for songs in lists if len(songs) > depth)
        if len(out) >= K_OUT:
            break
    return out[:K_OUT]


def own_artists_ranker(user, mode):
    lists = [iter(by_artist[c]) for c in user["artist_seeds"][:30]]
    out = []
    while lists and len(out) < K_OUT:
        for it in list(lists):
            s = next((int(x) for x in it if int(x) not in user["known"]), None)
            if s is None:
                lists.remove(it)
            else:
                out.append(s)
    return out[:K_OUT]


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


rankings = {}


def record(name, params, ranked, group, split_name, mode, seconds):
    rankings[(name, split_name, mode)] = ranked
    row = {"model": name, "params": params, "split": split_name, "mode": mode, **metrics(ranked, group, mode),
           "eval_seconds": round(seconds)}
    results.append(row)
    save()
    log(json.dumps(row))
    return row


def evaluate(name, params, ranker, group, split_name, mode, **job):
    start = time.time()
    _job.update(ranker=ranker, mode=mode, **job)
    if "fork" in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context("fork").Pool(os.cpu_count()) as pool:
            ranked = pool.map(_rank, group, chunksize=32)
    else:
        ranked = [_rank(u) for u in group]
    return record(name, params, ranked, group, split_name, mode, time.time() - start)


def fuse(name, parts, group, split_name, mode):
    """RRF of the ranked lists already kept for these parts."""
    start = time.time()
    fused = []
    for i in range(len(group)):
        score = {}
        for part in parts:
            for place, s in enumerate(rankings[(part, split_name, mode)][i]):
                score[s] = score.get(s, 0.0) + 1 / (RRF_K + place + 1)
        fused.append(sorted(score, key=lambda s: (-score[s], -popularity[s]))[:K_OUT])
    return record(name, {"parts": parts}, fused, group, split_name, mode, time.time() - start)


val, test = eval_users(1), eval_users(2)
log(f"validation users {len(val):,}, test users {len(test):,}")


# ---------------------------------------------------------------- similarity tables
def pool_from(model, matrix):
    """Every song's POOL best neighbours by the model's own score: ids and scores."""
    model.fit(matrix, show_progress=False)
    ids_out = np.full((n_songs, POOL), -1, dtype=np.int32)
    scores_out = np.zeros((n_songs, POOL), dtype=np.float32)
    for lo in range(0, n_songs, 20000):
        chunk = np.arange(lo, min(lo + 20000, n_songs))
        ids, scores = model.similar_items(chunk, N=POOL + 1)
        for i, row, values in zip(chunk, ids, scores):
            mask = (row != i) & (row >= 0) & (values > 0)
            kept = row[mask][:POOL]
            ids_out[i, :len(kept)] = kept
            scores_out[i, :len(kept)] = values[mask][:POOL]
    return ids_out, scores_out


def top_of(pool_ids, score, n):
    valid = pool_ids >= 0
    score = np.where(valid, score, -np.inf)
    n = min(n, pool_ids.shape[1])
    top = np.argpartition(-score, n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(score, top, axis=1), axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    out = np.take_along_axis(pool_ids, top, axis=1)
    out[~np.take_along_axis(valid, top, axis=1)] = -1
    return out


def knn_table(matrix, shrink=100.0):
    weighted = matrix.copy()
    weighted.data = np.log1p(weighted.data)
    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=0)).ravel()).astype(np.float32)
    ids, cos = pool_from(CosineRecommender(K=POOL), weighted)
    candidate_norms = norms[np.where(ids >= 0, ids, 0)]
    dot = cos * norms[:, None] * candidate_norms
    table = top_of(ids, dot / (norms[:, None] * candidate_norms + shrink + 1e-9), POOL)
    # Check: without shrinkage the rescoring gives implicit's cosine order back
    sample = np.random.default_rng(SEED).choice(n_songs, min(2000, n_songs), replace=False)
    plain = top_of(ids[sample], dot[sample] / (norms[sample, None] * candidate_norms[sample] + 1e-9), 20)
    same = float(np.mean([set(a[a >= 0].tolist()) == set(b[b >= 0].tolist())
                          for a, b in zip(plain, ids[sample, :20])]))
    log(f"check: cosine rescoring gives implicit's neighbours for {same:.1%} of the sample")
    assert same > 0.98
    return table


# Memory: the first run ran out of Kaggle's 30 GB with a million songs, so item2vec and EASE come first while the
# memory is free, the daily listens are dropped as soon as the sessions are cut, every RP3beta pool is scored and
# dropped at once, and only the main kNN keeps POOL neighbours: the other tables keep SMALL
tables = {}

# item2vec over the daily sessions of training users
start = time.time()
song_index = pd.Series(np.arange(n_songs), index=song_ids)
daily = daily[daily["user_id"].isin(set(user_ids[split == 0].tolist())) & daily["song"].isin(song_ids)]
daily = daily.sort_values(["user_id", "ts"], kind="stable")
items = song_index.loc[daily["song"].to_numpy()].to_numpy()
user_col, ts = daily["user_id"].to_numpy(), daily["ts"].to_numpy()
del daily
breaks = np.flatnonzero((user_col[1:] != user_col[:-1]) | (ts[1:] - ts[:-1] > SESSION_GAP)) + 1
sessions = []
for chunk in np.split(items, breaks):
    chunk = chunk[np.concatenate([[True], chunk[1:] != chunk[:-1]])]
    if len(chunk) >= 2:
        sessions.append(chunk)
del items, user_col, ts, breaks
total = sum(len(s) for s in sessions)
if total > MAX_TOKENS:
    keep_session = np.random.default_rng(SEED).random(len(sessions)) < MAX_TOKENS / total
    sessions = [s for s, k in zip(sessions, keep_session) if k]
sentences = [s.astype(str).tolist() for s in sessions]
del sessions
w2v = Word2Vec(sentences, vector_size=64, window=10, min_count=5, sg=1, negative=10, ns_exponent=-0.5, sample=1e-4,
               epochs=5, workers=os.cpu_count(), seed=SEED)
del sentences
vectors = np.zeros((n_songs, 64), dtype=np.float32)
for token, index in w2v.wv.key_to_index.items():
    vectors[int(token)] = w2v.wv.vectors[index]
del w2v
timed("item2vec", start)


def vector_table(vectors, ids, n=SMALL, batch=128):
    norms_v = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = (vectors / np.maximum(norms_v, 1e-9)).astype(np.float32)
    has = norms_v.ravel() > 0
    out = np.full((n_songs, n), -1, dtype=np.int32)
    ids = np.asarray([i for i in ids if has[i]])
    n = min(n, int(has.sum()) - 1)
    for lo in range(0, len(ids), batch):
        chunk = ids[lo:lo + batch]
        with threadpool_limits(os.cpu_count()):
            scores = unit[chunk] @ unit.T
        scores[:, ~has] = -np.inf
        scores[np.arange(len(chunk)), chunk] = -np.inf
        top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        out[chunk, :n] = np.take_along_axis(top, order, axis=1)
    return out


start = time.time()
seed_ids = np.unique(np.concatenate([u["seeds"] for u in val + test]))
tables["item2vec"] = vector_table(vectors, seed_ids)
del vectors
timed("item2vec neighbours", start)

# EASE on the most listened songs, the inverse worked out in place
start = time.time()
binary = train.copy()
binary.data[:] = 1.0
ease_songs = popular_order[:EASE_SONGS]
x = binary[:, ease_songs].tocsc()
with threadpool_limits(os.cpu_count()):
    gram = (x.T @ x).toarray().astype(np.float32)
    del x
    gram[np.diag_indices_from(gram)] += EASE_LAMBDA
    weights_b = np.linalg.inv(gram)
del gram
weights_b /= -np.diag(weights_b)[None, :]
np.fill_diagonal(weights_b, -np.inf)
n_ease = min(SMALL, len(ease_songs) - 1)
ease_table = np.full((n_songs, SMALL), -1, dtype=np.int32)
# In row blocks: an argpartition of the whole matrix would need an index array as large as the matrix
for lo in range(0, len(ease_songs), 1000):
    block = weights_b[lo:lo + 1000]
    local = np.argpartition(-block, n_ease - 1, axis=1)[:, :n_ease]
    local = np.take_along_axis(local, np.argsort(-np.take_along_axis(block, local, axis=1), axis=1), axis=1)
    ease_table[ease_songs[lo:lo + 1000], :n_ease] = ease_songs[local]
tables["ease"] = ease_table
del weights_b, ease_table
timed("ease", start)
save()

# kNN, and the same on 500 songs per user
start = time.time()
tables["knn"] = knn_table(train)
timed("knn", start)
start = time.time()
tables["knn, 500 songs per user"] = knn_table(train_old_cap)[:, :SMALL].copy()
del train_old_cap
timed("knn old cap", start)
save()

# ---------------------------------------------------------------- validation
start = time.time()
for mode in MODES:
    evaluate("knn reference (100 neighbours)", {}, song_model_ranker, val, "val", mode,
             related=tables["knn"][:, :REFERENCE_N])
    evaluate("knn big pool", {}, song_model_ranker, val, "val", mode, related=tables["knn"])
    evaluate("knn, 500 songs per user", {}, song_model_ranker, val, "val", mode,
             related=tables["knn, 500 songs per user"])
    evaluate("ease", {"songs": EASE_SONGS, "lambda": EASE_LAMBDA}, song_model_ranker, val, "val", mode,
             related=tables["ease"])
    evaluate("item2vec", {}, song_model_ranker, val, "val", mode, related=tables["item2vec"])
    evaluate("popularity", {}, popularity_ranker, val, "val", mode)
    evaluate("app today, 3 per artist", {}, app_today_ranker, val, "val", mode, per_artist=3)
    evaluate("app today, 1 per artist", {}, app_today_ranker, val, "val", mode, per_artist=1)
    evaluate("artist then song", {}, artist_then_song_ranker, val, "val", mode, related=tables["knn"])
evaluate("own artists' top songs", {}, own_artists_ranker, val, "val", "all")

# RP3beta: P(u|i) = 1/d_i and P(j|u) = 1/d_u on the binary matrix, with the walk damped by alpha. Within a song's
# row 1/d_i^alpha is the same for every candidate, so the ranking is sum_u x_ui x_uj d_u^-alpha / d_j^beta.
# Each alpha's pool is scored on the validation users at once and only the best table per task is kept
user_degree = np.asarray(binary.sum(axis=1)).ravel()
song_degree = np.maximum(popularity, 1).astype(np.float32)
best_rp3 = {}
for alpha in RP3_ALPHAS:
    pool_start = time.time()
    scaled = sp.diags(np.power(np.maximum(user_degree, 1), -alpha / 2).astype(np.float32)) @ binary
    # implicit's plain item-item model wants float64
    ids, co = pool_from(ItemItemRecommender(K=POOL), scaled.astype(np.float64).tocsr())
    del scaled
    timed(f"rp3 pool alpha {alpha}", pool_start)
    for beta in RP3_BETAS:
        table = top_of(ids, co / song_degree[np.where(ids >= 0, ids, 0)] ** beta, SMALL)
        for mode in MODES:
            row = evaluate("rp3beta", {"alpha": alpha, "beta": beta}, song_model_ranker, val, "val", mode,
                           related=table)
            if row["ndcg@12"] > best_rp3.get(mode, (None, -1))[1]:
                best_rp3[mode] = ((alpha, beta), row["ndcg@12"])
                rankings[("rp3beta best", "val", mode)] = rankings[("rp3beta", "val", mode)]
                tables[f"rp3beta {mode}"] = table
        del table
    del ids, co
del binary
summary["best_rp3"] = {mode: {"alpha": a, "beta": b} for mode, ((a, b), _) in best_rp3.items()}
save()

FUSIONS = {
    "all": [["knn big pool", "item2vec"], ["knn big pool", "rp3beta best"], ["knn big pool", "ease"],
            ["knn big pool", "rp3beta best", "item2vec"], ["knn big pool", "rp3beta best", "item2vec", "ease"],
            ["knn big pool", "own artists' top songs"]],
    "discovery": [["knn big pool", "item2vec"], ["knn big pool", "rp3beta best"], ["knn big pool", "ease"],
                  ["knn big pool", "rp3beta best", "item2vec"], ["knn big pool", "artist then song"],
                  ["knn big pool", "rp3beta best", "artist then song"]],
}
for mode in MODES:
    for parts in FUSIONS[mode]:
        fuse("rrf", parts, val, "val", mode)
timed("validation", start)

# ---------------------------------------------------------------- test: reference, every family's best, baselines
start = time.time()
summary["test"] = {}
singles = {"knn reference (100 neighbours)": lambda: tables["knn"][:, :REFERENCE_N],
           "knn big pool": lambda: tables["knn"], "knn, 500 songs per user": lambda: tables["knn, 500 songs per user"],
           "ease": lambda: tables["ease"], "item2vec": lambda: tables["item2vec"]}
for mode in MODES:
    val_rows = [r for r in results if r["split"] == "val" and r["mode"] == mode]
    best_fusion = max((r for r in val_rows if r["model"] == "rrf"), key=lambda r: r["ndcg@12"])
    rows = [evaluate(name, {}, song_model_ranker, test, "test", mode, related=get()) for name, get in singles.items()]
    a, b = best_rp3[mode][0]
    rows.append(evaluate("rp3beta", {"alpha": a, "beta": b}, song_model_ranker, test, "test", mode,
                         related=tables[f"rp3beta {mode}"]))
    rankings[("rp3beta best", "test", mode)] = rankings[("rp3beta", "test", mode)]
    rows.append(evaluate("artist then song", {}, artist_then_song_ranker, test, "test", mode, related=tables["knn"]))
    rows.append(evaluate("popularity", {}, popularity_ranker, test, "test", mode))
    rows.append(evaluate("app today, 3 per artist", {}, app_today_ranker, test, "test", mode, per_artist=3))
    rows.append(evaluate("app today, 1 per artist", {}, app_today_ranker, test, "test", mode, per_artist=1))
    if mode == "all":
        rows.append(evaluate("own artists' top songs", {}, own_artists_ranker, test, "test", mode))
    rows.append(fuse("rrf", best_fusion["params"]["parts"], test, "test", mode))
    reference = rows[0]
    best = max(rows, key=lambda r: r["ndcg@12"])
    summary["test"][mode] = {
        "rows": rows, "best": {"model": best["model"], "params": best["params"]},
        "best_vs_reference": {k: round(best[k] / reference[k] - 1, 4) for k in ("recall@12", "hit@12", "ndcg@12")
                              if reference[k]}}
    save()
timed("test", start)

# ---------------------------------------------------------------- export the best single table for all songs
best_single = max((r for r in results if r["split"] == "test" and r["mode"] == "all" and r["model"] != "rrf"
                   and r["model"] in list(singles) + ["rp3beta"]), key=lambda r: r["ndcg@12"])
name = best_single["model"]
table = tables["rp3beta all"] if name == "rp3beta" else singles[name]()
np.savez_compressed(f"{OUT}/neighbours_song_best.npz", ids=table[:, :EXPORT_N].astype(np.int32))
np.savez_compressed(f"{OUT}/neighbours_song_knn_big.npz", ids=tables["knn"][:, :200].astype(np.int32))
meta.assign(s=np.arange(n_songs, dtype=np.int32), users=popularity.astype(np.int32))[
    ["s", "song", "key", "artist", "title", "mbid", "users"]].to_parquet(f"{OUT}/songs_model.parquet", index=False)
summary["exported"] = {"best_single_all": name, "params": best_single["params"], "songs": n_songs}
summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("done")
