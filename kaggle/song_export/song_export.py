"""
Songs, step 5 on Kaggle: the tables for the chosen song model, knn + RP3beta (option C of
reports/sarki_iyilestirme.md). The data, split and both models are those of kaggle/song_improve, so the test scores
must come out the same.

The app scores one neighbour table per song. Two ways to use both models are measured on the test users:
- user RRF: the app's scoring runs on each table and the two ranked lists are fused (what song_improve measured).
- song RRF: each song's two neighbour lists are fused once into one table, which the app can use as it is.

Exported: neighbours_song_knn_big.npz, neighbours_song_rp3.npz and neighbours_song_fused.npz (FUSED_N neighbours
each) and songs_model.parquet.
"""

import gc
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
EASE_SONGS = 200 if QUICK else 15000
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
del songs_all
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
del pairs, rank_in_user, listens_v, hidden_mask, keep, old_keep, u_of, s_of
gc.collect()

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


def rescored(pool_ids, score_of, n, block=50000):
    """top_of in row blocks: on a million songs every whole (songs, POOL) temporary is 1.7 GB or more, and a dozen
    of them at once ran Kaggle out of memory in the third run"""
    out = np.full((pool_ids.shape[0], min(n, pool_ids.shape[1])), -1, dtype=np.int32)
    for lo in range(0, pool_ids.shape[0], block):
        rows = np.arange(lo, min(lo + block, pool_ids.shape[0]))
        out[rows] = top_of(pool_ids[rows], score_of(rows), n)
    return out


def knn_table(matrix, n=POOL, shrink=100.0):
    weighted = matrix.copy()
    weighted.data = np.log1p(weighted.data)
    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=0)).ravel()).astype(np.float32)
    ids, cos = pool_from(CosineRecommender(K=POOL), weighted)
    del weighted
    gc.collect()

    def score_of(rows, extra):
        products = norms[rows, None] * norms[np.where(ids[rows] >= 0, ids[rows], 0)]
        return cos[rows] * products / (products + extra + 1e-9)

    table = rescored(ids, lambda rows: score_of(rows, shrink), n)
    # Check: without shrinkage the rescoring gives implicit's cosine order back
    sample = np.random.default_rng(SEED).choice(n_songs, min(2000, n_songs), replace=False)
    plain = top_of(ids[sample], score_of(sample, 0.0), 20)
    same = float(np.mean([set(a[a >= 0].tolist()) == set(b[b >= 0].tolist())
                          for a, b in zip(plain, ids[sample, :20])]))
    log(f"check: cosine rescoring gives implicit's neighbours for {same:.1%} of the sample")
    assert same > 0.98
    del ids, cos
    gc.collect()
    return table


FUSED_N = 40 if QUICK else 200
RP3_ALPHA, RP3_BETA = 0.5, 0.6
tables = {}

start = time.time()
tables["knn"] = knn_table(train)
timed("knn", start)

start = time.time()
binary = train.copy()
binary.data[:] = 1.0
user_degree = np.asarray(binary.sum(axis=1)).ravel()
song_degree = np.maximum(popularity, 1).astype(np.float32)
scaled = sp.diags(np.power(np.maximum(user_degree, 1), -RP3_ALPHA / 2).astype(np.float32)) @ binary
del binary
# implicit's plain item-item model wants float64
ids, co = pool_from(ItemItemRecommender(K=POOL), scaled.astype(np.float64).tocsr())
del scaled
tables["rp3"] = rescored(ids, lambda rows: co[rows] / song_degree[np.where(ids[rows] >= 0, ids[rows], 0)] ** RP3_BETA,
                         FUSED_N)
del ids, co
gc.collect()
timed("rp3", start)


def song_fused(first, second, n=FUSED_N, block=50000):
    """Per song, RRF of its two neighbour lists: ties by popularity, then by id."""
    out = np.full((n_songs, n), -1, dtype=np.int32)
    width = first.shape[1] + second.shape[1]
    for lo in range(0, n_songs, block):
        rows = np.arange(lo, min(lo + block, n_songs))
        cand = np.concatenate([first[rows], second[rows]], axis=1).astype(np.int64)
        places = np.concatenate([np.arange(first.shape[1]), np.arange(second.shape[1])])
        gain = np.broadcast_to(1.0 / (RRF_K + places + 1), cand.shape)
        row_of = np.repeat(np.arange(len(rows)), width)
        ok = cand.ravel() >= 0
        keys, inverse = np.unique(row_of[ok] * n_songs + cand.ravel()[ok], return_inverse=True)
        score = np.bincount(inverse, weights=gain.ravel()[ok])
        key_row, key_song = keys // n_songs, keys % n_songs
        order = np.lexsort((key_song, -popularity[key_song], -score, key_row))
        key_row, key_song = key_row[order], key_song[order]
        first_of_row = np.searchsorted(key_row, np.arange(len(rows)))
        place = np.arange(len(key_row)) - first_of_row[key_row]
        keep = place < n
        out[rows[key_row[keep]], place[keep]] = key_song[keep]
    return out


start = time.time()
tables["fused"] = song_fused(tables["knn"][:, :FUSED_N], tables["rp3"])
# Check against a plain per song loop on a sample
sample = np.random.default_rng(SEED).choice(n_songs, min(500, n_songs), replace=False)
same = 0
for i in sample:
    score = {}
    for row in (tables["knn"][i, :FUSED_N], tables["rp3"][i]):
        for place, s in enumerate(row):
            if s >= 0:
                score[int(s)] = score.get(int(s), 0.0) + 1 / (RRF_K + place + 1)
    want = sorted(score, key=lambda s: (-round(score[s], 12), -popularity[s], s))[:FUSED_N]
    got = tables["fused"][i]
    same += want == got[got >= 0].tolist()
log(f"check: song fusion matches the plain loop for {same / len(sample):.1%} of the sample")
assert same / len(sample) > 0.99
timed("song fusion", start)
save()

# ---------------------------------------------------------------- test
start = time.time()
summary["test"] = {}
for mode in MODES:
    rows = [evaluate("knn reference (100 neighbours)", {}, song_model_ranker, test, "test", mode,
                     related=tables["knn"][:, :REFERENCE_N]),
            evaluate("knn big pool", {}, song_model_ranker, test, "test", mode, related=tables["knn"]),
            evaluate("rp3beta best", {"alpha": RP3_ALPHA, "beta": RP3_BETA}, song_model_ranker, test, "test", mode,
                     related=tables["rp3"])]
    rows.append(fuse("user rrf", ["knn big pool", "rp3beta best"], test, "test", mode))
    rows.append(evaluate("song rrf", {"neighbours": FUSED_N}, song_model_ranker, test, "test", mode,
                         related=tables["fused"]))
    summary["test"][mode] = rows
    save()
timed("test", start)

# ---------------------------------------------------------------- export
n_out = min(FUSED_N, tables["knn"].shape[1])
np.savez_compressed(f"{OUT}/neighbours_song_knn_big.npz", ids=tables["knn"][:, :n_out].astype(np.int32))
np.savez_compressed(f"{OUT}/neighbours_song_rp3.npz", ids=tables["rp3"].astype(np.int32))
np.savez_compressed(f"{OUT}/neighbours_song_fused.npz", ids=tables["fused"].astype(np.int32))
meta.assign(s=np.arange(n_songs, dtype=np.int32), users=popularity.astype(np.int32))[
    ["s", "song", "key", "artist", "title", "mbid", "users"]].to_parquet(f"{OUT}/songs_model.parquet", index=False)
summary["exported"] = {"rp3": {"alpha": RP3_ALPHA, "beta": RP3_BETA}, "neighbours": FUSED_N, "songs": n_songs,
                       "scoring": list(SCORING), "rrf_k": RRF_K}
summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("done")
