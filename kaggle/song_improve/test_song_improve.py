"""
Runs song_improve.py locally (ST_QUICK) on made up data with a known structure: 6 genres of 8 artists with 12 songs
each, every user mostly in one genre. Checks that the song models and the app's way beat popularity and that the
outputs are written. Not uploaded to Kaggle.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
root = tempfile.mkdtemp()
songs_dir, stats_dir, tune_dir, out = (os.path.join(root, n) for n in ("songs", "stats", "tune", "out"))
for d in (os.path.join(songs_dir, "listens"), stats_dir, tune_dir):
    os.makedirs(d)
rng = np.random.default_rng(3)

genres, per_genre, per_artist = 6, 8, 12
n_artists = genres * per_genre
artist_of = np.repeat(np.arange(n_artists), per_artist)
n_songs = len(artist_of)
names = [f"Artist {a // per_genre}-{a % per_genre}" for a in range(n_artists)]
songs = pd.DataFrame({"song": np.arange(n_songs, dtype=np.int32),
                      "key": [f"{names[a].lower()}|song {i}" for i, a in enumerate(artist_of)],
                      "artist": [names[a] for a in artist_of], "title": [f"Song {i}" for i in range(n_songs)],
                      "mbid": [""] * n_songs, "listeners": 1, "listens": 1})
songs.to_parquet(os.path.join(songs_dir, "songs.parquet"))
songs.drop(columns=["listeners", "listens"]).to_parquet(os.path.join(stats_dir, "songs_all.parquet"))

# Popular artists and songs within each genre
artist_weight = np.tile(1 / np.arange(1, per_genre + 1) ** 0.7, genres)
song_weight = np.tile(1 / np.arange(1, per_artist + 1) ** 0.8, n_artists)
stats_rows, daily_rows = [], []
for user in range(1, 3001):
    g = rng.integers(genres)
    own = np.arange(g * per_genre, (g + 1) * per_genre)
    # Six artists of the genre with only a few songs each, so hiding a fifth of the songs sometimes hides a whole
    # artist: that is what the discovery task looks for
    liked = rng.choice(own, 6, replace=False, p=artist_weight[own] / artist_weight[own].sum())
    pool = np.flatnonzero(np.isin(artist_of, liked))
    picks = rng.choice(pool, 14, replace=False, p=song_weight[pool] / song_weight[pool].sum())
    picks = np.concatenate([picks, rng.choice(n_songs, 1, replace=False)])
    for s in np.unique(picks):
        stats_rows.append((user, int(s), int(rng.integers(1, 100))))
    if user % 2:
        t = 1_790_000_000
        for i in range(40):
            t += 200 if i % 10 else 7200
            daily_rows.append((user, t, int(rng.choice(picks))))
pd.DataFrame(stats_rows, columns=["user_id", "song", "listens"]).astype("int32") \
    .to_parquet(os.path.join(stats_dir, "user_songs.parquet"))
pd.DataFrame(daily_rows, columns=["user_id", "ts", "song"]).astype("int32") \
    .to_parquet(os.path.join(songs_dir, "listens", "listenbrainz-spark-dump-1-incremental.parquet"))

# The tuned artist model: same genre artists first
pd.DataFrame({"a": np.arange(n_artists, dtype=np.int32), "key": names, "name": names, "mbid": [""] * n_artists,
              "users": (1000 * artist_weight).astype(np.int32)}).to_parquet(os.path.join(tune_dir, "artists.parquet"))
ids = np.full((n_artists, 50), -1, dtype=np.int32)
for a in range(n_artists):
    g = a // per_genre
    same = [b for b in range(g * per_genre, (g + 1) * per_genre) if b != a]
    other = [b for b in range(n_artists) if b // per_genre != g]
    row = same + other
    ids[a, :len(row[:50])] = row[:50]
np.savez_compressed(os.path.join(tune_dir, "neighbours_knn_tuned.npz"), ids=ids)

env = dict(os.environ, ST_SONGS=songs_dir, ST_STATS=stats_dir, ST_TUNE=tune_dir, ST_OUT=out, ST_QUICK="1",
           PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, os.path.join(here, "song_improve.py")], check=True, env=env)

summary = json.load(open(os.path.join(out, "summary.json")))
for mode, block in summary["test"].items():
    print(mode, "best", block["best"], "vs reference", block["best_vs_reference"])
    rows = block["rows"]
    for r in rows:
        print(f"  {r['model']:32} {json.dumps(r['params'])[:60]:60} recall@12 {r['recall@12']:.3f} "
              f"ndcg@12 {r['ndcg@12']:.3f} users {r['users']}")
    pop = next(r for r in rows if r["model"] == "popularity")["recall@12"]
    for name in ("knn big pool", "rp3beta", "rrf"):
        got = next(r for r in rows if r["model"] == name)["recall@12"]
        assert got > pop, f"{mode}: {name} does not beat popularity"
assert set(summary["best_rp3"]) == {"all", "discovery"}
model = pd.read_parquet(os.path.join(out, "songs_model.parquet"))
for name, width in (("neighbours_song_best.npz", 50), ("neighbours_song_knn_big.npz", 200)):
    with np.load(os.path.join(out, name)) as npz:
        assert npz["ids"].shape[0] == len(model) and npz["ids"].shape[1] <= width, npz["ids"].shape
print("seconds", summary["seconds"])
print("ok")
shutil.rmtree(root, ignore_errors=True)
