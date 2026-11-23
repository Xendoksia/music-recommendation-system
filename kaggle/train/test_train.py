"""
Runs kaggle/build/build.py and then train.py (quick grid) locally on made up data with a known structure: 6 genres
of 40 artists, each user mostly in one genre. The models should find the genres and beat popularity.
Not uploaded to Kaggle.
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
raw, built, out = (os.path.join(root, name) for name in ("raw", "built", "out"))
os.makedirs(os.path.join(raw, "listens"))
rng = np.random.default_rng(1)

genres, per_genre = 6, 40
n_artists = genres * per_genre
pd.DataFrame({"artist": np.arange(n_artists, dtype=np.int32), "key": [f"m-{i}" for i in range(n_artists)],
              "name": [f"Artist {i}" for i in range(n_artists)],
              "mbid": [f"m-{i}" for i in range(n_artists)]}).to_parquet(f"{raw}/artists.parquet")
# Within a genre a few artists are much more played, as in real data
weights = np.tile(1 / np.arange(1, per_genre + 1) ** 0.8, genres)
rows, listens = [], []
for user in range(1, 1501):
    genre = rng.integers(genres)
    own = np.arange(genre * per_genre, (genre + 1) * per_genre)
    p = weights[own] / weights[own].sum()
    picks = set(rng.choice(own, 13, replace=False, p=p)) | set(rng.choice(n_artists, 2, replace=False))
    for artist in picks:
        rows.append((user, int(artist), int(rng.integers(1, 200))))
    t = 1_790_000_000
    for i in range(30):
        t += 240 if i % 10 else 7200
        listens.append((user, t, int(rng.choice(list(picks)))))
pd.DataFrame(rows, columns=["user_id", "artist", "listens"]).astype("int32").to_parquet(f"{raw}/user_artists.parquet")
pd.DataFrame(listens, columns=["user_id", "ts", "artist"]).astype("int32") \
    .to_parquet(f"{raw}/listens/listenbrainz-spark-dump-1-incremental.parquet")

env = dict(os.environ, BUILD_INPUT=raw, BUILD_OUT=built, TRAIN_INPUT=built, TRAIN_OUT=out, TRAIN_QUICK="1",
           TUNE_USERS="300", PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, os.path.join(here, "..", "build", "build.py")], check=True, env=env)
subprocess.run([sys.executable, os.path.join(here, "train.py")], check=True, env=env)

results = json.load(open(f"{out}/results.json"))
test = {r["model"]: r for r in results if r["split"] == "test"}
for name, row in test.items():
    print(f"{name:10} recall@20 {row['recall@20']:.3f}  ndcg@20 {row['ndcg@20']:.3f}  hit@12 {row['hit@12']:.3f}")
for name in ("knn", "als", "item2vec"):
    assert test[name]["recall@20"] > test["popularity"]["recall@20"], f"{name} does not beat popularity"
    with np.load(f"{out}/neighbours_{name}.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    assert data["ids"].shape == (n_artists, 50), data["ids"].shape
    # The closest neighbour of an artist should mostly be in its own genre. build.py numbers artists by
    # popularity, so the genre comes from the original id in the key ("m-<id>")
    genre = pd.read_parquet(f"{out}/artists.parquet")["key"].str[2:].astype(int).to_numpy() // per_genre
    first = data["ids"][:, 0]
    same = np.mean([genre[a] == genre[b] for a, b in enumerate(first) if b >= 0])
    print(f"{name:10} nearest neighbour in the same genre: {same:.0%}")
    assert same > 0.6, f"{name} neighbours ignore the genres"
print("ok")
shutil.rmtree(root, ignore_errors=True)
