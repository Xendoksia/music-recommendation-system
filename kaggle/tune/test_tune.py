"""
Runs kaggle/build/build.py and then tune.py (quick grid) locally on made up data with a known structure, as
kaggle/train/test_train.py does, and checks the outputs. Not uploaded to Kaggle.
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

env = dict(os.environ, BUILD_INPUT=raw, BUILD_OUT=built, TUNE_INPUT=built, TUNE_OUT=out, TUNE_QUICK="1",
           TUNE_USERS="300", TUNE_POOL="100", PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, os.path.join(here, "..", "build", "build.py")], check=True, env=env)
subprocess.run([sys.executable, os.path.join(here, "tune.py")], check=True, env=env)

results = json.load(open(f"{out}/results.json"))
summary = json.load(open(f"{out}/summary.json"))
stages = {r["stage"] for r in results}
assert stages == {"A", "B", "C", "test"}, stages
# 2 betas x 2 shrinks, then 2 x 2 x 1 scorings, then 1 runner up, then 2 test rows
assert len(results) == 4 + 4 + 1 + 2, len(results)
current, best = summary["test_current"], summary["test_best"]
print(f"current ndcg@12 {current['ndcg@12']:.3f} hit@12 {current['hit@12']:.3f}")
print(f"best    ndcg@12 {best['ndcg@12']:.3f} hit@12 {best['hit@12']:.3f}  {summary['best']}")
assert current["params"] == {"beta": 0.5, "shrink": 0.0, "seeds": 10, "related": 20, "decay": 0.5}

with np.load(f"{out}/neighbours_knn_tuned.npz") as npz:
    ids = npz["ids"]
assert ids.shape == (n_artists, 50), ids.shape
assert all(ids[i, 0] != i for i in range(n_artists))
genre = pd.read_parquet(f"{out}/artists.parquet")["key"].str[2:].astype(int).to_numpy() // per_genre
same = np.mean([genre[a] == genre[b] for a, b in enumerate(ids[:, 0]) if b >= 0])
print(f"nearest neighbour in the same genre: {same:.0%}")
assert same > 0.6
print("ok")
shutil.rmtree(root, ignore_errors=True)
