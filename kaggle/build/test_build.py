"""Runs build.py locally on a small made up sample and checks its outputs. Not uploaded to Kaggle."""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
root = tempfile.mkdtemp()
src, out = os.path.join(root, "in"), os.path.join(root, "out")
os.makedirs(os.path.join(src, "listens"))

# 0 Göksel (MBID), 1 "n:goksel" (same artist without MBID), 2 a collaboration starting with Göksel,
# 3, 4 and 6-11 bands (MBID), 5 a rare artist that the filter drops
bands = [f"m-{c}" for c in "abcdefgh"]
pd.DataFrame({"artist": np.arange(6 + 6, dtype=np.int32),
              "key": ["m-goksel", "n:goksel", "n:goksel, someone", *bands[:2], "m-rare", *bands[2:]],
              "name": ["Göksel", "Goksel", "Göksel, Someone", "Band A", "Band B", "Rare",
                       *[f"Band {c.upper()}" for c in "cdefgh"]],
              "mbid": ["m-goksel", None, None, *bands[:2], "m-rare", *bands[2:]]}).to_parquet(f"{src}/artists.parquet")
rng = np.random.default_rng(0)
rows = []
for user in range(1, 61):
    for artist in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11):
        if rng.random() < 0.8:
            rows.append((user, artist, int(rng.integers(1, 100))))
    if user <= 3:
        rows.append((user, 5, 5))
stats = pd.DataFrame(rows, columns=["user_id", "artist", "listens"]).astype("int32")
stats.to_parquet(f"{src}/user_artists.parquet")
listens = []
for user in range(1, 61):
    t = 1_790_000_000
    for i in range(12):
        t += 200 if i != 6 else 4000
        listens.append((user, t, int(rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]))))
pd.DataFrame(listens, columns=["user_id", "ts", "artist"]).astype("int32") \
    .to_parquet(f"{src}/listens/listenbrainz-spark-dump-1-incremental.parquet")

env = dict(os.environ, BUILD_INPUT=src, BUILD_OUT=out, PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, os.path.join(here, "build.py")], check=True, env=env)

artists = pd.read_parquet(f"{out}/artists.parquet")
train = pd.read_parquet(f"{out}/train.parquet")
holdout = pd.read_parquet(f"{out}/holdout.parquet")
users = pd.read_parquet(f"{out}/users.parquet")
items = np.load(f"{out}/sessions_items.npy")
offsets = np.load(f"{out}/sessions_offsets.npy")
print(artists.to_string())
assert list(artists["key"]).count("m-goksel") == 1 and "n:goksel" not in set(artists["key"]), "Göksel not merged"
assert "m-rare" not in set(artists["key"]), "rare artist kept"
assert len(artists) == 9, f"expected 9 artists, got {len(artists)}"
assert not set(map(tuple, train[["u", "a"]].values)) & set(map(tuple, holdout[["u", "a"]].values)), "leak"
assert set(holdout["u"]) <= set(users.loc[users.split != "train", "u"]), "train user in holdout"
assert offsets[-1] == len(items) and (np.diff(offsets) >= 2).all(), "bad sessions"
sessions = [items[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
assert all((s[1:] != s[:-1]).all() for s in sessions), "repeats left in a session"
print(f"ok: {len(train)} train pairs, {len(holdout)} hidden, {len(sessions)} sessions")
shutil.rmtree(root)
