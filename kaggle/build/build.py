"""
Part 2, step 4 on Kaggle: the training data, from kaggle/prep's output.

1. One artist per real artist: a listen without an MBID ("n:" + folded name) joins the MBID artist with the same
   folded name (the one with most listeners), and a collaboration ("A, B" or "A feat. B") goes to its first artist,
   as the app does.
2. Filtering: artists with at least MIN_ARTIST_USERS listeners, then users with at least MIN_USER_ARTISTS of them.
3. Split by user: VAL_SHARE of users for choosing hyperparameters, TEST_SHARE for the final score. HIDE_SHARE of
   each of their artists is hidden and the rest stays in the training matrix as what is known about them.
4. Sessions for item2vec from the daily dumps: a user's listens in time order, broken where the gap is longer than
   SESSION_GAP, repeats in a row dropped. Validation and test users are left out so their hidden artists do not leak.

Listen counts are kept raw, as their weighting (log or BM25) is a hyperparameter of Part 3.
Outputs in OUT:
  artists.parquet   a (int32, index used everywhere), key, name, mbid, users
  users.parquet     u (int32), user_id, split (train | val | test)
  train.parquet     u, a, listens: every training pair, and the visible pairs of validation and test users
  holdout.parquet   u, a, listens, split: the hidden pairs of validation and test users
  sessions_items.npy (int32) and sessions_offsets.npy (int64): session i is items[offsets[i]:offsets[i + 1]]
  summary.json
"""

import glob
import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MIN_ARTIST_USERS = 10
MIN_USER_ARTISTS = 5
VAL_SHARE = 0.10
TEST_SHARE = 0.10
HIDE_SHARE = 0.20
SESSION_GAP = 30 * 60
SEED = 42

# BUILD_INPUT and BUILD_OUT let the script run locally on a small sample
INPUT = os.environ.get("BUILD_INPUT") or os.path.dirname(
    glob.glob("/kaggle/input/**/artists.parquet", recursive=True)[0])
OUT = os.environ.get("BUILD_OUT") or "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
summary = {"settings": {"min_artist_users": MIN_ARTIST_USERS, "min_user_artists": MIN_USER_ARTISTS,
                        "val_share": VAL_SHARE, "test_share": TEST_SHARE, "hide_share": HIDE_SHARE,
                        "session_gap": SESSION_GAP, "seed": SEED}}
_CREDIT_SEPARATORS = re.compile(r",| feat\.? | ft\.? ", flags=re.IGNORECASE)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def fold(text):
    text = (text or "").replace("ı", "i").replace("İ", "i")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def first_credit(name):
    parts = [part.strip() for part in _CREDIT_SEPARATORS.split(name or "") if part.strip()]
    return parts[0] if parts else (name or "")


# 1. One artist per real artist
artists = pq.read_table(f"{INPUT}/artists.parquet").to_pandas()
stats = pq.read_table(f"{INPUT}/user_artists.parquet").to_pandas()
log(f"read {len(artists):,} artists and {len(stats):,} user artist rows")
listeners = np.bincount(stats["artist"].to_numpy(), minlength=len(artists))
is_name_key = artists["key"].str.startswith("n:").to_numpy()

# The MBID artist each folded name stands for: the one with most listeners
mbid_rows = artists.loc[~is_name_key, ["artist", "name"]].copy()
mbid_rows["fold"] = mbid_rows["name"].map(fold)
mbid_rows["users"] = listeners[mbid_rows["artist"].to_numpy()]
by_fold = (mbid_rows.sort_values("users", ascending=False).drop_duplicates("fold")
           .set_index("fold")["artist"].to_dict())
key_to_artist = dict(zip(artists["key"], artists["artist"]))

canon = np.arange(len(artists), dtype=np.int32)
merged_to_mbid = merged_to_first = 0
for artist, name in zip(artists.loc[is_name_key, "artist"], artists.loc[is_name_key, "name"]):
    first = fold(first_credit(name))
    target = by_fold.get(first)
    if target is not None:
        merged_to_mbid += 1
    else:
        target = key_to_artist.get("n:" + first, artist)
        if target != artist:
            merged_to_first += 1
    canon[artist] = target
summary["merge"] = {"name_keys": int(is_name_key.sum()), "joined_mbid_artist": merged_to_mbid,
                    "joined_first_credit": merged_to_first}
log(json.dumps(summary["merge"]))

stats["artist"] = canon[stats["artist"].to_numpy()]
stats = stats.groupby(["user_id", "artist"], as_index=False)["listens"].sum()

# 2. Filtering
artist_users = stats.groupby("artist").size()
keep_artists = artist_users.index[artist_users >= MIN_ARTIST_USERS]
stats = stats[stats["artist"].isin(keep_artists)]
user_artists = stats.groupby("user_id").size()
stats = stats[stats["user_id"].isin(user_artists.index[user_artists >= MIN_USER_ARTISTS])]
artist_users = stats.groupby("artist").size().sort_values(ascending=False)

# Compact indexes: artists by listeners, most first
old_to_a = np.full(len(artists), -1, dtype=np.int32)
old_to_a[artist_users.index.to_numpy()] = np.arange(len(artist_users), dtype=np.int32)
final_artists = artists.set_index("artist").loc[artist_users.index, ["key", "name", "mbid"]].reset_index(drop=True)
final_artists.insert(0, "a", np.arange(len(final_artists), dtype=np.int32))
final_artists["users"] = artist_users.to_numpy().astype(np.int32)
final_artists.to_parquet(f"{OUT}/artists.parquet", index=False)

user_ids = np.sort(stats["user_id"].unique())
stats = pd.DataFrame({"u": np.searchsorted(user_ids, stats["user_id"].to_numpy()).astype(np.int32),
                      "a": old_to_a[stats["artist"].to_numpy()], "listens": stats["listens"].to_numpy()})
summary["matrix"] = {"users": len(user_ids), "artists": len(final_artists), "pairs": len(stats),
                     "density": round(len(stats) / (len(user_ids) * len(final_artists)), 6)}
log(json.dumps(summary["matrix"]))

# 3. Split by user
rng = np.random.default_rng(SEED)
order = rng.permutation(len(user_ids))
n_val, n_test = int(len(user_ids) * VAL_SHARE), int(len(user_ids) * TEST_SHARE)
split = np.array(["train"] * len(user_ids), dtype=object)
split[order[:n_val]] = "val"
split[order[n_val:n_val + n_test]] = "test"
pd.DataFrame({"u": np.arange(len(user_ids), dtype=np.int32), "user_id": user_ids, "split": split}) \
    .to_parquet(f"{OUT}/users.parquet", index=False)

stats["split"] = split[stats["u"].to_numpy()]
held = stats["split"] != "train"
# A random share of each validation and test user's artists, at least one
stats["r"] = rng.random(len(stats))
stats["rank"] = stats.groupby("u")["r"].rank(method="first")
stats["n"] = stats.groupby("u")["a"].transform("size")
hidden = held & (stats["rank"] <= np.maximum(1, np.floor(stats["n"] * HIDE_SHARE)))
stats.loc[~hidden, ["u", "a", "listens"]].to_parquet(f"{OUT}/train.parquet", index=False)
stats.loc[hidden, ["u", "a", "listens", "split"]].to_parquet(f"{OUT}/holdout.parquet", index=False)
summary["split"] = {"train_users": int((split == "train").sum()), "val_users": n_val, "test_users": n_test,
                    "train_pairs": int((~hidden).sum()), "hidden_pairs": int(hidden.sum())}
log(json.dumps(summary["split"]))
del stats

# 4. Sessions from the daily dumps
train_user_ids = set(user_ids[split == "train"])
frames = []
for path in sorted(glob.glob(f"{INPUT}/**/listenbrainz-spark-dump-*.parquet", recursive=True)):
    day = pq.read_table(path).to_pandas()
    day["a"] = old_to_a[canon[day["artist"].to_numpy()]]
    day = day[(day["a"] >= 0) & day["user_id"].isin(train_user_ids)]
    frames.append(day[["user_id", "ts", "a"]])
listens = pd.concat(frames, ignore_index=True).drop_duplicates(["user_id", "ts", "a"])
listens = listens.sort_values(["user_id", "ts"], kind="stable")
summary["listen_users_in_statistics"] = int(listens["user_id"].nunique())
user = listens["user_id"].to_numpy()
ts = listens["ts"].to_numpy().astype(np.int64)
items = listens["a"].to_numpy()
del listens, frames
new_session = np.ones(len(items), dtype=bool)
new_session[1:] = (user[1:] != user[:-1]) | (ts[1:] - ts[:-1] > SESSION_GAP)
repeat = np.zeros(len(items), dtype=bool)
repeat[1:] = (items[1:] == items[:-1]) & ~new_session[1:]
items, new_session = items[~repeat], new_session[~repeat]
starts = np.flatnonzero(new_session)
lengths = np.diff(np.append(starts, len(items)))
keep = lengths >= 2
kept_items = items[np.repeat(keep, lengths)]
offsets = np.concatenate([[0], np.cumsum(lengths[keep])]).astype(np.int64)
np.save(f"{OUT}/sessions_items.npy", kept_items.astype(np.int32))
np.save(f"{OUT}/sessions_offsets.npy", offsets)
summary["sessions"] = {"sessions": int(keep.sum()), "items": int(len(kept_items)),
                       "mean_length": round(float(lengths[keep].mean()), 1) if keep.any() else 0,
                       "artists_seen": int(len(np.unique(kept_items)))}
log(json.dumps(summary["sessions"]))

with open(f"{OUT}/summary.json", "w") as out:
    json.dump(summary, out, indent=1)
log("done")
