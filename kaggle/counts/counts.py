"""
Part 2, step 3 on Kaggle: listeners per artist in the prepared data, so the user's artists can be looked up locally
without sending their list anywhere. Only artists with at least MIN_USERS listeners in either source are kept.
Output: /kaggle/working/artist_counts.parquet  artist, key, name, stats_users, stats_listens, recent_users,
recent_listens
"""

import glob
import os

import pandas as pd
import pyarrow.parquet as pq

MIN_USERS = 5
source = os.path.dirname(glob.glob("/kaggle/input/**/artists.parquet", recursive=True)[0])
print("input:", source, flush=True)

stats = pq.read_table(f"{source}/user_artists.parquet").to_pandas()
by_artist = stats.groupby("artist").agg(stats_users=("user_id", "size"), stats_listens=("listens", "sum"))
del stats

pairs, totals = [], []
for path in sorted(glob.glob(f"{source}/**/listenbrainz-spark-dump-*.parquet", recursive=True)):
    day = pq.read_table(path, columns=["user_id", "artist"]).to_pandas()
    totals.append(day["artist"].value_counts())
    pairs.append(day.drop_duplicates())
recent = pd.concat(pairs, ignore_index=True).drop_duplicates()
recent_users = recent.groupby("artist").size().rename("recent_users")
recent_listens = pd.concat(totals).groupby(level=0).sum().rename("recent_listens")
print("recent pairs:", len(recent), flush=True)

counts = by_artist.join(recent_users, how="outer").join(recent_listens, how="outer").fillna(0).astype("int64")
counts = counts[(counts.stats_users >= MIN_USERS) | (counts.recent_users >= MIN_USERS)]
artists = pq.read_table(f"{source}/artists.parquet", columns=["artist", "key", "name"]).to_pandas()
counts = artists.merge(counts, left_on="artist", right_index=True)
counts.to_parquet("/kaggle/working/artist_counts.parquet", index=False)
print(len(counts), "artists kept", flush=True)
print(counts.sort_values("stats_users", ascending=False).head(10).to_string(), flush=True)
