"""
Part 2, step 2 on Kaggle: the ListenBrainz data the models learn from, made small.

- From the statistics dump only artists_all_time.jsonl is read (each user's artists with listen counts). It sits near
  the start of the archive, so the stream is closed as soon as it has been read.
- Every daily incremental spark dump still on the server (about 15 days) is read for listening order.

Artists get one shared number: their first credited MBID, or for a listen without one "n:" + its folded name.
Outputs in /kaggle/working:
  artists.parquet         artist (int32), key, name
  user_artists.parquet    user_id (int32), artist (int32), listens (int32), from the all time statistics
  listens/<dump>.parquet  user_id (int32), ts (int32 epoch seconds), artist (int32), one file per daily dump
  summary.json
"""

import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.request
from array import array

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard", "orjson"], check=True)
import numpy as np  # noqa: E402
import orjson  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import zstandard  # noqa: E402

BASE = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz"
USER_AGENT = "PlaylistTrackerResearch/0.1 ( https://github.com/Xendoksia/Spotify-Youtube--Track-Ur-Playlist )"
# PREP_TEST_FOLDER runs only the incremental part for that one folder, into ./prep_test, to test the code locally
TEST_FOLDER = os.environ.get("PREP_TEST_FOLDER")
OUT = "prep_test" if TEST_FOLDER else "/kaggle/working"
TMP = "prep_test" if TEST_FOLDER else "/tmp"
os.makedirs(f"{OUT}/listens", exist_ok=True)
summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

vocab: dict[str, int] = {}
names: list[str] = []


def fold(text):
    text = (text or "").replace("ı", "i").replace("İ", "i")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def artist_ids(keys: pd.Series, artist_names: pd.Series) -> np.ndarray:
    """The shared artist number of each row, new artists added to the vocabulary with their first seen name."""
    codes, uniques = pd.factorize(keys)
    first = pd.Series(artist_names.values).groupby(codes).first()
    ids = np.empty(len(uniques), dtype=np.int32)
    for i, unique in enumerate(uniques):
        found = vocab.get(unique)
        if found is None:
            found = vocab[unique] = len(names)
            names.append(first.iloc[i])
        ids[i] = found
    return ids[codes]


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=300)


def links(url):
    return re.findall(r'href="([^"?/][^"]*)"', get(url).read().decode())


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


# 1. All time user artists from the statistics dump
def artist_id(key, name):
    found = vocab.get(key)
    if found is None:
        found = vocab[key] = len(names)
        names.append(name)
    return found


def read_statistics():
    full = [d for d in links(f"{BASE}/fullexport/") if d.endswith("/")][-1]
    stats_name = [f for f in links(f"{BASE}/fullexport/{full}")
                  if f.startswith("listenbrainz-statistics") and f.endswith(".tar.zst")][0]
    summary["stats_dump"] = full + stats_name
    users, artists, counts = array("i"), array("i"), array("i")
    start = time.time()
    with get(f"{BASE}/fullexport/{full}{stats_name}") as response:
        reader = zstandard.ZstdDecompressor().stream_reader(response, read_size=1 << 20)
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if not member.name.endswith("/artists_all_time.jsonl"):
                    continue
                for line in archive.extractfile(member):
                    row = orjson.loads(line)
                    for entry in row.get("data") or []:
                        name = entry.get("artist_name") or ""
                        mbid = entry.get("artist_mbid")
                        users.append(row["user_id"])
                        artists.append(artist_id(mbid if mbid else "n:" + fold(name), name))
                        counts.append(min(entry.get("listen_count") or 0, 2**31 - 1))
                break
    log(f"statistics read: {len(users):,} rows in {time.time() - start:.0f}s")
    frame = pd.DataFrame({"user_id": np.frombuffer(users, dtype=np.int32),
                          "artist": np.frombuffer(artists, dtype=np.int32),
                          "listens": np.frombuffer(counts, dtype=np.int32)})
    del users, artists, counts
    # A user's artist may appear twice when two spellings fold to one key: their listens are added
    frame = frame.groupby(["user_id", "artist"], as_index=False)["listens"].sum()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), f"{OUT}/user_artists.parquet")
    summary["user_artists"] = {"rows": len(frame), "users": int(frame.user_id.nunique()),
                               "artists": int(frame.artist.nunique()), "listens": int(frame.listens.sum()),
                               "seconds": round(time.time() - start)}
    log(json.dumps(summary["user_artists"]))
    del frame


if not TEST_FOLDER:
    read_statistics()

# 2. Daily incremental spark dumps: listening order
folders = [TEST_FOLDER] if TEST_FOLDER else [d for d in links(f"{BASE}/incremental/") if d.endswith("/")]
summary["incremental"] = []
for folder in folders:
    spark = [f for f in links(f"{BASE}/incremental/{folder}")
             if "spark" in f and ".tar" in f and not f.endswith((".md5", ".sha256"))]
    if not spark:
        continue
    name = spark[0]
    start = time.time()
    path = f"{TMP}/{name}"
    with get(f"{BASE}/incremental/{folder}{name}") as response, open(path, "wb") as out:
        while chunk := response.read(1 << 22):
            out.write(chunk)
    raw = open(path, "rb")
    source = zstandard.ZstdDecompressor().stream_reader(raw) if name.endswith(".zst") else raw
    parts = []
    with tarfile.open(fileobj=source, mode="r|") as archive:
        for member in archive:
            if not member.name.endswith(".parquet"):
                continue
            # pyarrow needs a seekable file and a member of a streamed archive is not one
            table = pq.read_table(io.BytesIO(archive.extractfile(member).read()),
                                  columns=["listened_at", "user_id", "recording_msid", "artist_name",
                                           "artist_credit_mbids"])
            day = table.to_pandas()
            day = day.drop_duplicates(["user_id", "listened_at", "recording_msid"])
            mbid = day["artist_credit_mbids"].map(lambda m: m[0] if m is not None and len(m) else None)
            unique_names = pd.unique(day["artist_name"])
            folded = dict(zip(unique_names, ("n:" + fold(n) for n in unique_names)))
            key = mbid.where(mbid.notna(), day["artist_name"].map(folded))
            parts.append(pd.DataFrame({
                "user_id": day["user_id"].astype(np.int32).values,
                # Seconds whatever unit pandas keeps the timestamps in
                "ts": day["listened_at"].astype("datetime64[s]").astype("int64").astype(np.int32).values,
                "artist": artist_ids(key.reset_index(drop=True), day["artist_name"].reset_index(drop=True)),
            }))
    raw.close()
    os.remove(path)
    if not parts:
        log(f"{name}: no listens")
        continue
    listens = pd.concat(parts, ignore_index=True).sort_values(["user_id", "ts"], kind="stable")
    stem = name.split(".tar")[0]
    pq.write_table(pa.Table.from_pandas(listens, preserve_index=False), f"{OUT}/listens/{stem}.parquet")
    info = {"dump": stem, "rows": len(listens), "users": int(listens.user_id.nunique()),
            "first": int(listens.ts.min()) if len(listens) else None,
            "last": int(listens.ts.max()) if len(listens) else None, "seconds": round(time.time() - start)}
    summary["incremental"].append(info)
    log(json.dumps(info))

# 3. The shared artist vocabulary
mbids = [key if not key.startswith("n:") else None for key in vocab]
pq.write_table(pa.table({"artist": np.arange(len(names), dtype=np.int32), "key": list(vocab), "name": names,
                         "mbid": mbids}), f"{OUT}/artists.parquet")
summary["artists"] = len(names)
summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(f"{OUT}/summary.json", "w") as out:
    json.dump(summary, out, indent=1)
log(json.dumps({k: v for k, v in summary.items() if k != "incremental"}))
