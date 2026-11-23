"""
Songs, step 3a on Kaggle: every user's all time songs from the ListenBrainz statistics dump
(recordings_all_time.jsonl), with the same song keys as kaggle/songs (first credited artist | plain title, folded).

The 15 days of daily dumps (kaggle/songs) know 47 thousand users. The statistics dump keeps years of listening for
about twice as many, though probably only each user's most played songs. Its songs join the vocabulary of
kaggle/songs, so the song numbers of the daily listens stay valid and new songs are added after them.

The archive is streamed and never stored. Only recordings_all_time.jsonl is read.

Outputs in /kaggle/working (STATS_OUT locally):
  user_songs.parquet  user_id (int32), song (int32), listens (int32)
  songs_all.parquet   song (int32), key, artist, title, mbid: kaggle/songs' songs and then the new ones
  summary.json        with the first line of the file, to see its fields
"""

import glob
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

if not os.environ.get("STATS_TEST_FILE"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard", "orjson"], check=True)
import numpy as np  # noqa: E402
import orjson  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import zstandard  # noqa: E402

BASE = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz"
USER_AGENT = "PlaylistTrackerResearch/0.1 ( https://github.com/Xendoksia/Spotify-Youtube--Track-Ur-Playlist )"
MEMBER = "/recordings_all_time.jsonl"
TITLE_FIELDS = ("track_name", "recording_name", "title", "name")
# STATS_TEST_FILE reads a local .tar.zst instead of the server, STATS_SONGS points to a local songs.parquet
TEST_FILE = os.environ.get("STATS_TEST_FILE")
OUT = os.environ.get("STATS_OUT") or "/kaggle/working"
SONGS = os.environ.get("STATS_SONGS") or glob.glob("/kaggle/input/**/songs.parquet", recursive=True)[0]
os.makedirs(OUT, exist_ok=True)
summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

_CREDIT_SEPARATORS = re.compile(r",| feat\.? | ft\.? ", flags=re.IGNORECASE)


def fold(text):
    text = (text or "").replace("ı", "i").replace("İ", "i")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def first_artist(name):
    parts = [p.strip() for p in _CREDIT_SEPARATORS.split(name or "") if p.strip()]
    return parts[0] if parts else (name or "")


def plain_title(name):
    name = name or ""
    return re.sub(r"\s*(\(.*?\)|\[.*?\])", "", name.split(" - ")[0]).strip() or name


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=600)


def links(url):
    return re.findall(r'href="([^"?/][^"]*)"', get(url).read().decode())


# The vocabulary of kaggle/songs, extended with the songs only the statistics know
known = pd.read_parquet(SONGS, columns=["song", "key", "artist", "title", "mbid"])
vocab = dict(zip(known["key"], known["song"].astype(int)))
new_info: list[tuple[str, str, str, str]] = []
log(f"{len(vocab):,} songs from the daily dumps")
artist_cache: dict[str, str] = {}


def song_id(artist, title, mbid):
    folded_artist = artist_cache.get(artist)
    if folded_artist is None:
        folded_artist = artist_cache[artist] = fold(first_artist(artist))
    key = folded_artist + "|" + fold(plain_title(title))
    found = vocab.get(key)
    if found is None:
        found = vocab[key] = len(known) + len(new_info)
        new_info.append((key, first_artist(artist), plain_title(title), mbid or ""))
    return found


def source():
    if TEST_FILE:
        return open(TEST_FILE, "rb")
    full = [d for d in links(f"{BASE}/fullexport/") if d.endswith("/")][-1]
    name = [f for f in links(f"{BASE}/fullexport/{full}")
            if f.startswith("listenbrainz-statistics") and f.endswith(".tar.zst")][0]
    summary["stats_dump"] = full + name
    return get(f"{BASE}/fullexport/{full}{name}")


users, songs, counts = array("i"), array("i"), array("i")
title_field = None
start = time.time()
with source() as response:
    reader = zstandard.ZstdDecompressor().stream_reader(response, read_size=1 << 20)
    with tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            if not member.name.endswith(MEMBER):
                continue
            log(f"reading {member.name} ({member.size / 1e9:.1f} GB)")
            for n, line in enumerate(archive.extractfile(member)):
                row = orjson.loads(line)
                if n == 0:
                    summary["first_line"] = line[:2000].decode("utf-8", "replace")
                    log(summary["first_line"])
                for entry in row.get("data") or []:
                    if title_field is None:
                        title_field = next((f for f in TITLE_FIELDS if entry.get(f)), None)
                        if title_field is None:
                            raise SystemExit(f"no title field in {entry}")
                        summary["title_field"] = title_field
                    title = entry.get(title_field) or ""
                    if not title.strip():
                        continue
                    users.append(row["user_id"])
                    songs.append(song_id(entry.get("artist_name") or "", title, entry.get("recording_mbid")))
                    counts.append(min(entry.get("listen_count") or 0, 2**31 - 1))
                if n % 10000 == 0 and n:
                    log(f"{n:,} users, {len(users):,} rows, {len(new_info):,} new songs")
            break
log(f"read {len(users):,} rows in {time.time() - start:.0f}s")

frame = pd.DataFrame({"user_id": np.frombuffer(users, dtype=np.int32), "song": np.frombuffer(songs, dtype=np.int32),
                      "listens": np.frombuffer(counts, dtype=np.int32)})
del users, songs, counts
# Two spellings of one song in a user's list fold to one key: their listens are added
frame = frame.groupby(["user_id", "song"], as_index=False)["listens"].sum()
pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), f"{OUT}/user_songs.parquet")
added = pd.DataFrame(new_info, columns=["key", "artist", "title", "mbid"])
added.insert(0, "song", np.arange(len(known), len(known) + len(added), dtype=np.int32))
all_songs = pd.concat([known.astype({"song": np.int32}), added], ignore_index=True)
pq.write_table(pa.Table.from_pandas(all_songs, preserve_index=False), f"{OUT}/songs_all.parquet")
listeners = np.bincount(frame["song"].to_numpy(), minlength=len(all_songs))
summary["user_songs"] = {
    "rows": len(frame), "users": int(frame.user_id.nunique()), "songs_listened": int((listeners > 0).sum()),
    "new_songs": len(added), "per_user_max": int(frame.groupby("user_id").size().max()),
    "per_user_median": float(frame.groupby("user_id").size().median()),
    **{f"songs_with_{n}_listeners": int((listeners >= n).sum()) for n in (5, 10, 20, 50)},
    "seconds": round(time.time() - start)}
log(json.dumps(summary["user_songs"]))
summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(f"{OUT}/summary.json", "w") as out:
    json.dump(summary, out, indent=1)
log("done")
