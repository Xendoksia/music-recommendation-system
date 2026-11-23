"""
Songs, step 1 on Kaggle: the ListenBrainz daily dumps read again, this time keeping the song of every listen.

A song's key is its first credited artist and its plain title, both folded, by the app's rules: a collaboration
("A, B", "A feat. B") goes to its first artist, and version notes such as " - Remastered 2026" or " (Deluxe)" are
dropped (playlist_tracker/music/artwork.py _plain_title). The user's own plays carry no MBIDs, so songs are matched
by these keys, and a single and its album version count as one song.

Outputs in /kaggle/working (SONGS_OUT locally):
  songs.parquet           song (int32), key, artist, title, mbid (a recording MBID seen for it, if any),
                          listeners, listens
  song_counts.parquet     key, listeners, listens, for songs with at least MIN_LISTENERS listeners: downloaded
                          to measure the user's coverage locally, so the user's songs never leave their computer
  listens/<dump>.parquet  user_id (int32), ts (int32 epoch seconds), song (int32), one file per daily dump
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

if not os.environ.get("SONGS_TEST_DIR"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard"], check=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

BASE = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz"
USER_AGENT = "PlaylistTrackerResearch/0.1 ( https://github.com/Xendoksia/Spotify-Youtube--Track-Ur-Playlist )"
MIN_LISTENERS = 3
COLUMNS = ["listened_at", "user_id", "recording_msid", "artist_name", "recording_name", "recording_mbid"]
# SONGS_TEST_DIR reads the .tar files of a local folder instead of the server, to test the code locally
TEST_DIR = os.environ.get("SONGS_TEST_DIR")
OUT = os.environ.get("SONGS_OUT") or "/kaggle/working"
TMP = TEST_DIR or "/tmp"
os.makedirs(f"{OUT}/listens", exist_ok=True)
summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "dumps": []}

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


def song_key(artist, title):
    return fold(first_artist(artist)) + "|" + fold(plain_title(title))


vocab: dict[str, int] = {}
info: list[tuple[str, str, str]] = []  # artist, title, mbid of each song, as first seen


def song_ids(day: pd.DataFrame) -> np.ndarray:
    """The song number of each listen. Keys are built once per distinct artist and title pair of the day."""
    pairs = day[["artist_name", "recording_name"]]
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(pairs))
    mbids = day["recording_mbid"].groupby(codes).first()
    ids = np.empty(len(uniques), dtype=np.int32)
    for i, (artist, title) in enumerate(uniques):
        key = song_key(artist, title)
        found = vocab.get(key)
        if found is None:
            found = vocab[key] = len(info)
            info.append((first_artist(artist), plain_title(title), mbids.iloc[i] or ""))
        elif not info[found][2] and mbids.iloc[i]:
            info[found] = (info[found][0], info[found][1], mbids.iloc[i])
        ids[i] = found
    return ids[codes]


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=300)


def links(url):
    return re.findall(r'href="([^"?/][^"]*)"', get(url).read().decode())


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def dump_files():
    """(name, local path) of every daily spark dump, downloaded one at a time on Kaggle."""
    if TEST_DIR:
        for name in sorted(os.listdir(TEST_DIR)):
            if name.endswith(".tar"):
                yield name, os.path.join(TEST_DIR, name)
        return
    for folder in [d for d in links(f"{BASE}/incremental/") if d.endswith("/")]:
        spark = [f for f in links(f"{BASE}/incremental/{folder}")
                 if "spark" in f and ".tar" in f and not f.endswith((".md5", ".sha256"))]
        if not spark:
            continue
        path = f"{TMP}/{spark[0]}"
        with get(f"{BASE}/incremental/{folder}{spark[0]}") as response, open(path, "wb") as out:
            while chunk := response.read(1 << 22):
                out.write(chunk)
        yield spark[0], path
        os.remove(path)


for name, path in dump_files():
    start = time.time()
    raw = open(path, "rb")
    if name.endswith(".zst"):
        import zstandard
        source = zstandard.ZstdDecompressor().stream_reader(raw)
    else:
        source = raw
    parts = []
    with tarfile.open(fileobj=source, mode="r|") as archive:
        for member in archive:
            if not member.name.endswith(".parquet"):
                continue
            table = pq.read_table(io.BytesIO(archive.extractfile(member).read()), columns=COLUMNS)
            day = table.to_pandas()
            day = day.drop_duplicates(["user_id", "listened_at", "recording_msid"])
            day = day[day["recording_name"].fillna("").str.strip() != ""].reset_index(drop=True)
            day["recording_mbid"] = day["recording_mbid"].fillna("")
            parts.append(pd.DataFrame({
                "user_id": day["user_id"].astype(np.int32).values,
                "ts": day["listened_at"].astype("datetime64[s]").astype("int64").astype(np.int32).values,
                "song": song_ids(day),
            }))
    raw.close()
    if not parts:
        log(f"{name}: no listens")
        continue
    listens = pd.concat(parts, ignore_index=True).sort_values(["user_id", "ts"], kind="stable")
    stem = name.split(".tar")[0]
    pq.write_table(pa.Table.from_pandas(listens, preserve_index=False), f"{OUT}/listens/{stem}.parquet")
    row = {"dump": stem, "listens": len(listens), "users": int(listens.user_id.nunique()), "songs_so_far": len(info),
           "seconds": round(time.time() - start)}
    summary["dumps"].append(row)
    log(json.dumps(row))

# Listeners and listens per song over every dump
start = time.time()
files = sorted(f for f in os.listdir(f"{OUT}/listens") if f.endswith(".parquet"))
all_listens = pd.concat([pd.read_parquet(f"{OUT}/listens/{f}", columns=["user_id", "song"]) for f in files],
                        ignore_index=True)
listens_per_song = np.bincount(all_listens["song"].to_numpy(), minlength=len(info))
pairs = all_listens.drop_duplicates()
listeners_per_song = np.bincount(pairs["song"].to_numpy(), minlength=len(info))
summary["total"] = {"listens": len(all_listens), "users": int(all_listens.user_id.nunique()), "songs": len(info),
                    **{f"songs_with_{n}_listeners": int((listeners_per_song >= n).sum()) for n in (3, 5, 10, 20, 50)},
                    "user_song_pairs": len(pairs), "seconds": round(time.time() - start)}
log(json.dumps(summary["total"]))
del all_listens, pairs

keys = list(vocab)
songs = pd.DataFrame({"song": np.arange(len(info), dtype=np.int32), "key": keys,
                      "artist": [i[0] for i in info], "title": [i[1] for i in info], "mbid": [i[2] for i in info],
                      "listeners": listeners_per_song.astype(np.int32), "listens": listens_per_song.astype(np.int32)})
pq.write_table(pa.Table.from_pandas(songs, preserve_index=False), f"{OUT}/songs.parquet")
counts = songs.loc[songs["listeners"] >= MIN_LISTENERS, ["key", "listeners", "listens"]]
pq.write_table(pa.Table.from_pandas(counts, preserve_index=False), f"{OUT}/song_counts.parquet")
summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(f"{OUT}/summary.json", "w") as out:
    json.dump(summary, out, indent=1)
log("done")
