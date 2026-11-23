"""Runs song_stats.py locally on a made up statistics archive and checks the keys and counts. Not uploaded."""

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

import pandas as pd
import zstandard

here = os.path.dirname(os.path.abspath(__file__))
root = tempfile.mkdtemp()
out = os.path.join(root, "out")
songs_path = os.path.join(root, "songs.parquet")
pd.DataFrame({"song": [0, 1], "key": ["goksel|acıyor".replace("ı", "i"), "three days grace|animal i have become"],
              "artist": ["Göksel", "Three Days Grace"], "title": ["Acıyor", "Animal I Have Become"],
              "mbid": ["", "m-2"], "listeners": [1, 2], "listens": [1, 2]}).to_parquet(songs_path)

rows = [
    {"user_id": 7, "data": [
        {"listen_count": 50, "track_name": "Animal I Have Become - Remastered", "artist_name": "Three Days Grace",
         "recording_mbid": "m-2"},
        {"listen_count": 9, "track_name": "Acıyor (Live)", "artist_name": "Goksel", "recording_mbid": None},
        {"listen_count": 3, "track_name": "Numb", "artist_name": "Linkin Park, Jay-Z", "recording_mbid": "m-3"},
        {"listen_count": 2, "track_name": "Numb", "artist_name": "Linkin Park", "recording_mbid": None},
        {"listen_count": 1, "track_name": "", "artist_name": "Nobody"}]},
    {"user_id": 8, "data": [{"listen_count": 4, "track_name": "Numb", "artist_name": "Linkin Park feat. Jay-Z"}]},
]
payload = b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in rows)
tar_bytes = io.BytesIO()
with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
    for name, data in (("x/lbdump/statistics/artists_all_time.jsonl", b"{}\n"),
                       ("x/lbdump/statistics/recordings_all_time.jsonl", payload)):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
archive = os.path.join(root, "stats.tar.zst")
with open(archive, "wb") as f:
    f.write(zstandard.ZstdCompressor().compress(tar_bytes.getvalue()))

env = dict(os.environ, STATS_TEST_FILE=archive, STATS_OUT=out, STATS_SONGS=songs_path, PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, os.path.join(here, "song_stats.py")], check=True, env=env)

songs = pd.read_parquet(os.path.join(out, "songs_all.parquet"))
user_songs = pd.read_parquet(os.path.join(out, "user_songs.parquet"))
print(songs)
print(user_songs)
assert songs["song"].tolist() == [0, 1, 2] and songs.loc[2, "key"] == "linkin park|numb", songs
pairs = {(u, s): n for u, s, n in user_songs.itertuples(index=False)}
assert pairs == {(7, 1): 50, (7, 0): 9, (7, 2): 5, (8, 2): 4}, pairs
summary = json.load(open(os.path.join(out, "summary.json")))
assert summary["title_field"] == "track_name" and summary["user_songs"]["new_songs"] == 1
print("ok")
shutil.rmtree(root, ignore_errors=True)
