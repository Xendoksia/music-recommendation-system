"""Runs songs.py locally on two made up daily dumps and checks the song keys and counts. Not uploaded to Kaggle."""

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

here = os.path.dirname(os.path.abspath(__file__))
root = tempfile.mkdtemp()
dumps, out = os.path.join(root, "dumps"), os.path.join(root, "out")
os.makedirs(dumps)

# (user, minute, artist_name, recording_name, recording_mbid)
days = [
    [(1, 0, "Göksel", "Bu Aşk Fazla Sana", None),
     (1, 4, "Goksel", "Bu Aşk Fazla Sana - Remastered 2020", "m-1"),  # same song, other spelling and a version note
     (2, 0, "Three Days Grace", "Animal I Have Become", "m-2"),
     (2, 0, "Three Days Grace", "Animal I Have Become", "m-2"),  # the same listen twice: dropped
     (3, 1, "Linkin Park, Jay-Z", "Numb / Encore", None),  # a collaboration goes to its first artist
     (3, 5, "Linkin Park", "Numb / Encore (Live)", None),
     (4, 2, "Nobody", "", None)],  # no title: dropped
    [(1, 0, "Three Days Grace", "Animal I Have Become [Deluxe]", None),
     (5, 0, "Three Days Grace", "Animal I Have Become", "m-2"),
     (5, 3, "Linkin Park feat. Jay-Z", "Numb / Encore", None)],
]
for d, rows in enumerate(days):
    frame = pd.DataFrame({
        "listened_at": pd.to_datetime([1_790_000_000 + d * 86400 + m * 60 for _, m, *_ in rows], unit="s"),
        "created": pd.to_datetime([1_790_000_000] * len(rows), unit="s"),
        "user_id": [u for u, *_ in rows],
        "recording_msid": [f"{a}|{t}|{u}" for u, _, a, t, _ in rows],
        "artist_name": [a for *_, a, _, _ in rows],
        "recording_name": [t for *_, t, _ in rows],
        "recording_mbid": [m for *_, m in rows],
        "release_name": ["x"] * len(rows),
    })
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), buffer)
    with tarfile.open(os.path.join(dumps, f"listenbrainz-spark-dump-{d}-incremental.tar"), "w") as tar:
        info = tarfile.TarInfo(f"listenbrainz-spark-dump-{d}-incremental/0.parquet")
        info.size = buffer.tell()
        buffer.seek(0)
        tar.addfile(info, buffer)

env = dict(os.environ, SONGS_TEST_DIR=dumps, SONGS_OUT=out, PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, os.path.join(here, "songs.py")], check=True, env=env)

songs = pd.read_parquet(os.path.join(out, "songs.parquet")).set_index("key")
print(songs[["artist", "title", "mbid", "listeners", "listens"]])
assert set(songs.index) == {"goksel|bu ask fazla sana", "three days grace|animal i have become",
                            "linkin park|numb / encore"}, list(songs.index)
assert songs.loc["goksel|bu ask fazla sana", ["listeners", "listens"]].tolist() == [1, 2]
assert songs.loc["goksel|bu ask fazla sana", "mbid"] == "m-1"
assert songs.loc["three days grace|animal i have become", ["listeners", "listens"]].tolist() == [3, 3]
assert songs.loc["linkin park|numb / encore", ["listeners", "listens"]].tolist() == [2, 3]
counts = pd.read_parquet(os.path.join(out, "song_counts.parquet"))
assert counts["key"].tolist() == ["three days grace|animal i have become"], counts
listens = pd.concat(pd.read_parquet(os.path.join(out, "listens", f)) for f in sorted(os.listdir(os.path.join(out, "listens"))))
assert len(listens) == 8 and list(listens.columns) == ["user_id", "ts", "song"]
summary = json.load(open(os.path.join(out, "summary.json")))
assert summary["total"]["listens"] == 8 and summary["total"]["songs"] == 3
print("ok")
shutil.rmtree(root, ignore_errors=True)
