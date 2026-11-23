"""
Part 2, step 1 on Kaggle: what the ListenBrainz dumps hold, before the real download is written.
- one daily incremental spark dump: its files, parquet columns, row count and a few rows
- the statistics dump, streamed and never stored: its member names and sizes, and the start of the user artist files
Writes /kaggle/working/explore.json.
"""

import io
import json
import re
import subprocess
import sys
import tarfile
import time
import urllib.request

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard"], check=True)
import pyarrow.parquet as pq  # noqa: E402
import zstandard  # noqa: E402

BASE = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz"
USER_AGENT = "PlaylistTrackerResearch/0.1 ( https://github.com/Xendoksia/Spotify-Youtube--Track-Ur-Playlist )"
result = {}


def get(url, **kwargs):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=120,
                                  **kwargs)


def links(url):
    return re.findall(r'href="([^"?/][^"]*)"', get(url).read().decode())


# 1. A daily incremental spark dump
incremental = [d for d in links(f"{BASE}/incremental/") if d.endswith("/")]
result["incremental_dirs"] = len(incremental)
result["incremental_first_last"] = [incremental[0], incremental[-1]]
chosen = None
for folder in reversed(incremental):
    files = links(f"{BASE}/incremental/{folder}")
    spark = [f for f in files if "spark" in f and not f.endswith((".md5", ".sha256"))]
    if spark:
        size = int(get(f"{BASE}/incremental/{folder}{spark[0]}").headers.get("Content-Length", 0))
        if size > 50_000_000:
            chosen = (folder, spark[0], size)
            break
folder, name, size = chosen
result["incremental_sample"] = {"folder": folder, "file": name, "bytes": size}
start = time.time()
path = f"/tmp/{name}"
with get(f"{BASE}/incremental/{folder}{name}") as response, open(path, "wb") as out:
    while chunk := response.read(1 << 20):
        out.write(chunk)
result["incremental_sample"]["download_seconds"] = round(time.time() - start, 1)

members = []
raw = open(path, "rb")
source = zstandard.ZstdDecompressor().stream_reader(raw) if name.endswith(".zst") else raw
with tarfile.open(fileobj=source, mode="r|") as archive:
    for member in archive:
        members.append({"name": member.name, "bytes": member.size})
        if member.name.endswith(".parquet") and "parquet" not in result:
            table = pq.read_table(io.BytesIO(archive.extractfile(member).read()))
            result["parquet"] = {"member": member.name, "rows": table.num_rows,
                                 "schema": str(table.schema),
                                 "sample": table.slice(0, 3).to_pylist()}
result["incremental_members"] = members[:50]
result["incremental_member_count"] = len(members)
print(json.dumps(result, indent=1, default=str)[:6000])

# 2. The statistics dump, streamed: member names and sizes, and the first lines of artist files
full = [d for d in links(f"{BASE}/fullexport/") if d.endswith("/")][-1]
stats_name = [f for f in links(f"{BASE}/fullexport/{full}") if f.startswith("listenbrainz-statistics")
              and f.endswith(".tar.zst")][0]
result["stats_file"] = full + stats_name
start = time.time()
stats_members = []
peeks = {}
with get(f"{BASE}/fullexport/{full}{stats_name}") as response:
    reader = zstandard.ZstdDecompressor().stream_reader(response, read_size=1 << 20)
    with tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            stats_members.append({"name": member.name, "bytes": member.size})
            if member.isfile() and "artist" in member.name and len(peeks) < 6:
                head = archive.extractfile(member).read(3000)
                peeks[member.name] = head.decode("utf-8", errors="replace")
result["stats_seconds"] = round(time.time() - start, 1)
result["stats_member_count"] = len(stats_members)
result["stats_members"] = stats_members[:400]
result["stats_artist_peeks"] = peeks

with open("/kaggle/working/explore.json", "w") as out:
    json.dump(result, out, indent=1, default=str)
print(json.dumps({k: v for k, v in result.items() if k not in ("incremental_members", "stats_members")},
                 indent=1, default=str)[:8000])
