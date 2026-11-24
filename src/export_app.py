"""
Exports the chosen models for Playlist Tracker in a form the app reads with the standard library only (gzip, json,
array): no numpy, pandas or parquet needed.

- artists.json.gz, artist_neighbours.i32.gz: the tuned artist kNN (kaggle/tune, B settings)
- songs.json.gz, song_neighbours.i32.gz: the song kNN (kaggle/song_train)
- manifest.json: shapes, settings, sources and test scores

Only artists and songs with at least MIN_USERS listeners are kept, and every neighbour list is filtered to the kept
ones, so a list can be shorter than NEIGHBOURS: the rest of a row is -1.

    python -m src.export_app <out folder>
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
import unicodedata
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TUNE = ROOT / "kaggle" / "tune" / "output"
SONGS = ROOT / "kaggle" / "song_train" / "output"
MIN_USERS = 50
NEIGHBOURS = 50


def fold(text: str) -> str:
    """The app's fold (playlist_tracker/music/recommend.py): no case, accents or Turkish letters."""
    text = (text or "").replace("ı", "i").replace("İ", "i")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def kept_table(ids: np.ndarray, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows with at least MIN_USERS listeners, renumbered, their lists filtered to kept rows."""
    keep = np.flatnonzero(users >= MIN_USERS)
    new_id = np.full(len(users), -1, dtype=np.int64)
    new_id[keep] = np.arange(len(keep))
    out = np.full((len(keep), NEIGHBOURS), -1, dtype=np.int32)
    for row, old in enumerate(keep):
        mapped = new_id[ids[old][ids[old] >= 0]]
        mapped = mapped[mapped >= 0][:NEIGHBOURS]
        out[row, :len(mapped)] = mapped
    return keep, out


def write_ids(path: Path, table: np.ndarray) -> None:
    data = array("i", table.astype("<i4").ravel().tolist())
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(data.tobytes())


def write_json(path: Path, rows: list) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))


def label(row: dict) -> str:
    """A result's name, with the songs per artist for the two kinds of the app's way."""
    per_artist = row["params"].get("per_artist")
    return f"{row['model']}, {per_artist} per artist" if per_artist else row["model"]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()

    artists = pd.read_parquet(TUNE / "artists.parquet")
    keep, table = kept_table(np.load(TUNE / "neighbours_knn_tuned.npz")["ids"], artists["users"].to_numpy())
    rows = artists.iloc[keep]
    write_json(out / "artists.json.gz", [
        {"name": n, "key": fold(n), "mbid": m or "", "listeners": int(u)}
        for n, m, u in zip(rows["name"], rows["mbid"].fillna(""), rows["users"])])
    write_ids(out / "artist_neighbours.i32.gz", table)
    artist_shape = table.shape

    songs = pd.read_parquet(SONGS / "songs_model.parquet")
    keep, table = kept_table(np.load(SONGS / "neighbours_song_knn.npz")["ids"], songs["users"].to_numpy())
    rows = songs.iloc[keep]
    write_json(out / "songs.json.gz", [
        {"artist": a, "title": t, "key": k, "mbid": m or "", "listeners": int(u)}
        for a, t, k, m, u in zip(rows["artist"], rows["title"], rows["key"], rows["mbid"].fillna(""), rows["users"])])
    write_ids(out / "song_neighbours.i32.gz", table)
    song_shape = table.shape

    tune = json.loads((TUNE / "summary.json").read_text())
    song = json.loads((SONGS / "summary.json").read_text())
    manifest = {
        "created": time.strftime("%Y-%m-%d"),
        "source": f"https://github.com/Xendoksia/music-recommendation-system (commit {commit})",
        "min_listeners": MIN_USERS,
        "artists": {"count": artist_shape[0], "neighbours": artist_shape[1],
                    "files": ["artists.json.gz", "artist_neighbours.i32.gz"],
                    "model": "item-item kNN, asymmetric cosine on log listens", "similarity": {
                        k: tune["best"][k] for k in ("beta", "shrink")},
                    "scoring": {"SEEDS": tune["best"]["seeds"], "RELATED": tune["best"]["related"],
                                "SEED_DECAY": tune["best"]["decay"]},
                    "test": {k: tune["test_best"][k] for k in ("recall@12", "hit@12", "ndcg@12")}},
        "songs": {"count": song_shape[0], "neighbours": song_shape[1],
                  "files": ["songs.json.gz", "song_neighbours.i32.gz"],
                  "model": "item-item kNN, cosine with shrinkage on log listens, version 1",
                  "similarity": {k: song["exported"]["knn"][k] for k in ("beta", "shrink")},
                  "scoring": dict(zip(("SEEDS", "RELATED", "SEED_DECAY"), song["exported"]["knn"]["scoring"])),
                  "test": {mode: {label(r): {k: r[k] for k in ("recall@12", "hit@12", "ndcg@12")}
                                  for r in block["rows"]} for mode, block in song["test"].items()}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in sorted(out.iterdir()):
        print(f"{f.name:28} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
