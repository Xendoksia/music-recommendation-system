"""
The user's own listening, read from Playlist Tracker's data folder. The app's files are opened read only and
nothing is ever written there. Artists are taken out of the plays by the app's own rule: a collaboration lists
its artists with commas or "feat.", while an ampersand is left alone, as bands such as "Simon & Garfunkel" carry
one in their name. The others of a collaboration count only when they are heard on their own too.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from pathlib import Path

APP_DIR = Path(os.environ.get("PLAYLIST_TRACKER_DATA") or Path(os.environ["APPDATA"]) / "Playlist Tracker")
DB_PATH = APP_DIR / "listening.db"
ARTIST_INFO_PATH = APP_DIR / "artwork" / "artist_info.json"
RELATED_PATH = APP_DIR / "artwork" / "related.json"

_CREDIT_SEPARATORS = re.compile(r",| feat\.? | ft\.? ", flags=re.IGNORECASE)


def key(text: str) -> str:
    """The app's artist key: case folded, spaces collapsed."""
    return " ".join((text or "").casefold().split())


def fold(text: str) -> str:
    """A name without case, accents or Turkish letters, so Göksel and "Goksel" match."""
    text = (text or "").replace("ı", "i").replace("İ", "i")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def credit_parts(artist: str) -> list[str]:
    return [part.strip() for part in _CREDIT_SEPARATORS.split(artist or "") if part.strip()] or [artist]


def known_artists(names) -> set[str]:
    return {key(credit_parts(name)[0]) for name in names}


def credited(artist: str, known: set[str]) -> list[str]:
    parts = credit_parts(artist)
    return [parts[0]] + [part for part in parts[1:] if key(part) in known]


def _open(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def plays(db_path: Path = DB_PATH) -> list[dict]:
    """Every recorded play, oldest first: {played_at, artist, name, album, track_key}."""
    with closing(_open(db_path)) as db:
        rows = db.execute("SELECT played_at, artist, name, album, track_key FROM listens ORDER BY played_at").fetchall()
    return [dict(zip(("played_at", "artist", "name", "album", "track_key"), row)) for row in rows]


def artist_plays(db_path: Path = DB_PATH) -> dict[str, dict]:
    """The user's artists by folded name: {fold: {name, key, plays, tracks}}. A collaboration's play counts for
    each artist it credits."""
    rows = plays(db_path)
    solo = known_artists(row["artist"] for row in rows)
    artists: dict[str, dict] = {}
    for row in rows:
        for part in credited(row["artist"], solo):
            entry = artists.setdefault(fold(part), {"name": part, "key": key(part), "plays": 0, "tracks": set()})
            entry["plays"] += 1
            entry["tracks"].add(row["track_key"])
    for entry in artists.values():
        entry["tracks"] = len(entry["tracks"])
    return dict(sorted(artists.items(), key=lambda item: -item[1]["plays"]))


def artist_info(path: Path = ARTIST_INFO_PATH) -> dict[str, dict]:
    """The app's MusicBrainz cache by artist key: {key: {id (MBID), genres, area, ...}}."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("artists", {})
    except (OSError, ValueError):
        return {}


def mbids(path: Path = ARTIST_INFO_PATH) -> dict[str, str]:
    """MusicBrainz ids by folded artist name, for the artists the app has looked up."""
    return {fold(name): info["id"] for name, info in artist_info(path).items() if info and info.get("id")}


def deezer_related(path: Path = RELATED_PATH) -> dict[str, list[dict]]:
    """Deezer's related artists from the app's cache by artist key, best first. Empty until the app has made its
    suggestions once."""
    try:
        cache = json.loads(path.read_text(encoding="utf-8")).get("artists", {})
    except (OSError, ValueError):
        return {}
    return {name: entry.get("related", []) for name, entry in cache.items()}


if __name__ == "__main__":
    artists = artist_plays()
    ids = mbids()
    total = sum(a["plays"] for a in artists.values())
    print(f"{len(plays())} plays, {len(artists)} artists, {total} artist credits, "
          f"{sum(1 for f in artists if f in ids)} with an MBID, Deezer cache: {len(deezer_related())} artists")
    for folded, a in list(artists.items())[:15]:
        print(f"{a['plays']:4} {a['name']}{'' if folded in ids else '  (no MBID)'}")
    print(Counter(len(credit_parts(p["artist"])) for p in plays()))
