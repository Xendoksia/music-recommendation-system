"""
Part 1: how well each candidate data set covers the user's artists, before any model is built.

- ListenBrainz: measured through its popularity API (listeners and listens per MBID), no dump needed yet.
- Last.fm 360K and 1K: read from the downloaded archives in data/, matched by MBID and then by folded name.

Lookups are cached in data/ so the script can be run again cheaply. Writes reports/kapsam.md.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import truststore

from src import user_data
from src.user_data import fold

# Python's own certificate bundle rejects ListenBrainz's chain while Windows accepts it, so the system store is used
truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MBID_CACHE = DATA / "mbid_lookup.json"
LB_CACHE = DATA / "listenbrainz_popularity.json"
LASTFM_360K = DATA / "lastfm-dataset-360K.tar.gz"
LASTFM_1K = DATA / "lastfm-dataset-1K.tar.gz"
LASTFM_CACHE = DATA / "lastfm_coverage.json"

# An artist with fewer listeners than this gives a model too little to learn from
MIN_LISTENERS = 50
DECISION = "## Karar"
# MusicBrainz asks for a contact in the User-Agent: the app's public repository, as the app itself sends
USER_AGENT = "PlaylistTrackerResearch/0.1 ( https://github.com/Xendoksia/Spotify-Youtube--Track-Ur-Playlist )"

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


def loose(name: str) -> str:
    """fold, with curly apostrophes and quotes made straight: MusicBrainz writes "Guns N’ Roses"."""
    return fold(name).replace("’", "'").replace("‘", "'").replace("`", "'")


def _get_json(url: str, data: bytes | None = None) -> object:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(5):
        try:
            with urlopen(Request(url, data=data, headers=headers), timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            # 503 is MusicBrainz's rate limit: wait longer each time and try again
            if error.code != 503 or attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def resolve_mbids(artists: dict[str, dict]) -> dict[str, str | None]:
    """MBIDs for every user artist: the app's MusicBrainz cache first, then a MusicBrainz search for the rest
    (an exact folded name match with the best score), at most one request a second as MusicBrainz asks."""
    known = user_data.mbids()
    cache = _load(MBID_CACHE)
    for folded, artist in artists.items():
        if folded in known or cache.get(folded):
            continue
        query = urlencode({"query": f'artist:"{artist["name"]}"', "fmt": "json", "limit": 10})
        results = _get_json(f"https://musicbrainz.org/ws/2/artist/?{query}").get("artists", [])
        same = [r for r in results if loose(r["name"]) == loose(folded)
                or any(loose(a.get("name", "")) == loose(folded) for a in r.get("aliases", []))]
        cache[folded] = max(same, key=lambda r: r.get("score", 0))["id"] if same else None
        _save(MBID_CACHE, cache)
        time.sleep(1.1)
    return {folded: known.get(folded) or cache.get(folded) for folded in artists}


def listenbrainz(ids: dict[str, str | None]) -> dict[str, dict]:
    """ListenBrainz listeners and listens by folded name, for the artists with an MBID."""
    cache = _load(LB_CACHE)
    missing = sorted({mbid for mbid in ids.values() if mbid and mbid not in cache})
    for start in range(0, len(missing), 50):
        body = json.dumps({"artist_mbids": missing[start:start + 50]}).encode()
        for row in _get_json("https://api.listenbrainz.org/1/popularity/artist", body):
            cache[row["artist_mbid"]] = {"users": row.get("total_user_count") or 0,
                                         "listens": row.get("total_listen_count") or 0}
        _save(LB_CACHE, cache)
    return {folded: cache.get(mbid, {"users": 0, "listens": 0}) if mbid else {"users": 0, "listens": 0}
            for folded, mbid in ids.items()}


def _tar_lines(path: Path, member_suffix: str):
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.name.endswith(member_suffix):
                stream = io.TextIOWrapper(archive.extractfile(member), encoding="utf-8", errors="replace")
                yield from stream
                return
    raise FileNotFoundError(f"{member_suffix} not in {path.name}")


def lastfm(ids: dict[str, str | None]) -> dict[str, dict]:
    """Listeners and plays per user artist in Last.fm 360K and 1K, matched by MBID or folded name. Both archives
    are streamed once and the counts cached, as reading them takes minutes."""
    cache = _load(LASTFM_CACHE)
    by_mbid = {mbid: folded for folded, mbid in ids.items() if mbid}
    wanted = set(ids)
    by_name = {loose(folded): folded for folded in ids}

    def match(mbid: str, name: str) -> str | None:
        if mbid in by_mbid:
            return by_mbid[mbid]
        return by_name.get(loose(name))

    if "360k" not in cache or set(cache["360k"]) != wanted:
        counts = {f: {"users": 0, "plays": 0} for f in wanted}
        # user-mboxsha1 \t musicbrainz-artist-id \t artist-name \t plays
        for line in _tar_lines(LASTFM_360K, "usersha1-artmbid-artname-plays.tsv"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            folded = match(parts[1], parts[2])
            if folded:
                counts[folded]["users"] += 1
                counts[folded]["plays"] += int(parts[3]) if parts[3].isdigit() else 0
        cache["360k"] = counts
        _save(LASTFM_CACHE, cache)

    if "1k" not in cache or set(cache["1k"]) != wanted:
        users: dict[str, set] = defaultdict(set)
        plays: dict[str, int] = defaultdict(int)
        # userid \t timestamp \t musicbrainz-artist-id \t artist-name \t musicbrainz-track-id \t track-name
        for line in _tar_lines(LASTFM_1K, "userid-timestamp-artid-artname-traid-traname.tsv"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            folded = match(parts[2], parts[3])
            if folded:
                users[folded].add(parts[0])
                plays[folded] += 1
        cache["1k"] = {f: {"users": len(users[f]), "plays": plays[f]} for f in wanted}
        _save(LASTFM_CACHE, cache)
    return cache


def is_turkish(folded: str) -> bool:
    info = user_data.artist_info()
    for name, entry in info.items():
        if fold(name) == folded and entry:
            areas = [entry.get("area") or {}, entry.get("begin_area") or {}]
            if any((a.get("names") or {}).get("en") in ("Turkey", "Türkiye") for a in areas):
                return True
    return False


def report() -> str:
    artists = user_data.artist_plays()
    ids = resolve_mbids(artists)
    lb = listenbrainz(ids)
    lfm = lastfm(ids) if LASTFM_360K.exists() and LASTFM_1K.exists() else {}
    turkish = {f for f in artists if is_turkish(f)}
    total_plays = sum(a["plays"] for a in artists.values())

    sources = {"ListenBrainz": lb}
    if lfm:
        sources["Last.fm 360K"] = lfm["360k"]
        sources["Last.fm 1K"] = lfm["1k"]
    thresholds = {"ListenBrainz": MIN_LISTENERS, "Last.fm 360K": MIN_LISTENERS, "Last.fm 1K": 5}

    def summary(group: set[str]) -> list[str]:
        group_plays = sum(artists[f]["plays"] for f in group) or 1
        rows = []
        for source, counts in sources.items():
            limit = thresholds[source]
            present = {f for f in group if counts[f]["users"] > 0}
            enough = {f for f in group if counts[f]["users"] >= limit}
            covered = sum(artists[f]["plays"] for f in enough)
            rows.append(f"| {source} | {len(present)}/{len(group)} | {len(enough)}/{len(group)} (≥{limit}) | "
                        f"%{100 * covered / group_plays:.0f} |")
        return rows

    head = "| Veri seti | Var | Yeterli sinyal | Dinlemelerin kapsanan payı |\n|---|---|---|---|"
    lines = [
        "# Parça 1: Kapsam ölçümü",
        "",
        f"Tarih: {time.strftime('%Y-%m-%d')} · Kullanıcı: {len(artists)} sanatçı, {total_plays} sanatçı çalması "
        f"(ortak şarkılar her sanatçıya ayrı sayılır) · MBID bulunan: {sum(1 for v in ids.values() if v)}"
        f"/{len(artists)} · Türk sanatçı: {len(turkish)}",
        "",
        "## Tüm sanatçılar", "", head, *summary(set(artists)), "",
        f"## Türk sanatçılar ({len(turkish)})", "", head, *summary(turkish), "",
        "## Sanatçı bazında (dinleyici sayıları)", "",
    ]
    columns = ["Sanatçı", "Çalma", "TR", "MBID", *sources]
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for folded, artist in artists.items():
        cells = [artist["name"], str(artist["plays"]), "✓" if folded in turkish else "",
                 "✓" if ids[folded] else "–", *(f"{counts[folded]['users']:,}" for counts in sources.values())]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "kapsam.md"
    text = report()
    # The decision is written by hand under DECISION and kept when the tables are made again
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if DECISION in old:
        head, _, tables = text.partition("## Tüm sanatçılar")
        decision = old[old.index(DECISION):].partition("## Tüm sanatçılar")[0]
        text = head + decision + "## Tüm sanatçılar" + tables
    path.write_text(text, encoding="utf-8")
    print(text[:3000])
