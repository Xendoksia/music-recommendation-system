"""
Deezer's related artists for every artist the user has played, looked up the way the app does
(playlist_tracker/music/recommend.py: _find_id, then /artist/{id}/related). The app's own cache
(artwork/related.json) is never written: the lists go to data/deezer_related.json here. More than the app's 20 are
kept (LIMIT), so a longer list can be tried too, and the first 20 are exactly what the app would read.

    python -m src.deezer_fetch
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.coverage import USER_AGENT
from src.user_data import artist_plays, fold

API = "https://api.deezer.com"
LIMIT = 50
OUT = Path(__file__).resolve().parent.parent / "data" / "deezer_related.json"


def _get(url: str) -> dict:
    for attempt in range(5):
        with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        # Deezer answers its rate limit (50 requests in 5 seconds) with error code 4 inside a 200
        if (data.get("error") or {}).get("code") != 4:
            return data
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Deezer kept refusing {url}")


def find_id(name: str) -> int | None:
    """The app's _find_id: of the results with the same name apart from accents, the one with most fans."""
    results = _get(f"{API}/search/artist?" + urlencode({"q": name, "limit": 10})).get("data") or []
    same = [r for r in results if fold(r.get("name", "")) == fold(name)]
    return max(same, key=lambda r: r.get("nb_fan") or 0)["id"] if same else None


def main() -> None:
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    artists = artist_plays()
    for folded, artist in artists.items():
        if folded in cache:
            continue
        artist_id = find_id(artist["name"])
        found = (_get(f"{API}/artist/{artist_id}/related?limit={LIMIT}").get("data") or []) if artist_id else []
        cache[folded] = {"name": artist["name"], "id": artist_id,
                         "related": [{"id": a["id"], "name": a["name"], "fans": a.get("nb_fan") or 0} for a in found]}
        print(f"{artist['name']}: {'not on Deezer' if not artist_id else len(found)}")
        time.sleep(0.25)
    OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    known = sum(1 for e in cache.values() if e["related"])
    print(f"{len(cache)} artists, {known} with related artists -> {OUT}")


if __name__ == "__main__":
    main()
