"""
Songs, step 2 (the decision gate): how many of the user's songs ListenBrainz's daily dumps know, and with how many
listeners. The user's songs get the same keys as on Kaggle (kaggle/songs/songs.py): first credited artist and plain
title, folded. Only the counts table comes down from Kaggle (data/listenbrainz/song_counts.parquet), the user's
songs stay here. Writes reports/kapsam_sarki.md.

    python -m src.song_coverage
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.user_data import credit_parts, fold, plays

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "listenbrainz" / "song_counts.parquet"
SUMMARY = ROOT / "kaggle" / "songs" / "output" / "summary.json"
REPORT = ROOT / "reports" / "kapsam_sarki.md"
THRESHOLDS = (3, 5, 10, 20, 50)


def plain_title(name: str) -> str:
    """The app's _plain_title, as in kaggle/songs/songs.py."""
    name = name or ""
    return re.sub(r"\s*(\(.*?\)|\[.*?\])", "", name.split(" - ")[0]).strip() or name


def song_key(artist: str, title: str) -> str:
    return fold(credit_parts(artist)[0]) + "|" + fold(plain_title(title))


def user_songs() -> tuple[Counter, dict[str, str]]:
    """Plays per song key, and a readable name for each."""
    counts, names = Counter(), {}
    for row in plays():
        key = song_key(row["artist"], row["name"])
        counts[key] += 1
        names.setdefault(key, f"{row['artist']} – {row['name']}")
    return counts, names


def main() -> None:
    counts, names = user_songs()
    listeners = pd.read_parquet(COUNTS).set_index("key")["listeners"]
    total = sum(counts.values())
    found = {k: int(listeners.get(k, 0)) for k in counts}
    summary = json.loads(SUMMARY.read_text())["total"] if SUMMARY.exists() else {}
    lines = [
        "# Şarkı kapsamı",
        "",
        f"ListenBrainz günlük dökümlerinde {summary.get('listens', 0):,} dinleme, {summary.get('users', 0):,} "
        f"kullanıcı ve {summary.get('songs', 0):,} şarkı var. Kullanıcının {len(counts)} şarkısı ({total} çalma) "
        "aynı anahtarla (ilk sanatçı | sade şarkı adı) arandı.".replace(",", "."),
        "",
        "| En az dinleyici | Şarkı | Şarkı payı | Çalma payı |\n|---:|---:|---:|---:|",
    ]
    for n in THRESHOLDS:
        songs = [k for k, v in found.items() if v >= n]
        share = sum(counts[k] for k in songs) / total
        lines.append(f"| {n} | {len(songs)} | %{len(songs) / len(counts) * 100:.0f} | %{share * 100:.0f} |")
    if summary:
        lines += ["", "Genel veride bu eşiklerin üstündeki şarkı sayısı: " + ", ".join(
            f"{n}+ dinleyici {summary.get(f'songs_with_{n}_listeners', 0):,}".replace(",", ".") for n in THRESHOLDS)
            + "."]
    missing = sorted((k for k, v in found.items() if v < 5), key=lambda k: -counts[k])[:20]
    lines += ["", "## En çok dinlenip 5'ten az dinleyicisi olan şarkılar", "",
              *[f"- {names[k]} ({counts[k]} çalma, {found[k]} dinleyici)" for k in missing], ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
