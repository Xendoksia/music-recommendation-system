"""
Part 2, step 3: the user's artists in the ListenBrainz data actually downloaded on Kaggle, rather than in
ListenBrainz's own all time counts. Kaggle (kaggle/counts) gives listeners per artist, so the user's list never
leaves this computer. An artist is matched by its MBID and also by "n:" + its folded name, as listens without an
MBID were keyed that way. Writes reports/kapsam_listenbrainz.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src import coverage, user_data

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "listenbrainz" / "artist_counts.parquet"
REPORT = ROOT / "reports" / "kapsam_listenbrainz.md"
MIN_LISTENERS = 50
COLUMNS = ["stats_users", "stats_listens", "recent_users", "recent_listens"]


def main() -> str:
    counts = pd.read_parquet(COUNTS).set_index("key")
    mine = user_data.artist_plays()
    ids = coverage.resolve_mbids(mine)
    total = sum(a["plays"] for a in mine.values())

    rows = []
    for folded, artist in mine.items():
        keys = {ids.get(folded), "n:" + folded, "n:" + coverage.loose(folded)} - {None}
        found = counts.loc[counts.index.intersection(list(keys)), COLUMNS]
        # A name key and the MBID key may share listeners, so the larger of them is a safe lower bound for users
        rows.append({"name": artist["name"], "plays": artist["plays"], "mbid": bool(ids.get(folded)),
                     "stats_users": int(found["stats_users"].max()) if len(found) else 0,
                     "recent_users": int(found["recent_users"].max()) if len(found) else 0,
                     "recent_listens": int(found["recent_listens"].sum()) if len(found) else 0})
    table = pd.DataFrame(rows)

    def share(column: str, limit: int) -> str:
        enough = table[table[column] >= limit]
        return f"{len(enough)}/{len(table)} | %{100 * enough['plays'].sum() / total:.0f}"

    lines = [
        "# ListenBrainz alt kümesinde kapsam",
        "",
        f"Tarih: {time.strftime('%Y-%m-%d')} · Veri Kaggle'da hazırlandı: tüm zamanların istatistiği (101.313 "
        "kullanıcı) ve son 15 günün dökümleri (126 milyon dinleme). 5'ten az dinleyicisi olan sanatçılar sayım "
        "tablosuna alınmadığı için 0 görünüyor.",
        "",
        "| Veri | Sanatçı | Dinlemelerin kapsanan payı |",
        "|---|---|---|",
        f"| Tüm zamanların istatistiği, en az {MIN_LISTENERS} dinleyici | {share('stats_users', MIN_LISTENERS)} |",
        f"| Tüm zamanların istatistiği, en az 10 dinleyici | {share('stats_users', 10)} |",
        f"| Son 15 gün, en az {MIN_LISTENERS} dinleyici | {share('recent_users', MIN_LISTENERS)} |",
        f"| Son 15 gün, en az 10 dinleyici | {share('recent_users', 10)} |",
        "",
        "## Sanatçı bazında",
        "",
        "| Sanatçı | Çalma | MBID | İstatistikte dinleyici | Son 15 günde dinleyici | Son 15 günde dinleme |",
        "|---|---|---|---|---|---|",
    ]
    for row in table.itertuples():
        lines.append(f"| {row.name} | {row.plays} | {'✓' if row.mbid else '–'} | {row.stats_users:,} | "
                     f"{row.recent_users:,} | {row.recent_listens:,} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = main()
    REPORT.write_text(text, encoding="utf-8")
    print(text[:1500])
