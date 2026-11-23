"""
Songs, local sanity check of kaggle/song_train on the user's own plays (not the deciding metric: the model is for
general use and is judged on ListenBrainz test users). Same rankers as on Kaggle:

- song kNN: the user's most played songs are the seeds of the app's scoring (B settings: 30, 50, 0.75)
- own artists' top songs: the most listened songs the user has not played, by the user's artists in turn
- the app today: the tuned artist kNN suggests new artists, each with its 3 most listened songs

Leave one out over the user's songs the model knows, and today's suggestions of each. Writes reports/sarki.md.

    python -m src.song_compare
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.song_coverage import song_key
from src.user_data import credit_parts, fold, plays

from src.compare import ROOT

TRAIN_DIR = ROOT / "kaggle" / "song_train" / "output"
TUNE_DIR = ROOT / "kaggle" / "tune" / "output"
REPORT = ROOT / "reports" / "sarki.md"
SEEDS, RELATED, DECAY = 30, 50, 0.75
SHOWN, KS = 12, (12, 20)


def app_rank(seeds, weights, related, ok, k, tie):
    shares = {}
    for s, w in zip(seeds, weights):
        candidates = [int(a) for a in related[s] if a >= 0 and ok(int(a))][:RELATED]
        for place, a in enumerate(candidates):
            entry = shares.setdefault(a, {})
            entry[s] = entry.get(s, 0.0) + w * (1 - place / max(len(candidates), RELATED))
    used = {s: 0 for s in seeds}
    ranked = []
    while shares and len(ranked) < k:
        best = max(shares, key=lambda a: (sum(v * DECAY ** used[s] for s, v in shares[a].items()), tie[a]))
        for s in shares.pop(best):
            used[s] += 1
        ranked.append(best)
    return ranked


def main() -> None:
    model = pd.read_parquet(TRAIN_DIR / "songs_model.parquet")
    neighbours = np.load(TRAIN_DIR / "neighbours_song_knn.npz")["ids"]
    popularity = model["users"].to_numpy()
    index = dict(zip(model["key"], model["s"]))
    artist_code, artist_keys = pd.factorize(model["artist"].map(fold))
    by_artist = pd.Series(np.arange(len(model))).groupby(artist_code).apply(
        lambda s: s.to_numpy()[np.argsort(-popularity[s.to_numpy()], kind="stable")])
    code_of_name = {k: c for c, k in enumerate(artist_keys)}

    artists = pd.read_parquet(TUNE_DIR / "artists.parquet")
    artist_neighbours = np.load(TUNE_DIR / "neighbours_knn_tuned.npz")["ids"]
    model_of_name = {}
    for a, name, users in sorted(zip(artists["a"], artists["name"], artists["users"]), key=lambda x: x[2]):
        model_of_name[fold(name)] = int(a)
    name_of_model = {a: n for n, a in model_of_name.items()}

    # The user's plays per song key and per artist
    song_plays, artist_plays, names = Counter(), Counter(), {}
    for row in plays():
        k = song_key(row["artist"], row["name"])
        song_plays[k] += 1
        artist_plays[fold(credit_parts(row["artist"])[0])] += 1
        names.setdefault(k, f"{row['artist']} – {row['name']}")
    known_keys = [k for k in song_plays if k in index]

    def song_knn(played: dict[int, int], heard_artists: set[str], discovery: bool) -> list[int]:
        seeds = sorted(played, key=lambda s: -played[s])[:SEEDS]
        def ok(s):
            return s not in played and not (discovery and artist_keys[artist_code[s]] in heard_artists)
        return app_rank(seeds, [played[s] for s in seeds], neighbours, ok, max(KS), popularity)

    def own_artists(played: dict[int, int], artist_weight: Counter) -> list[int]:
        lists = [iter(by_artist[code_of_name[a]]) for a, _ in artist_weight.most_common(30) if a in code_of_name]
        out = []
        while lists and len(out) < max(KS):
            for it in list(lists):
                s = next((int(x) for x in it if int(x) not in played), None)
                if s is None:
                    lists.remove(it)
                else:
                    out.append(s)
        return out[:max(KS)]

    def app_today(played: dict[int, int], artist_weight: Counter) -> list[int]:
        seeds = [(model_of_name[a], w) for a, w in artist_weight.most_common() if a in model_of_name][:SEEDS]
        known = {model_of_name[a] for a in artist_weight if a in model_of_name}
        ranked = app_rank([s for s, _ in seeds], [w for _, w in seeds], artist_neighbours,
                          lambda a: a not in known and name_of_model.get(a) in code_of_name, max(KS),
                          np.zeros(len(artist_neighbours)))
        out = []
        for a in ranked:
            out.extend(int(s) for s in by_artist[code_of_name[name_of_model[a]]][:3] if int(s) not in played)
        return out[:max(KS)]

    played_all = {index[k]: n for k, n in song_plays.items() if k in index}
    heard = set(artist_plays)
    rankers = {
        "şarkı knn": lambda p, aw, h: song_knn(p, h, False),
        "şarkı knn (keşif)": lambda p, aw, h: song_knn(p, h, True),
        "kendi sanatçılarının en popülerleri": lambda p, aw, h: own_artists(p, aw),
        "uygulamanın şu anki yolu": lambda p, aw, h: app_today(p, aw),
    }
    places = {name: [] for name in rankers}
    for k in known_keys:
        s = index[k]
        played = {x: n for x, n in played_all.items() if x != s}
        artist = artist_keys[artist_code[s]]
        weight = artist_plays.copy()
        weight[artist] -= song_plays[k]
        weight = Counter({a: w for a, w in weight.items() if w > 0})
        heard_now = set(weight)
        for name, ranker in rankers.items():
            ranked = ranker(played, weight, heard_now)
            places[name].append(ranked.index(s) + 1 if s in ranked else None)

    lines = [
        "# Şarkı önerisi: kullanıcının verisiyle kontrol",
        "",
        "Karar ölçütü değil, yalnızca sağlama. Kullanıcının modelin tanıdığı "
        f"{len(known_keys)}/{len(song_plays)} şarkısı sırayla gizlendi. Keşif yöntemleri gizlenen şarkının sanatçısı "
        "hâlâ dinlenmiş sayıldığında onu bulamaz, bu yüzden burada düşük çıkmaları beklenir.",
        "",
        "| Yöntem | hit@12 | hit@20 |\n|---|---:|---:|",
    ]
    for name, found in places.items():
        cells = [f"{np.mean([bool(p and p <= k) for p in found]):.3f}" for k in KS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Bugünkü öneriler", ""]
    for name, ranker in rankers.items():
        ranked = ranker(played_all, artist_plays, heard)[:SHOWN]
        lines.append(f"**{name}:** " + ", ".join(f"{model['artist'][s]} – {model['title'][s]}" for s in ranked) + ".")
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
