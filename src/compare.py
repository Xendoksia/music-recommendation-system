"""
Part 4: Deezer against the models trained on Kaggle, on the user's own listening, with the app's own scoring.

Every source only gives what the app's related() gives: for an artist, up to RELATED similar artists, best first.
The seeds (the SEEDS artists played most lately, each play halved every HALF_LIFE_DAYS days), the scoring and
SEED_DECAY are copied from playlist_tracker/music/recommend.py, so a source wins only by its lists.

Measure 1, leave one out: each of the user's artists is taken out of the history in turn, the rest give the seeds,
and the place of the hidden artist in the suggestions is read (HitRate@10/12/20, MRR).
Measure 2, by time: for every day, the plays before it give the seeds, and the artists first played that day
should come up.
Measure 3 (the dataset's test users) was run on Kaggle and is copied from kaggle/train/output/results.json.

Writes reports/karsilastirma.md, and data/kor_karsilastirma.json with which blind list is which.

    python -m src.compare
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.coverage import _get_json, loose
from src.user_data import artist_info, credit_parts, credited, known_artists, mbids, plays

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "kaggle" / "train" / "output"
DEEZER_PATH = ROOT / "data" / "deezer_related.json"
COUNTRY_PATH = ROOT / "data" / "mb_country.json"
BLIND_PATH = ROOT / "data" / "kor_karsilastirma.json"
REPORT_PATH = ROOT / "reports" / "karsilastirma.md"

# The app's settings (recommend.py)
SEEDS = 10
HALF_LIFE_DAYS = 30
RELATED = 20
SEED_DECAY = 0.5
SHOWN = 12
KS = (10, 12, 20)
MODELS = ("knn", "als", "item2vec")
ALPHAS = (0.25, 0.5, 0.75)
RRF_K = 60


def utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- the user's history


class History:
    """The user's plays, with each play's credited artists by loose name, as the app counts them."""

    def __init__(self) -> None:
        self.rows = plays()
        solo = known_artists(r["artist"] for r in self.rows)
        self.names: dict[str, str] = {}
        self.credits: list[tuple[datetime, list[str]]] = []
        for r in self.rows:
            parts = []
            for part in credited(r["artist"], solo):
                self.names.setdefault(loose(part), part)
                parts.append(loose(part))
            self.credits.append((utc(r["played_at"]), parts))
        # The app's heard(): every part of every credit, the others of a collaboration too
        self.heard_at: list[tuple[datetime, set[str]]] = [
            (utc(r["played_at"]), {loose(p) for p in credit_parts(r["artist"])}) for r in self.rows]
        self.last = self.credits[-1][0]

    def weights(self, before: datetime | None = None, now: datetime | None = None) -> dict[str, float]:
        now = now or before or self.last
        out: dict[str, float] = defaultdict(float)
        for at, parts in self.credits:
            if before is not None and at >= before:
                continue
            age = max(0.0, (now - at).total_seconds() / 86400)
            for p in parts:
                out[p] += 0.5 ** (age / HALF_LIFE_DAYS)
        return dict(out)

    def heard(self, before: datetime | None = None) -> set[str]:
        return set().union(*(h for at, h in self.heard_at if before is None or at < before))

    def first_played(self) -> dict[str, datetime]:
        first: dict[str, datetime] = {}
        for at, parts in self.credits:
            for p in parts:
                first.setdefault(p, at)
        return first


def seeds_of(weights: dict[str, float], leave_out: str | None = None) -> list[tuple[str, float]]:
    ordered = sorted(((a, w) for a, w in weights.items() if a != leave_out), key=lambda x: -x[1])
    return ordered[:SEEDS]


# ---------------------------------------------------------------- the sources


class Catalogue:
    """The models' artists: ListenBrainz listeners, and the user's artists found by MBID, else by name."""

    def __init__(self, history: History) -> None:
        self.table = pd.read_parquet(MODEL_DIR / "artists.parquet")
        self.names = self.table["name"].tolist()
        self.users = self.table["users"].to_numpy()
        by_mbid = {m: a for a, m in zip(self.table["a"], self.table["mbid"]) if m}
        self.by_name: dict[str, int] = {}
        for a, name, users in sorted(zip(self.table["a"], self.names, self.users), key=lambda x: x[2]):
            self.by_name[loose(name)] = int(a)  # most listened last, so it wins a shared name
        ids = {loose(n): m for n, m in ((h, mb) for h, mb in mbids().items())}
        self.user_index: dict[str, int] = {}
        for folded in history.names:
            a = by_mbid.get(ids.get(folded, "")) if folded in ids else None
            if a is None:
                a = self.by_name.get(folded)
            if a is not None:
                self.user_index[folded] = int(a)
        self.user_of = {a: f for f, a in self.user_index.items()}

    def key(self, a: int) -> str:
        """An artist's name key: the user's own spelling for the user's artists, so MBID matches count."""
        return self.user_of.get(a) or loose(self.names[a])

    def listeners(self, name_key: str) -> int | None:
        a = self.user_index.get(name_key, self.by_name.get(name_key))
        return int(self.users[a]) if a is not None else None


def deezer_source() -> dict[str, list[dict]]:
    cache = json.loads(DEEZER_PATH.read_text(encoding="utf-8"))
    return {loose(e["name"]): [{"key": loose(r["name"]), "name": r["name"], "pop": r["fans"]}
                               for r in e["related"][:RELATED]] for e in cache.values()}


def model_source(name: str, cat: Catalogue) -> dict[str, list[dict]]:
    ids = np.load(MODEL_DIR / f"neighbours_{name}.npz")["ids"]
    out = {}
    for folded, a in cat.user_index.items():
        row = [int(b) for b in ids[a] if b >= 0 and b != a][:RELATED]
        out[folded] = [{"key": cat.key(b), "name": cat.names[b], "pop": int(cat.users[b])} for b in row]
    return out


def popularity_source(cat: Catalogue, everyone: set[str]) -> dict[str, list[dict]]:
    top = np.argsort(-cat.users)[:RELATED + SEEDS]
    row = [{"key": cat.key(int(b)), "name": cat.names[int(b)], "pop": int(cat.users[b])} for b in top]
    return {f: [r for r in row if r["key"] != f][:RELATED] for f in everyone}


def merged(lists: dict[str, list[dict]], score) -> list[dict]:
    """Several sources' lists for one artist in one, by score(source, place) summed, best RELATED first."""
    total: dict[str, float] = defaultdict(float)
    items: dict[str, dict] = {}
    for source, row in lists.items():
        for place, item in enumerate(row):
            total[item["key"]] += score(source, place)
            items.setdefault(item["key"], item)
    return [items[k] for k in sorted(total, key=lambda k: -total[k])[:RELATED]]


def hybrids(deezer: dict, model: dict, tag: str) -> dict[str, dict[str, list[dict]]]:
    everyone = set(deezer) | set(model)
    out = {}
    for alpha in ALPHAS:
        weight = {"model": alpha, "deezer": 1 - alpha}
        out[f"{tag}+deezer α={alpha}"] = {
            f: merged({"model": model.get(f, []), "deezer": deezer.get(f, [])},
                      lambda s, p, w=weight: w[s] * (1 - p / RELATED)) for f in everyone}
    out[f"{tag}+deezer RRF"] = {
        f: merged({"model": model.get(f, []), "deezer": deezer.get(f, [])}, lambda s, p: 1 / (RRF_K + p + 1))
        for f in everyone}
    out[f"{tag}, yoksa deezer"] = {f: model.get(f) or deezer.get(f, []) for f in everyone}
    return out


# ---------------------------------------------------------------- the app's scoring


def score(seeds: list[tuple[str, float]], related_of: dict[str, list[dict]], played: set[str]) -> list[dict]:
    """recommend.score(), with name keys for ids and the source's popularity for fans."""
    shares: dict[str, dict] = {}
    for seed, weight in seeds:
        candidates = related_of.get(seed) or []
        for place, artist in enumerate(candidates):
            if artist["key"] in played:
                continue
            entry = shares.setdefault(artist["key"], {**artist, "shares": {}})
            entry["shares"][seed] = entry["shares"].get(seed, 0.0) + weight * (1 - place / max(len(candidates), RELATED))
    used = {seed: 0 for seed, _ in seeds}
    ranked = []
    while shares:
        value = lambda a: sum(s * SEED_DECAY ** used[n] for n, s in a["shares"].items())
        best = max(shares.values(), key=lambda a: (value(a), a["pop"]))
        del shares[best["key"]]
        ranked.append({**best, "because": sorted(best["shares"], key=lambda n: -best["shares"][n])})
        for n in best["shares"]:
            used[n] += 1
    return ranked


def place_of(ranked: list[dict], target: str) -> int | None:
    return next((i + 1 for i, a in enumerate(ranked) if a["key"] == target), None)


def summarise(places: list[int | None]) -> dict:
    n = len(places)
    out = {f"hit@{k}": sum(1 for p in places if p and p <= k) / n for k in KS}
    out["mrr"] = sum(1 / p for p in places if p) / n
    out["anywhere"] = sum(1 for p in places if p) / n
    out["n"] = n
    return out


def leave_one_out(history: History, sources: dict) -> tuple[dict[str, dict], dict[str, list]]:
    weights = history.weights()
    heard = history.heard()
    out, every = {}, {}
    for name, related_of in sources.items():
        places = []
        for hidden in history.names:
            ranked = score(seeds_of(weights, hidden), related_of, heard - {hidden})
            places.append(place_of(ranked, hidden))
        out[name], every[name] = summarise(places), places
    return out, every


def paired(a: list, b: list, k: int) -> tuple[int, int, float]:
    """Hidden artists only a found in its first k, only b found, and the two-sided sign test's p value."""
    from scipy.stats import binomtest
    only_a = sum(1 for x, y in zip(a, b) if (x and x <= k) and not (y and y <= k))
    only_b = sum(1 for x, y in zip(a, b) if (y and y <= k) and not (x and x <= k))
    return only_a, only_b, binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0


def by_day(history: History, sources: dict) -> tuple[dict[str, dict], int, int]:
    first = history.first_played()
    start = history.credits[0][0].date() + timedelta(days=1)
    days = []
    day = start
    while day <= history.last.date():
        cut = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        new = [a for a, at in first.items() if cut <= at < cut + timedelta(days=1)]
        if new:
            days.append((cut, new))
        day += timedelta(days=1)
    cases = [(cut, history.weights(before=cut), history.heard(before=cut), new) for cut, new in days]
    out = {}
    for name, related_of in sources.items():
        places = []
        for cut, weights, heard, new in cases:
            ranked = score(seeds_of(weights), related_of, heard)
            places.extend(place_of(ranked, a) for a in new)
        out[name] = summarise(places)
    return out, len(days), sum(len(n) for _, _, _, n in cases)


# ---------------------------------------------------------------- what the lists look like


def countries(names: list[tuple[str, str | None]], known: dict[str, str]) -> dict[str, str]:
    """MusicBrainz country codes by name key: known ones first, then a lookup by MBID, else a name search."""
    cache = json.loads(COUNTRY_PATH.read_text(encoding="utf-8")) if COUNTRY_PATH.exists() else {}
    cache.update(known)
    for name, mbid in names:
        k = loose(name)
        if k in cache:
            continue
        try:
            if isinstance(mbid, str) and mbid:
                data = _get_json(f"https://musicbrainz.org/ws/2/artist/{mbid}?fmt=json")
            else:
                found = _get_json("https://musicbrainz.org/ws/2/artist?fmt=json&limit=1&query="
                                  + quote(f'artist:"{name}"')).get("artists") or []
                data = found[0] if found and found[0].get("score", 0) >= 90 else {}
            cache[k] = data.get("country") or ""
        except Exception:
            cache[k] = ""
        time.sleep(1.1)  # MusicBrainz allows one request a second
    COUNTRY_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache


def table(rows: dict[str, dict], cols: list[str]) -> str:
    head = "| Kaynak | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols) + "\n"
    body = ""
    for name, r in rows.items():
        cells = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        body += f"| {name} | " + " | ".join(cells) + " |\n"
    return head + body


def main() -> None:
    history = History()
    cat = Catalogue(history)
    deezer = deezer_source()
    models = {m: model_source(m, cat) for m in MODELS}
    sources = {"popülerlik": popularity_source(cat, set(history.names)), "deezer": deezer, **models}
    for m in ("knn", "als"):
        sources.update(hybrids(deezer, models[m], m))

    started = time.time()
    loo, loo_places = leave_one_out(history, sources)
    daily, n_days, n_new = by_day(history, sources)
    print(f"measured {len(sources)} sources in {time.time() - started:.0f} s")

    # Today's real suggestions from each main source, as the app would show them
    weights = history.weights()
    seeds = seeds_of(weights)
    heard = history.heard()
    best_hybrid = max((n for n in sources if "+" in n or "yoksa" in n), key=lambda n: (loo[n]["mrr"], loo[n]["hit@12"]))
    best_model = max(MODELS, key=lambda n: (loo[n]["mrr"], loo[n]["hit@12"]))
    shown_sources = ["deezer", *MODELS, best_hybrid]
    today = {n: score(seeds, sources[n], heard)[:SHOWN] for n in shown_sources}

    # The user's artists: the app's MusicBrainz cache has their country (area) already
    area = {loose(n): ((i or {}).get("area") or {}).get("names", {}).get("en", "") for n, i in artist_info().items()}
    known_country = {f: "TR" if area.get(f) in ("Turkey", "Türkiye") else "" for f in history.names if area.get(f)}
    mbid_of = dict(zip(cat.table["a"], cat.table["mbid"]))
    to_look = {(a["name"], mbid_of.get(cat.by_name[a["key"]]) if a["key"] in cat.by_name else None)
               for row in today.values() for a in row}
    to_look |= {(history.names[f], None) for f in history.names if f not in known_country}
    country = countries(sorted(to_look, key=lambda x: x[0]), known_country)

    user_tr = [f for f in history.names if country.get(f) == "TR"]
    known = {n: sum(1 for f in history.names if sources[n].get(f)) for n in shown_sources}
    qual = {}
    for n in shown_sources:
        row = today[n]
        listeners = [cat.listeners(a["key"]) for a in row]
        listeners = [x for x in listeners if x is not None]
        qual[n] = {
            "tanıdığı sanatçı": f"{known[n]}/{len(history.names)}",
            "Türk sanatçı (12'de)": sum(1 for a in row if country.get(a["key"]) == "TR"),
            "ortanca LB dinleyicisi": int(np.median(listeners)) if listeners else 0,
            "farklı başlangıç": len({a["because"][0] for a in row}),
            "Deezer ile ortak": len({a["key"] for a in row} & {a["key"] for a in today["deezer"]}),
        }
    tr_seed = max(user_tr, key=lambda f: weights.get(f, 0)) if user_tr else None

    # Blind lists: Deezer against the best of the rest, sides drawn at random
    rival = best_hybrid if loo[best_hybrid]["mrr"] >= loo[best_model]["mrr"] else best_model
    # Drawn once: the user has answered, so a rerun keeps the same sides
    if BLIND_PATH.exists():
        blind = json.loads(BLIND_PATH.read_text(encoding="utf-8"))
        sides = [blind["A"], blind["B"]]
    else:
        sides = ["deezer", rival]
        random.shuffle(sides)
        BLIND_PATH.write_text(json.dumps({"A": sides[0], "B": sides[1]}, ensure_ascii=False, indent=1), encoding="utf-8")

    sizes = {m: (MODEL_DIR / f"neighbours_{m}.npz").stat().st_size / 1e6 for m in MODELS}
    main_rows = ["popülerlik", "deezer", *MODELS]
    hybrid_rows = [n for n in sources if n not in main_rows]
    cols = ["hit@10", "hit@12", "hit@20", "mrr", "anywhere", "n"]
    test = [r for r in json.loads((MODEL_DIR / "results.json").read_text()) if r["split"] == "test"]

    lines = [
        "# Parça 4: Deezer ile karşılaştırma",
        "",
        f"Tarih: {datetime.now():%Y-%m-%d}. Veri: {len(history.rows)} çalma, {len(history.names)} sanatçı "
        f"({history.credits[0][0]:%d.%m}–{history.last:%d.%m.%Y}). Bütün kaynaklar uygulamanın `score()` kuralıyla "
        f"çalıştırıldı: en çok dinlenen {SEEDS} başlangıç sanatçısı, her birinin {RELATED} benzeri, SEED_DECAY "
        f"{SEED_DECAY}. Deezer listeleri `src/deezer_fetch.py` ile uygulamanın yöntemiyle çekildi "
        "(`data/deezer_related.json`) ve uygulamanın önbelleğine dokunulmadı.",
        "",
        "## Ölçüm 1: Tek tek gizleme",
        "",
        "Her sanatçı sırayla geçmişten çıkarıldı ve kalan geçmişle yapılan önerilerde kaçıncı sırada çıktığına "
        "bakıldı. `anywhere` sanatçının listenin herhangi bir yerinde çıkma oranı, yani başlangıç sanatçılarının "
        "benzer listelerinden en az birinde geçme oranı.",
        "",
        table({n: loo[n] for n in main_rows}, cols),
        "Örnek küçük (102 sanatçı), bu yüzden her model Deezer ile aynı sanatçılar üzerinde eşleştirilmiş işaret "
        "testiyle karşılaştırıldı (yalnız birinin bulduğu gizli sanatçılar sayılır):",
        "",
        *[f"- {m}, {k}'de: yalnız {m} {x}, yalnız Deezer {y}, p = {pv:.3f}" for m in MODELS for k in (12, 20)
          for x, y, pv in [paired(loo_places[m], loo_places["deezer"], k)]],
        "",
        "`knn, yoksa deezer` knn ile aynı sonucu veriyor, çünkü knn'in tanımadığı 3 sanatçı başlangıç listesine "
        "girmiyor.",
        "",
        "Karma çözümler:",
        "",
        table({n: loo[n] for n in hybrid_rows}, cols),
        "## Ölçüm 2: Zamana göre",
        "",
        f"Her gün için o günden önceki dinlemeler başlangıç oldu ve o gün ilk kez dinlenen sanatçılar arandı. "
        f"{n_days} günde {n_new} yeni sanatçı var. Bunların çoğu öneriden değil başka yollardan (çalma listeleri, "
        "arkadaş tavsiyesi) geldiği için oranların düşük olması beklenir.",
        "",
        table({n: daily[n] for n in main_rows + hybrid_rows}, cols),
        "## Ölçüm 3: Genel veri setindeki test kullanıcıları (Kaggle)",
        "",
        "8.518 kullanıcı, her birinin sanatçılarının %20'si gizli. Deezer burada ölçülemiyor.",
        "",
        "| Model | recall@12 | hit@12 | NDCG@12 | recall@20 |\n|---|---:|---:|---:|---:|",
        *[f"| {r['model']} | {r['recall@12']:.3f} | {r['hit@12']:.3f} | {r['ndcg@12']:.3f} | {r['recall@20']:.3f} |"
          for r in test],
        "",
        "## Nitel ölçütler (bugünkü 12 öneri)",
        "",
        f"Kullanıcının sanatçılarından {len(user_tr)} tanesi MusicBrainz'e göre Türk. Ortanca dinleyici sayısı "
        "ListenBrainz'deki dinleyici sayısıdır ve popülerlik yanlılığını gösterir. `farklı başlangıç`, 12 önerinin "
        "kaç farklı başlangıç sanatçısından geldiğini sayar.",
        "",
        table(qual, list(next(iter(qual.values())))),
        "**Hız ve boyut:** Deezer her başlangıç sanatçısı için iki istek ister ve internetsiz çalışmaz. Modeller "
        "yerel dosyadan okunur. 50 komşulu dosyalar: "
        + ", ".join(f"{m} {s:.0f} MB" for m, s in sizes.items())
        + ". Uygulamada yalnızca 20 komşu ve kullanılan sanatçılar tutulursa dosya birkaç MB'a iner.",
        "",
        "## Bugünkü öneriler",
        "",
        "Başlangıç sanatçıları: " + ", ".join(history.names[s] for s, _ in seeds) + ".",
        "",
    ]
    for n in shown_sources:
        lines.append(f"**{n}:** " + ", ".join(
            a["name"] + (" (TR)" if country.get(a["key"]) == "TR" else "") for a in today[n]) + ".")
        lines.append("")
    if tr_seed:
        lines.append(f"### Türk bir başlangıç sanatçısının benzerleri: {history.names[tr_seed]}")
        lines.append("")
        for n in ["deezer", *MODELS]:
            row = sources[n].get(tr_seed) or []
            lines.append(f"**{n}:** " + (", ".join(a["name"] for a in row[:10]) or "tanımıyor") + ".")
            lines.append("")
    lines += [
        "## Kör karşılaştırma",
        "",
        "İki liste kaynağı söylenmeden kullanıcıya gösterilecek. Hangisinin hangisi olduğu "
        "`data/kor_karsilastirma.json` dosyasında duruyor.",
        "",
        "**A:** " + ", ".join(a["name"] for a in score(seeds, sources[sides[0]], heard)[:SHOWN]) + ".",
        "",
        "**B:** " + ", ".join(a["name"] for a in score(seeds, sources[sides[1]], heard)[:SHOWN]) + ".",
        "",
        "## Karar",
        "",
    ]
    previous = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    decision = previous.split("## Karar\n", 1)[1].strip() if "## Karar\n" in previous else ""
    lines += [decision or "_Kör karşılaştırmanın sonucundan sonra yazılacak._", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"best model {best_model}, best hybrid {best_hybrid} -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
