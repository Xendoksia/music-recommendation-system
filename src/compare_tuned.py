"""
Part 4b, local half: the tuned kNN (kaggle/tune) against the kNN of Part 3 and Deezer on the user's own listening,
with the app's current scoring and the tuned one. Deezer gives at most 20 related artists whatever is asked.
Writes reports/ayar.md.

    python -m src.compare_tuned
"""

from __future__ import annotations

import json

import numpy as np

import src.compare as C

TUNE_DIR = C.ROOT / "kaggle" / "tune" / "output"
REPORT_PATH = C.ROOT / "reports" / "ayar.md"


def knn_source(path, cat: C.Catalogue, n: int) -> dict[str, list[dict]]:
    ids = np.load(path)["ids"]
    return {f: [{"key": cat.key(int(b)), "name": cat.names[b], "pop": int(cat.users[b])}
                for b in ids[a] if b >= 0 and b != a][:n] for f, a in cat.user_index.items()}


def main() -> None:
    summary = json.loads((TUNE_DIR / "summary.json").read_text())
    best = summary["best"]
    history = C.History()
    cat = C.Catalogue(history)
    deezer = C.deezer_source()
    scorings = {"şu anki (10, 20, 0,5)": (10, 20, 0.5),
                f"ayarlı ({best['seeds']}, {best['related']}, {best['decay']})".replace(".", ","):
                    (best["seeds"], best["related"], best["decay"])}
    cols = ["hit@10", "hit@12", "hit@20", "mrr", "anywhere", "n"]
    loo_rows, day_rows, places, popularity, today = {}, {}, {}, {}, {}
    for label, (seeds, related, decay) in scorings.items():
        C.SEEDS, C.RELATED, C.SEED_DECAY = seeds, related, decay
        sources = {"deezer": deezer,
                   "knn (Parça 3)": knn_source(C.MODEL_DIR / "neighbours_knn.npz", cat, related),
                   "knn ayarlı": knn_source(TUNE_DIR / "neighbours_knn_tuned.npz", cat, related)}
        loo, every = C.leave_one_out(history, sources)
        daily, _, _ = C.by_day(history, sources)
        weights, heard = history.weights(), history.heard()
        for name in sources:
            key = f"{name}, {label}"
            loo_rows[key], day_rows[key], places[key] = loo[name], daily[name], every[name]
            row = C.score(C.seeds_of(weights), sources[name], heard)[:C.SHOWN]
            today[key] = row
            known = [x for x in (cat.listeners(a["key"]) for a in row) if x is not None]
            popularity[key] = int(np.median(known)) if known else 0

    tuned_label = list(scorings)[1]
    new, old, dz = (f"knn ayarlı, {tuned_label}", f"knn (Parça 3), {list(scorings)[0]}", f"deezer, {tuned_label}")
    tests = [(f"ayarlı knn ve Deezer (ikisi de ayarlı puanlamayla)", new, dz),
             (f"ayarlı knn ve Parça 3'teki knn (şu anki puanlamayla)", new, old)]
    t_cur, t_best = summary["test_current"], summary["test_best"]
    lines = [
        "# Parça 4b: Ayar taraması",
        "",
        "Kaggle'daki `kaggle/tune/tune.py` knn benzerliğini ve uygulamanın puanlama ayarlarını 2.000 doğrulama "
        "kullanıcısında taradı (ölçüt NDCG@12). Benzerlik `dot / (|i|^(2(1−β)) · |j|^(2β))` biçiminde ve paydaya "
        "shrinkage ekleniyor. β = 0,5 cosine demek, büyük β popüler adayları cezalandırır, küçük β öne çıkarır.",
        "",
        f"**En iyi ayar:** β = {best['beta']}, shrinkage = {best['shrink']:g}, {best['seeds']} başlangıç sanatçısı, "
        f"her birinin {best['related']} benzeri, SEED_DECAY = {best['decay']}.",
        "",
        "- Popülerlik cezası işe yaramadı: β büyüdükçe bütün ölçütler düştü. En iyisi cosine'den biraz popüler "
        "yöne kaymış β = 0,4 oldu, 0,35–0,45 arası neredeyse düz bir tepe.",
        "- Shrinkage küçük bir katkı verdi. Kazancın büyük kısmı uygulamanın ayarlarından geldi: daha çok başlangıç "
        "sanatçısı ve daha yavaş azalan çeşitlilik cezası.",
        "",
        "## 8.518 test kullanıcısı",
        "",
        "| Ayar | recall@12 | hit@12 | NDCG@12 | recall@20 | hit@20 |\n|---|---:|---:|---:|---:|---:|",
        *[f"| {name} | {r['recall@12']:.3f} | {r['hit@12']:.3f} | {r['ndcg@12']:.3f} | {r['recall@20']:.3f} | "
          f"{r['hit@20']:.3f} |" for name, r in (("şu anki (cosine, 10, 20, 0,5)", t_cur), ("ayarlı", t_best))],
        "",
        f"NDCG@12 %{(t_best['ndcg@12'] / t_cur['ndcg@12'] - 1) * 100:.0f} arttı.",
        "",
        "## Kullanıcının verisi: tek tek gizleme",
        "",
        C.table(loo_rows, cols),
        "Eşleştirilmiş işaret testi (yalnız birinin bulduğu gizli sanatçılar):",
        "",
        *[f"- {title}, {k}'de: yalnız ilki {x}, yalnız ikincisi {y}, p = {p:.3f}" for title, a, b in tests for k in (12, 20)
          for x, y, p in [C.paired(places[a], places[b], k)]],
        "",
        "## Kullanıcının verisi: zamana göre",
        "",
        C.table(day_rows, cols),
        "## Popülerlik yanlılığı",
        "",
        "Bugünkü 12 önerinin ListenBrainz'deki ortanca dinleyici sayısı:",
        "",
        *[f"- {k}: " + f"{v:,}".replace(",", ".") for k, v in popularity.items()],
        "",
        f"**Ayarlı knn'in bugünkü önerileri:** " + ", ".join(a["name"] for a in today[new]) + ".",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {REPORT_PATH}")


if __name__ == "__main__":
    main()
