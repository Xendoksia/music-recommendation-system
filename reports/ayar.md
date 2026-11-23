# Parça 4b: Ayar taraması

Kaggle'daki `kaggle/tune/tune.py` knn benzerliğini ve uygulamanın puanlama ayarlarını 2.000 doğrulama kullanıcısında taradı (ölçüt NDCG@12). Benzerlik `dot / (|i|^(2(1−β)) · |j|^(2β))` biçiminde ve paydaya shrinkage ekleniyor. β = 0,5 cosine demek, büyük β popüler adayları cezalandırır, küçük β öne çıkarır.

**En iyi ayar:** β = 0.4, shrinkage = 100, 30 başlangıç sanatçısı, her birinin 50 benzeri, SEED_DECAY = 0.75.

- Popülerlik cezası işe yaramadı: β büyüdükçe bütün ölçütler düştü. En iyisi cosine'den biraz popüler yöne kaymış β = 0,4 oldu, 0,35–0,45 arası neredeyse düz bir tepe.
- Shrinkage küçük bir katkı verdi. Kazancın büyük kısmı uygulamanın ayarlarından geldi: daha çok başlangıç sanatçısı ve daha yavaş azalan çeşitlilik cezası.

## 8.518 test kullanıcısı

| Ayar | recall@12 | hit@12 | NDCG@12 | recall@20 | hit@20 |
|---|---:|---:|---:|---:|---:|
| şu anki (cosine, 10, 20, 0,5) | 0.069 | 0.791 | 0.316 | 0.092 | 0.833 |
| ayarlı | 0.076 | 0.812 | 0.381 | 0.104 | 0.851 |

NDCG@12 %21 arttı.

## Kullanıcının verisi: tek tek gizleme

| Kaynak | hit@10 | hit@12 | hit@20 | mrr | anywhere | n |
|---|---:|---:|---:|---:|---:|---:|
| deezer, şu anki (10, 20, 0,5) | 0.039 | 0.059 | 0.108 | 0.032 | 0.284 | 102 |
| knn (Parça 3), şu anki (10, 20, 0,5) | 0.118 | 0.127 | 0.206 | 0.053 | 0.373 | 102 |
| knn ayarlı, şu anki (10, 20, 0,5) | 0.167 | 0.176 | 0.186 | 0.053 | 0.382 | 102 |
| deezer, ayarlı (30, 50, 0,75) | 0.069 | 0.069 | 0.118 | 0.025 | 0.510 | 102 |
| knn (Parça 3), ayarlı (30, 50, 0,75) | 0.108 | 0.127 | 0.186 | 0.064 | 0.696 | 102 |
| knn ayarlı, ayarlı (30, 50, 0,75) | 0.157 | 0.186 | 0.186 | 0.074 | 0.647 | 102 |

Eşleştirilmiş işaret testi (yalnız birinin bulduğu gizli sanatçılar):

- ayarlı knn ve Deezer (ikisi de ayarlı puanlamayla), 12'de: yalnız ilki 15, yalnız ikincisi 3, p = 0.008
- ayarlı knn ve Deezer (ikisi de ayarlı puanlamayla), 20'de: yalnız ilki 13, yalnız ikincisi 6, p = 0.167
- ayarlı knn ve Parça 3'teki knn (şu anki puanlamayla), 12'de: yalnız ilki 11, yalnız ikincisi 5, p = 0.210
- ayarlı knn ve Parça 3'teki knn (şu anki puanlamayla), 20'de: yalnız ilki 6, yalnız ikincisi 8, p = 0.791

## Kullanıcının verisi: zamana göre

| Kaynak | hit@10 | hit@12 | hit@20 | mrr | anywhere | n |
|---|---:|---:|---:|---:|---:|---:|
| deezer, şu anki (10, 20, 0,5) | 0.035 | 0.047 | 0.047 | 0.020 | 0.200 | 85 |
| knn (Parça 3), şu anki (10, 20, 0,5) | 0.047 | 0.059 | 0.082 | 0.023 | 0.259 | 85 |
| knn ayarlı, şu anki (10, 20, 0,5) | 0.059 | 0.071 | 0.106 | 0.019 | 0.259 | 85 |
| deezer, ayarlı (30, 50, 0,75) | 0.024 | 0.035 | 0.047 | 0.024 | 0.353 | 85 |
| knn (Parça 3), ayarlı (30, 50, 0,75) | 0.035 | 0.047 | 0.059 | 0.025 | 0.600 | 85 |
| knn ayarlı, ayarlı (30, 50, 0,75) | 0.035 | 0.047 | 0.094 | 0.026 | 0.565 | 85 |

## Popülerlik yanlılığı

Bugünkü 12 önerinin ListenBrainz'deki ortanca dinleyici sayısı:

- deezer, şu anki (10, 20, 0,5): 3.414
- knn (Parça 3), şu anki (10, 20, 0,5): 3.456
- knn ayarlı, şu anki (10, 20, 0,5): 4.874
- deezer, ayarlı (30, 50, 0,75): 3.414
- knn (Parça 3), ayarlı (30, 50, 0,75): 4.874
- knn ayarlı, ayarlı (30, 50, 0,75): 8.725

**Ayarlı knn'in bugünkü önerileri:** Disturbed, System of a Down, Falling in Reverse, The Offspring, I Prevail, Teoman, Nirvana, Pinhani, We Came as Romans, Fall Out Boy, Mabel Matiz, Foo Fighters.
