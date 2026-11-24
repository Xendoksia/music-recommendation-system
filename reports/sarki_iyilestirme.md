# Şarkı önerisi: iyileştirme denemesi (adım 4)

Tarih: 2026-11-24. Kod `kaggle/song_improve/song_improve.py` dosyasında (Kaggle'da
`xendoksia/listenbrainz-song-improve`, 4. sürüm). Ham sonuçlar `kaggle/song_improve/output/results.json` ve
`summary.json` dosyalarında. Çalışma 2 saat 56 dakika sürdü.

## Özet

**En iyi yöntem, şarkı knn'i ile RP3beta'nın RRF birleşimi oldu.** Doğrulamada iki görevde de en iyi birleşim
buydu ve test kullanıcılarında da önde kaldı. 5.000 test kullanıcısında aynı veriyle eğitilen şu anki knn
ayarına göre şu farkları veriyor:

- **Keşif** ("Senin için" ekranının amacı): recall@12 %17, NDCG@12 %14 ve hit@12 %9 daha yüksek.
- **Bütün şarkılar:** recall@12 %7 ve NDCG@12 %6 daha yüksek. hit@12 ise yalnızca %1 arttı.

Daha fazla veri de işe yaradı ama kazancı küçük kaldı. item2vec, EASE ve "sanatçı sonra şarkı" yöntemleri knn'in
açıkça gerisinde kaldı.

## Veri

| | Önceki şarkı modeli | Bu deneme |
|---|---:|---:|
| Şarkının en az dinleyicisi | 20 | 10 |
| Kullanıcı başına en çok şarkı | 500 | 1.000 |
| Kullanıcı | 87.040 | 88.134 |
| Şarkı | 561.412 | 1.085.656 |
| Sanatçı | 88.218 | 160.170 |
| Kullanıcı–şarkı çifti | 29,0 M | 45,3 M |

Bölme aynı kuralla yapıldı: 2.000 doğrulama ve 5.000 test kullanıcısı seçildi ve her birinin şarkılarının %20'si
gizlendi. Kullanıcılar artık daha fazla şarkı taşıdığı için gizlenen şarkı sayısı da arttı. Bu yüzden bu tablodaki
recall değerleri önceki rapordaki değerlerle doğrudan karşılaştırılamaz. Karşılaştırma için aynı veride yeniden
eğitilen şu anki knn ayarı ("knn referans") kullanıldı.

## Test sonuçları (5.000 kullanıcı)

**Keşif** (yalnızca hiç dinlenmemiş sanatçıların gizli şarkıları, 4.765 kullanıcı)

| Yöntem | recall@12 | hit@12 | NDCG@12 | recall@20 | hit@20 |
|---|---:|---:|---:|---:|---:|
| **RRF: knn (400 komşu) + RP3beta** | **0,038** | **0,388** | **0,080** | **0,050** | **0,459** |
| RP3beta (α 0,5, β 0,6) | 0,034 | 0,367 | 0,073 | 0,044 | 0,434 |
| knn, 400 komşu | 0,033 | 0,352 | 0,073 | 0,044 | 0,423 |
| knn referans (şu anki ayar, 100 komşu) | 0,032 | 0,355 | 0,071 | 0,042 | 0,423 |
| knn, kullanıcı başına 500 şarkıyla | 0,030 | 0,334 | 0,067 | 0,039 | 0,394 |
| EASE (en popüler 15 bin şarkı) | 0,018 | 0,274 | 0,047 | 0,024 | 0,329 |
| item2vec | 0,015 | 0,173 | 0,028 | 0,020 | 0,226 |
| Sanatçı sonra şarkı | 0,011 | 0,192 | 0,023 | 0,017 | 0,269 |
| Uygulamanın şu anki yolu (sanatçı başına 1 şarkı) | 0,011 | 0,200 | 0,023 | 0,017 | 0,277 |
| Uygulamanın şu anki yolu (sanatçı başına 3 şarkı) | 0,008 | 0,140 | 0,017 | 0,013 | 0,216 |
| Popülerlik | 0,004 | 0,099 | 0,011 | 0,006 | 0,145 |

**Bütün şarkılar** (gizlenen herhangi bir şarkıyı bulmak)

| Yöntem | recall@12 | hit@12 | NDCG@12 | recall@20 | hit@20 |
|---|---:|---:|---:|---:|---:|
| **RRF: knn (400 komşu) + RP3beta** | **0,081** | **0,856** | **0,404** | **0,103** | **0,892** |
| RP3beta (α 0,5, β 0,6) | 0,077 | 0,844 | 0,377 | 0,098 | 0,880 |
| knn, 400 komşu | 0,076 | 0,847 | 0,383 | 0,097 | 0,884 |
| knn referans (şu anki ayar, 100 komşu) | 0,076 | 0,848 | 0,383 | 0,097 | 0,884 |
| knn, kullanıcı başına 500 şarkıyla | 0,073 | 0,838 | 0,375 | 0,094 | 0,875 |
| item2vec | 0,046 | 0,707 | 0,230 | 0,060 | 0,771 |
| Kendi sanatçılarının en popüler şarkıları | 0,043 | 0,808 | 0,285 | 0,060 | 0,849 |
| EASE (en popüler 15 bin şarkı) | 0,038 | 0,669 | 0,224 | 0,047 | 0,713 |
| Uygulamanın şu anki yolu (sanatçı başına 1 şarkı) | 0,004 | 0,190 | 0,021 | 0,006 | 0,264 |
| Popülerlik | 0,003 | 0,206 | 0,028 | 0,004 | 0,273 |

## Ne işe yaradı, ne yaramadı

- **RP3beta ile birleşim:** RP3beta tek başına knn'le hemen hemen aynı sonucu veriyor, ama iki yöntem farklı
  şarkıları buluyor. Bu yüzden listeleri RRF ile birleştirmek keşifte recall@12'yi %17 artırdı. Doğrulamada en iyi
  ayar α 0,5 ve β 0,6 oldu ve bu ayar popüler şarkıların ağırlığını düşürüyor.
- **Daha fazla veri:** Aynı kullanıcılarda kullanıcı başına 1.000 şarkı, 500 şarkıya göre recall@12'yi bütün
  şarkılarda %3, keşifte %9 artırdı. Dinleyici eşiğinin 10'a inmesiyle model artık 1,09 milyon şarkıyı tanıyor,
  yani önceki modelin yaklaşık iki katını.
- **400 komşuluk havuz:** Bütün şarkılarda fark yaratmadı. Keşifte ise NDCG@12'yi %3 artırdı.
- **Üç ya da dört yöntemi birleştirmek:** item2vec ya da EASE eklemek birleşimi iyileştirmedi, aksine kötüleştirdi.
- **EASE:** Bellek yüzünden yalnızca en popüler 15 bin şarkıda çalışabildi ve bu yüzden zayıf kaldı.
- **Sanatçı sonra şarkı:** Önce sanatçı knn'iyle yeni sanatçı seçip sonra o sanatçının şarkısını seçmek,
  uygulamanın şu anki yolu kadar zayıf kaldı. Şarkıyı doğrudan şarkı benzerliğiyle seçmek çok daha iyi çalışıyor.

## Uygulama açısından

- Kaggle çıktısında yalnızca en iyi tek tablo dışa aktarıldı. Bu tablo knn'in 400 komşulu hâli ve dosyası
  `neighbours_song_best.npz` (50 komşu, 153 MB). Ayrıca `neighbours_song_knn_big.npz` (200 komşu, 626 MB) ve
  `songs_model.parquet` (94 MB) dosyaları da çıktıda bulunuyor.
- RRF birleşimini uygulamada kullanmak için RP3beta tablosu da gerekiyor, ama o tablo dışa aktarılmadı. Bunun
  için kısa bir Kaggle çalışması daha gerekiyor. Bu çalışma item2vec'i atlayarak yalnızca veriyi, knn'i ve
  RP3beta'yı hesaplar ve tahminen 40 dakika sürer.
- Model iki kat daha fazla şarkı tanıdığı için uygulamaya girecek dosyalar da büyüyor. Birleşim seçilirse
  uygulama iki komşu tablosu taşır ve her öneride iki listeyi RRF ile birleştirir.
