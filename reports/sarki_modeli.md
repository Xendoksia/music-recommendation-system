# Şarkı önerisi: sonuçlar

Tarih: 2026-11-24. Kod `kaggle/songs`, `kaggle/song_stats` ve `kaggle/song_train` klasörlerinde. Ham sonuçlar
`kaggle/song_train/output/results.json` ve `summary.json` dosyalarında. Senin verinle yapılan kontrol
`reports/sarki.md` dosyasında.

## Özet

**Şarkı knn'i iki görevde de baştan konan ölçütü (%20 fark) açık farkla geçti.** 5.000 test kullanıcısında:

- **Bütün şarkılar:** recall@12 değeri en iyi rakipten %75 yüksek. En iyi rakip "kendi sanatçılarının dinlenmemiş
  en popüler şarkıları" yöntemi.
- **Keşif** (yalnızca hiç dinlenmemiş sanatçıların şarkıları, "Senin için" ekranının amacı): recall@12 değeri
  uygulamanın şu anki yönteminden 2,7 kat yüksek. Uygulamanın yöntemi, ayarlı sanatçı knn'inin önerdiği
  sanatçıların en popüler şarkılarını gösteriyor.

## Veri

| | |
|---|---:|
| Kaynak | İstatistik dökümü (tüm zamanlar, kullanıcı başına en çok 1.000 şarkı) ve 15 günlük günlük dökümler |
| Birleştirmeden önce | 108.427 kullanıcı, 80,5 milyon kullanıcı–şarkı çifti |
| Elemeden sonra (şarkı en az 20 dinleyici, kullanıcı en az 10 şarkı, kullanıcı başına en çok 500 şarkı) | 87.040 kullanıcı, 561.412 şarkı, 88.218 sanatçı, 29,0 milyon çift |
| Bölme | 2.000 doğrulama, 5.000 test kullanıcısı. Her birinin şarkılarının %20'si gizli |

## Test sonuçları (5.000 kullanıcı)

**Bütün şarkılar** (gizlenen herhangi bir şarkıyı bulmak)

| Yöntem | recall@12 | hit@12 | NDCG@12 | recall@20 | hit@20 |
|---|---:|---:|---:|---:|---:|
| **Şarkı knn** | **0,087** | **0,847** | **0,352** | **0,113** | **0,886** |
| Şarkı item2vec | 0,055 | 0,711 | 0,215 | 0,071 | 0,777 |
| Kendi sanatçılarının en popüler şarkıları | 0,050 | 0,804 | 0,248 | 0,069 | 0,853 |
| Uygulamanın şu anki yolu (sanatçı başına 1 şarkı) | 0,005 | 0,179 | 0,020 | 0,007 | 0,248 |
| Uygulamanın şu anki yolu (sanatçı başına 3 şarkı) | 0,004 | 0,132 | 0,015 | 0,006 | 0,200 |
| Popülerlik | 0,004 | 0,185 | 0,024 | 0,006 | 0,243 |

**Keşif** (yalnızca hiç dinlenmemiş sanatçıların gizli şarkıları, 4.760 kullanıcı)

| Yöntem | recall@12 | hit@12 | NDCG@12 | recall@20 | hit@20 |
|---|---:|---:|---:|---:|---:|
| **Şarkı knn** | **0,036** | **0,348** | **0,064** | **0,047** | **0,412** |
| Şarkı item2vec | 0,015 | 0,170 | 0,025 | 0,021 | 0,221 |
| Uygulamanın şu anki yolu (sanatçı başına 1 şarkı) | 0,013 | 0,188 | 0,022 | 0,020 | 0,261 |
| Uygulamanın şu anki yolu (sanatçı başına 3 şarkı) | 0,011 | 0,139 | 0,017 | 0,016 | 0,210 |
| Popülerlik | 0,005 | 0,086 | 0,010 | 0,008 | 0,125 |

Keşif görevinde şarkı knn'i kullanıcıların %35'ine ilk 12 öneride hiç dinlemediği bir sanatçının gerçekten
dinlediği bir şarkısını buluyor. Uygulamanın şu anki yolunda bu oran %14 (ekrandaki gibi sanatçı başına 3 şarkı).

## Seçilen ayar

- Şarkı knn: cosine (β = 0,5), shrinkage 100. Sanatçıdaki ayarlı puanlamanın aynısı: 30 başlangıç şarkısı, her
  birinin 50 benzeri, SEED_DECAY 0,75. Bu puanlama 10, 20, 0,5 ayarından her denemede belirgin biçimde iyi çıktı.
- Sanatçıda işe yarayan hafif popüler kayma (β = 0,4) şarkıda fark yaratmadı ve doğrulamada en iyisi β = 0,5 oldu.
- item2vec (15 günlük oturumlardan, 60 milyon şarkı dinlemesi, yaklaşık 51 dakika eğitim) knn'in açıkça
  gerisinde kaldı.
- Kontrol: β = 0,5 ile yeniden puanlanan komşular implicit'in cosine komşularıyla %100 aynı.

## Dikkat edilmesi gerekenler

- **Rakip yöntem bu ölçümde avantajlı.** Sanatçı modeli istatistik dökümünün bütün kullanıcılarıyla eğitildi. Buna
  test kullanıcılarının gizlenen şarkılarının sanatçıları da dahil. Şarkı modeli gizlenen şarkıları hiç görmedi.
  Buna rağmen şarkı knn'i kazandı.
- **Rakip yöntemin eşleşmesi eksik.** Şarkı sanatçılarının %78'i ada göre sanatçı modelinde bulundu. Kalanlar için
  uygulamanın yolu öneri üretemiyor. Uygulamada Deezer bu boşluğu kısmen doldurur.
- **"Kendi sanatçılarının en popüler şarkıları" güçlü bir rakip.** Hit@12'de şarkı knn'ine yakın (0,80 ve 0,85)
  ama recall ve NDCG'de geride. Bu yöntem keşif yapmıyor, yalnızca bilinen sanatçıların başka şarkılarını öneriyor.
- **Senin verinde tablo farklı** (`reports/sarki.md`, yalnızca sağlama). 185 şarkılık tek tek gizlemede "kendi
  sanatçılarının en popülerleri" yöntemi (hit@12 0,092) şarkı knn'inden (0,065) iyi çıktı. 3 haftalık ve az
  sanatçılı bir dinleme geçmişinde bu beklenen bir durum. Karar genel test kullanıcılarına göre veriliyor.
- **Veri tazeliği.** İstatistik dökümü kullanıcı başına yalnızca en çok dinlenen 1.000 şarkıyı tutuyor ve günlük
  dökümler 15 gün geriye gidiyor. Yeni çıkan şarkılar için model düzenli yenilenmeli (Kaggle'da ayda bir
  yeniden çalıştırma yeterli).
- **Dosya boyutu.** 561 bin şarkının 50'şer komşusu sıkıştırılmış hâlde 77 MB, şarkı tablosu 49 MB. Uygulamaya
  girerse en çok dinlenen 100–150 bin şarkıya ve 20 komşuya indirmek gerekir (yaklaşık 10 MB).

## Bugünkü örnek öneriler (senin verinle)

- **Şarkı knn (keşif):** System of a Down – Chop Suey!, We Came as Romans – Black Hole, Ice Nine Kills – A Grave
  Mistake, Disturbed – Down With the Sickness, System of a Down – Toxicity, The Plot in You – Left Behind, Stone
  Sour – Through Glass, Falling in Reverse – Voices in My Head, While She Sleeps – Silence Speaks, Trapt –
  Headstrong, System of a Down – Lonely Day, Ayla Dikmen – Anlamazdın.
- **Uygulamanın şu anki yolu:** Disturbed (3 şarkı), System of a Down (3), Falling in Reverse (3), The Offspring (3).
- **Şarkı knn (bütün şarkılar):** Çoğunlukla zaten dinlediğin sanatçıların başka şarkıları (Motionless in White –
  Reincarnate, Slipknot – Duality, Three Days Grace – Pain gibi).

## Karar (kullanıcıya bırakıldı)

Model uygulamaya eklenmedi. Seçenekler:

1. **Şarkı knn'ini "Senin için"e eklemek (önerilen).** Keşif modunda yeni sanatçıların şarkılarını doğrudan
   öneriyor ve şu anki yoldan 2,7 kat isabetli. Sanatçı kartlarındaki "en popüler 3 şarkı" yerine ya da yanında
   kullanılabilir.
2. **Ek olarak "Sevdiğin sanatçılardan" bölümü:** bütün şarkılar modu, bilinen sanatçıların dinlenmemiş
   şarkılarını isabetli biçimde öneriyor.
3. **Şimdilik eklememek:** sanatçı modeliyle Parça 5'e geçilir ve şarkı modeli sonraya kalır.
