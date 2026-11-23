# Parça 4: Deezer ile karşılaştırma

Tarih: 2026-11-23. Veri: 765 çalma, 102 sanatçı (01.11–23.11.2026). Bütün kaynaklar uygulamanın `score()` kuralıyla çalıştırıldı: en çok dinlenen 10 başlangıç sanatçısı, her birinin 20 benzeri, SEED_DECAY 0.5. Deezer listeleri `src/deezer_fetch.py` ile uygulamanın yöntemiyle çekildi (`data/deezer_related.json`) ve uygulamanın önbelleğine dokunulmadı.

## Ölçüm 1: Tek tek gizleme

Her sanatçı sırayla geçmişten çıkarıldı ve kalan geçmişle yapılan önerilerde kaçıncı sırada çıktığına bakıldı. `anywhere` sanatçının listenin herhangi bir yerinde çıkma oranı, yani başlangıç sanatçılarının benzer listelerinden en az birinde geçme oranı.

| Kaynak | hit@10 | hit@12 | hit@20 | mrr | anywhere | n |
|---|---:|---:|---:|---:|---:|---:|
| popülerlik | 0.010 | 0.010 | 0.029 | 0.004 | 0.029 | 102 |
| deezer | 0.039 | 0.059 | 0.108 | 0.032 | 0.284 | 102 |
| knn | 0.118 | 0.127 | 0.206 | 0.053 | 0.373 | 102 |
| als | 0.088 | 0.088 | 0.118 | 0.020 | 0.284 | 102 |
| item2vec | 0.098 | 0.098 | 0.137 | 0.040 | 0.284 | 102 |

Örnek küçük (102 sanatçı), bu yüzden her model Deezer ile aynı sanatçılar üzerinde eşleştirilmiş işaret testiyle karşılaştırıldı (yalnız birinin bulduğu gizli sanatçılar sayılır):

- knn, 12'de: yalnız knn 9, yalnız Deezer 2, p = 0.065
- knn, 20'de: yalnız knn 13, yalnız Deezer 3, p = 0.021
- als, 12'de: yalnız als 8, yalnız Deezer 5, p = 0.581
- als, 20'de: yalnız als 8, yalnız Deezer 7, p = 1.000
- item2vec, 12'de: yalnız item2vec 7, yalnız Deezer 3, p = 0.344
- item2vec, 20'de: yalnız item2vec 6, yalnız Deezer 3, p = 0.508

`knn, yoksa deezer` knn ile aynı sonucu veriyor, çünkü knn'in tanımadığı 3 sanatçı başlangıç listesine girmiyor.

Karma çözümler:

| Kaynak | hit@10 | hit@12 | hit@20 | mrr | anywhere | n |
|---|---:|---:|---:|---:|---:|---:|
| knn+deezer α=0.25 | 0.059 | 0.078 | 0.118 | 0.036 | 0.324 | 102 |
| knn+deezer α=0.5 | 0.088 | 0.108 | 0.157 | 0.037 | 0.373 | 102 |
| knn+deezer α=0.75 | 0.137 | 0.157 | 0.186 | 0.048 | 0.382 | 102 |
| knn+deezer RRF | 0.098 | 0.118 | 0.186 | 0.036 | 0.373 | 102 |
| knn, yoksa deezer | 0.118 | 0.127 | 0.206 | 0.053 | 0.373 | 102 |
| als+deezer α=0.25 | 0.049 | 0.059 | 0.098 | 0.033 | 0.294 | 102 |
| als+deezer α=0.5 | 0.078 | 0.088 | 0.127 | 0.019 | 0.294 | 102 |
| als+deezer α=0.75 | 0.078 | 0.088 | 0.127 | 0.019 | 0.294 | 102 |
| als+deezer RRF | 0.059 | 0.078 | 0.127 | 0.018 | 0.294 | 102 |
| als, yoksa deezer | 0.088 | 0.088 | 0.118 | 0.020 | 0.284 | 102 |

## Ölçüm 2: Zamana göre

Her gün için o günden önceki dinlemeler başlangıç oldu ve o gün ilk kez dinlenen sanatçılar arandı. 16 günde 85 yeni sanatçı var. Bunların çoğu öneriden değil başka yollardan (çalma listeleri, arkadaş tavsiyesi) geldiği için oranların düşük olması beklenir.

| Kaynak | hit@10 | hit@12 | hit@20 | mrr | anywhere | n |
|---|---:|---:|---:|---:|---:|---:|
| popülerlik | 0.000 | 0.000 | 0.024 | 0.001 | 0.024 | 85 |
| deezer | 0.035 | 0.047 | 0.047 | 0.020 | 0.200 | 85 |
| knn | 0.047 | 0.059 | 0.082 | 0.023 | 0.259 | 85 |
| als | 0.035 | 0.047 | 0.059 | 0.010 | 0.212 | 85 |
| item2vec | 0.035 | 0.035 | 0.047 | 0.021 | 0.235 | 85 |
| knn+deezer α=0.25 | 0.047 | 0.047 | 0.071 | 0.022 | 0.224 | 85 |
| knn+deezer α=0.5 | 0.035 | 0.047 | 0.094 | 0.022 | 0.247 | 85 |
| knn+deezer α=0.75 | 0.047 | 0.047 | 0.106 | 0.023 | 0.259 | 85 |
| knn+deezer RRF | 0.035 | 0.047 | 0.071 | 0.021 | 0.247 | 85 |
| knn, yoksa deezer | 0.047 | 0.059 | 0.082 | 0.023 | 0.259 | 85 |
| als+deezer α=0.25 | 0.024 | 0.035 | 0.047 | 0.019 | 0.200 | 85 |
| als+deezer α=0.5 | 0.012 | 0.024 | 0.071 | 0.009 | 0.212 | 85 |
| als+deezer α=0.75 | 0.035 | 0.035 | 0.071 | 0.009 | 0.224 | 85 |
| als+deezer RRF | 0.012 | 0.012 | 0.059 | 0.008 | 0.212 | 85 |
| als, yoksa deezer | 0.035 | 0.047 | 0.059 | 0.010 | 0.212 | 85 |

## Ölçüm 3: Genel veri setindeki test kullanıcıları (Kaggle)

8.518 kullanıcı, her birinin sanatçılarının %20'si gizli. Deezer burada ölçülemiyor.

| Model | recall@12 | hit@12 | NDCG@12 | recall@20 |
|---|---:|---:|---:|---:|
| popularity | 0.016 | 0.519 | 0.111 | 0.020 |
| knn | 0.069 | 0.791 | 0.316 | 0.092 |
| als | 0.055 | 0.749 | 0.272 | 0.077 |
| item2vec | 0.034 | 0.639 | 0.171 | 0.046 |

## Nitel ölçütler (bugünkü 12 öneri)

Kullanıcının sanatçılarından 35 tanesi MusicBrainz'e göre Türk. Ortanca dinleyici sayısı ListenBrainz'deki dinleyici sayısıdır ve popülerlik yanlılığını gösterir. `farklı başlangıç`, 12 önerinin kaç farklı başlangıç sanatçısından geldiğini sayar.

| Kaynak | tanıdığı sanatçı | Türk sanatçı (12'de) | ortanca LB dinleyicisi | farklı başlangıç | Deezer ile ortak |
|---|---:|---:|---:|---:|---:|
| deezer | 102/102 | 2 | 3414 | 8 | 12 |
| knn | 99/102 | 4 | 3456 | 7 | 3 |
| als | 99/102 | 3 | 4874 | 6 | 4 |
| item2vec | 91/102 | 4 | 1782 | 6 | 4 |
| knn, yoksa deezer | 102/102 | 4 | 3456 | 7 | 3 |

**Hız ve boyut:** Deezer her başlangıç sanatçısı için iki istek ister ve internetsiz çalışmaz. Modeller yerel dosyadan okunur. 50 komşulu dosyalar: knn 59 MB, als 48 MB, item2vec 25 MB. Uygulamada yalnızca 20 komşu ve kullanılan sanatçılar tutulursa dosya birkaç MB'a iner.

## Bugünkü öneriler

Başlangıç sanatçıları: Three Days Grace, Motionless In White, Linkin Park, Evanescence, Slipknot, Göksel, Thousand Foot Krutch, Burcu Güneş, STARSET, Asking Alexandria.

**deezer:** Hoobastank, I Prevail, Nothing More, Limp Bizkit, Mehmet Erdem (TR), Hollywood Undead, Disturbed, Ice Nine Kills, Thirty Seconds to Mars, Kenan Dogulu (TR), Nine Lashes, KoЯn.

**knn:** Disturbed, I Prevail, System of a Down, Falling in Reverse, Teoman (TR), Deniz Seki (TR), Adelitas Way, Nickelback, We Came as Romans, Limp Bizkit, Pinhani (TR), Ziynet Sali (TR).

**als:** Disturbed, I Prevail, System of a Down, Falling in Reverse, Thirty Seconds to Mars, Haluk Levent (TR), Muse, Erdem Kınay (TR), Limp Bizkit, We Came as Romans, Teoman (TR), Berkay.

**item2vec:** Disturbed, I Prevail, Drowning Pool, Catch Your Breath, Ayla Dikmen (TR), System of a Down, Bora Duran (TR), Nine Lashes, Falling in Reverse, Barış Akarsu (TR), İkilem (TR), Nothing More.

**knn, yoksa deezer:** Disturbed, I Prevail, System of a Down, Falling in Reverse, Teoman (TR), Deniz Seki (TR), Adelitas Way, Nickelback, We Came as Romans, Limp Bizkit, Pinhani (TR), Ziynet Sali (TR).

### Türk bir başlangıç sanatçısının benzerleri: Göksel

**deezer:** Sertab Erener, Yalın, Sezen Aksu, Nil Karaibrahimgil, Kenan Dogulu, Yaşar, Emre Altug, Mehmet Erdem, Candan Erçetin, Nazan Öncel.

**knn:** Şebnem Ferah, Teoman, Pinhani, Yüksek Sadakat, Emre Aydın, Mabel Matiz, Nilüfer, Sezen Aksu, Vega, Yaşar.

**als:** Haluk Levent, Teoman, Kenan Doğulu, Nilüfer, Ozan Doğulu, Kargo, Gripin, Ferdi Özbeğen, Gökhan Türkmen, Umut Kaya.

**item2vec:** Ayla Dikmen, Barış Akarsu, Nesrin Sipahi, Azer Bülbül, Cem Kısmet - Pilli Bebek, Mazhar Alanson, Ayten Alpman, Kayahan, Funda Sevilay, Sara Qədimova.

## Kör karşılaştırma

İki liste kaynağı söylenmeden kullanıcıya gösterilecek. Hangisinin hangisi olduğu `data/kor_karsilastirma.json` dosyasında duruyor.

**A:** Disturbed, I Prevail, System of a Down, Falling in Reverse, Teoman, Deniz Seki, Adelitas Way, Nickelback, We Came as Romans, Limp Bizkit, Pinhani, Ziynet Sali.

**B:** Hoobastank, I Prevail, Nothing More, Limp Bizkit, Mehmet Erdem, Hollywood Undead, Disturbed, Ice Nine Kills, Thirty Seconds to Mars, Kenan Dogulu, Nine Lashes, KoЯn.

## Karar

**Model, yoksa Deezer** (`knn`, tanımadığı sanatçıda Deezer'a düşer).

- Kör karşılaştırmada kullanıcı A listesini seçti (2026-11-23). A, "knn, yoksa Deezer" listesiydi ve B, Deezer'dı.
- Ölçüm 1'de knn Deezer'ın yaklaşık iki katı isabet verdi (hit@20 0,206 ve 0,108, eşleştirilmiş testte p = 0,02).
  Ölçüm 2 ve Ölçüm 3 aynı sıralamayı gösteriyor.
- Ağırlıklı ve RRF karışımları knn'i belirgin biçimde geçmedi, bu yüzden en basit karma çözüm seçildi: knn birincil
  kaynak, Deezer ise knn'in tanımadığı sanatçılar için yedek. Deezer ayrıca fotoğraf ve en iyi şarkılar için kalıyor.
- Uygulamada `RECOMMEND_SOURCE` varsayılanı `hybrid` olacak, `deezer` ve `model` seçenekleri de duracak.
- Örnek küçük (102 sanatçı ve 3 haftalık veri). Kararın canlı doğrulaması Parça 6'da yapılacak.
