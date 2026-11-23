# Parça 1: Kapsam ölçümü

Tarih: 2026-11-23 · Kullanıcı: 102 sanatçı, 763 sanatçı çalması (ortak şarkılar her sanatçıya ayrı sayılır) · MBID bulunan: 99/102 · Türk sanatçı: 28

## Karar

**Ana veri kaynağı ListenBrainz olacak.** Kullanıcının 102 sanatçısının 94'ünde en az 50 dinleyici var ve bu
sanatçılar dinlemelerin %97'sini karşılıyor. Türk sanatçılarda da kapsam %99. Last.fm veri setleri 2008–2009'dan
kalma olduğu için yeni gruplar bu verilerde hiç yok. STARSET, Bad Omens, Sleep Theory ve NF örneğin Last.fm'de sıfır
dinleyiciyle görünüyor, Motionless In White ise yalnızca 37 dinleyicide kalıyor. Last.fm 360K bu yüzden
dinlemelerin yalnızca %64'ünü, Last.fm 1K ise %57'sini karşılıyor.

**Last.fm 1K bırakılıyor.** Hem kapsamı en düşük veri seti hem de yalnızca 992 kullanıcısı var. Oturum verisi
(item2vec) ListenBrainz'in zaman damgalı dinlemelerinden de çıkarılabiliyor.

**Last.fm 360K yedekte kalıyor.** Klasik sanatçılarda (Metallica, Linkin Park, Sezen Aksu) güçlü bir sinyal taşıyor.
Ama yeni sanatçıları bilmediği için tek başına yetmiyor. ListenBrainz alt kümesinin kapsamı Parça 2'de beklenenden
düşük çıkarsa birleştirme denenecek.

**Modelin bilmediği sanatçılar için Deezer'a düşülecek.** Atiye, Furkan Kızılay ve Shotgun Parade'in MusicBrainz
kaydı bulunamadı. Hilal Bilen, Funda, Berk Baysal, Çağan Şengül ve Server Uraz ise 50'den az dinleyicide kalıyor.
Bu 8 sanatçı toplam dinlemelerin %3,3'ünü oluşturuyor ve bunun büyük kısmı Hilal Bilen'in 17 çalmasından geliyor.

### ListenBrainz'den hangi veri indirilecek?
Bu sayılar ListenBrainz'in tüm zamanlarına ait ve API'den alındı. Tam döküm 246 GB olduğu için diske sığmıyor.
Klasör D:'ye taşındıktan sonra (228 GB boş) iki seçenek de sığıyor:

| Seçenek | Boyut | İçerik | Artısı | Eksisi |
|---|---|---|---|---|
| Günlük artımlı dökümler (spark, parquet) | Günde yaklaşık 210 MB, 30 gün için yaklaşık 6 GB | Zaman damgalı, MBID'li ham dinlemeler | Oturumlar çıkarılabildiği için item2vec de denenebilir | Yalnızca son bir ayın dinleyicileri girer, bu yüzden az bilinen sanatçılarda sinyal zayıflar |
| İstatistik dökümü | 23 GB (sıkıştırılmış) | Kullanıcı başına en çok dinlenen sanatçılar | Tüm zamanları kapsar ve doğrudan kullanıcı × sanatçı matrisi verir | Sıra bilgisi olmadığı için item2vec için uygun değil |

Öneri, ikisini birlikte kullanmak. İstatistik dökümü ALS ve birlikte dinlenme modelleri için tüm zamanların
kullanıcı × sanatçı matrisini verecek. Günlük dökümler ise item2vec için gereken dinleme sıralarını sağlayacak.
Parça 2'nin ilk adımında indirilen veride kullanıcının sanatçılarının kapsamı yeniden ölçülecek.

### Bilinen eksikler
- Türk sanatçılar MusicBrainz'deki ülke bilgisine göre sayıldı. Ülkesi kayıtlı olmayan bazı Türk sanatçılar
  (Hilal Bilen, Simge, Yıldız Tilbe) bu yüzden "Türk sanatçılar" tablosuna girmedi. Gerçek sayı 28'den fazla.
- Hilal Bilen ve Funda için bulunan MBID büyük olasılıkla aynı adlı başka bir kayda ait, çünkü ikisinin de yalnızca
  5 dinleyicisi var. Parça 2'deki eşleştirmede bunlara elle bakılacak.

## Tüm sanatçılar

| Veri seti | Var | Yeterli sinyal | Dinlemelerin kapsanan payı |
|---|---|---|---|
| ListenBrainz | 99/102 | 94/102 (≥50) | %97 |
| Last.fm 360K | 67/102 | 54/102 (≥50) | %64 |
| Last.fm 1K | 59/102 | 49/102 (≥5) | %57 |

## Türk sanatçılar (28)

| Veri seti | Var | Yeterli sinyal | Dinlemelerin kapsanan payı |
|---|---|---|---|
| ListenBrainz | 28/28 | 27/28 (≥50) | %99 |
| Last.fm 360K | 19/28 | 15/28 (≥50) | %57 |
| Last.fm 1K | 17/28 | 14/28 (≥5) | %42 |

## Sanatçı bazında (dinleyici sayıları)

| Sanatçı | Çalma | TR | MBID | ListenBrainz | Last.fm 360K | Last.fm 1K |
|---|---|---|---|---|---|---|
| Three Days Grace | 71 |  | ✓ | 169,793 | 8,573 | 119 |
| Motionless In White | 66 |  | ✓ | 22,346 | 37 | 0 |
| Linkin Park | 47 |  | ✓ | 353,575 | 39,838 | 347 |
| Slipknot | 41 |  | ✓ | 213,485 | 18,694 | 194 |
| Evanescence | 40 |  | ✓ | 246,737 | 20,372 | 300 |
| Göksel | 29 | ✓ | ✓ | 3,020 | 385 | 15 |
| Thousand Foot Krutch | 26 |  | ✓ | 53,361 | 1,709 | 18 |
| Burcu Güneş | 26 | ✓ | ✓ | 928 | 95 | 3 |
| STARSET | 23 |  | ✓ | 6,700 | 0 | 0 |
| Asking Alexandria | 22 |  | ✓ | 49,881 | 66 | 0 |
| Metallica | 19 |  | ✓ | 341,352 | 45,301 | 398 |
| Skillet | 18 |  | ✓ | 92,953 | 3,328 | 25 |
| Hilal Bilen | 17 |  | ✓ | 5 | 0 | 0 |
| Bad Omens | 17 |  | ✓ | 8,744 | 0 | 0 |
| Pera | 15 | ✓ | ✓ | 116 | 0 | 0 |
| Breaking Benjamin | 15 |  | ✓ | 151,190 | 9,738 | 138 |
| Zero 9:36 | 14 |  | ✓ | 3,021 | 0 | 0 |
| Within Temptation | 11 |  | ✓ | 132,623 | 4,585 | 166 |
| SELIN | 10 | ✓ | ✓ | 176 | 0 | 0 |
| Güneş | 10 | ✓ | ✓ | 170 | 0 | 0 |
| Architects | 9 |  | ✓ | 50,732 | 1,093 | 14 |
| Apocalyptica | 8 |  | ✓ | 163,928 | 13,748 | 209 |
| emre aydın | 7 | ✓ | ✓ | 2,731 | 710 | 13 |
| Ninth Key | 7 |  | ✓ | 148 | 0 | 0 |
| My Darkest Days | 7 |  | ✓ | 14,756 | 2 | 0 |
| Avenged Sevenfold | 7 |  | ✓ | 167,531 | 7,641 | 121 |
| Şanışer | 7 | ✓ | ✓ | 147 | 1 | 0 |
| Perdenin Ardındakiler | 7 | ✓ | ✓ | 105 | 0 | 0 |
| Pierce The Veil | 6 |  | ✓ | 43,373 | 449 | 2 |
| Zeynep Casalini | 6 | ✓ | ✓ | 1,128 | 31 | 5 |
| Mark Eliyahu | 6 |  | ✓ | 538 | 0 | 0 |
| Cem Adrian | 6 | ✓ | ✓ | 1,822 | 391 | 10 |
| Bring Me The Horizon | 5 |  | ✓ | 102,237 | 4,511 | 18 |
| Five Finger Death Punch | 5 |  | ✓ | 76,275 | 1,170 | 10 |
| IVOXYGEN | 5 |  | ✓ | 2,085 | 0 | 0 |
| Sleep Theory | 5 |  | ✓ | 2,623 | 0 | 0 |
| Rammstein | 5 |  | ✓ | 233,586 | 27,200 | 315 |
| Shinedown | 5 |  | ✓ | 104,455 | 2,562 | 53 |
| Hande Yener | 5 | ✓ | ✓ | 3,627 | 785 | 20 |
| Zeynep Dizdar | 5 | ✓ | ✓ | 714 | 33 | 2 |
| Seether | 4 |  | ✓ | 131,856 | 5,475 | 121 |
| Cult To Follow | 4 |  | ✓ | 2,845 | 3 | 1 |
| NF | 4 |  | ✓ | 10,334 | 0 | 0 |
| Bullet For My Valentine | 4 |  | ✓ | 160,247 | 10,957 | 113 |
| Halsey | 4 |  | ✓ | 22,141 | 0 | 0 |
| Tarkan | 4 | ✓ | ✓ | 18,025 | 1,359 | 39 |
| Ben Fero | 4 | ✓ | ✓ | 155 | 0 | 0 |
| Bengü | 4 | ✓ | ✓ | 892 | 95 | 0 |
| Visions of Atlantis | 3 |  | ✓ | 20,967 | 1,058 | 19 |
| Our Lady Peace | 3 |  | ✓ | 69,171 | 2,141 | 108 |
| Simge | 3 |  | ✓ | 322 | 0 | 0 |
| Bohnes | 2 |  | ✓ | 2,762 | 0 | 0 |
| Guns N' Roses | 2 |  | ✓ | 309,644 | 23,985 | 417 |
| Shotgun Parade | 2 |  | – | 0 | 0 | 0 |
| Sertab Erener | 2 | ✓ | ✓ | 11,052 | 1,041 | 31 |
| The Neighbourhood | 2 |  | ✓ | 46,307 | 0 | 0 |
| maNga | 2 | ✓ | ✓ | 9,219 | 539 | 11 |
| Anathema | 2 |  | ✓ | 91,284 | 8,498 | 146 |
| Flyleaf | 2 |  | ✓ | 109,653 | 4,801 | 77 |
| Red | 2 |  | ✓ | 55,610 | 1,478 | 23 |
| M Lisa | 2 |  | ✓ | 53 | 0 | 0 |
| Anıl Piyancı | 2 | ✓ | ✓ | 143 | 0 | 0 |
| Escape the Fate | 1 |  | ✓ | 72,131 | 3,036 | 24 |
| Project Vela | 1 |  | ✓ | 453 | 0 | 0 |
| Pastelle | 1 |  | ✓ | 281 | 0 | 0 |
| LANDMVRKS | 1 |  | ✓ | 2,807 | 0 | 0 |
| Pantera | 1 |  | ✓ | 167,087 | 11,174 | 194 |
| Pearl Jam | 1 |  | ✓ | 296,938 | 20,413 | 373 |
| Finger Eleven | 1 |  | ✓ | 94,301 | 2,002 | 88 |
| Black Veil Brides | 1 |  | ✓ | 34,235 | 10 | 0 |
| Arcane | 1 |  | ✓ | 3,430 | 24 | 4 |
| Vega | 1 | ✓ | ✓ | 3,865 | 1,235 | 36 |
| Pamela | 1 |  | ✓ | 864 | 180 | 9 |
| Berk Baysal | 1 | ✓ | ✓ | 34 | 0 | 0 |
| Nomy | 1 |  | ✓ | 15,533 | 58 | 1 |
| Domina Noctis | 1 |  | ✓ | 2,622 | 20 | 0 |
| Cheriimoya | 1 |  | ✓ | 1,498 | 0 | 0 |
| Decyfer Down | 1 |  | ✓ | 12,835 | 177 | 4 |
| Arctic Monkeys | 1 |  | ✓ | 352,930 | 30,397 | 457 |
| Kolpa | 1 | ✓ | ✓ | 155 | 0 | 0 |
| Yıldız Tilbe | 1 |  | ✓ | 1,012 | 71 | 9 |
| Papa Roach | 1 |  | ✓ | 207,896 | 11,148 | 186 |
| Sezen Aksu | 1 | ✓ | ✓ | 10,280 | 1,880 | 36 |
| mor ve ötesi | 1 | ✓ | ✓ | 9,021 | 1,531 | 35 |
| Funda | 1 |  | ✓ | 5 | 5 | 4 |
| TNK | 1 |  | ✓ | 1,013 | 11 | 2 |
| Atiye | 1 |  | – | 0 | 12 | 0 |
| Furkan Kızılay | 1 |  | – | 0 | 0 | 0 |
| Model | 1 | ✓ | ✓ | 1,637 | 5 | 1 |
| Çağan Şengül | 1 |  | ✓ | 41 | 0 | 0 |
| Şebnem Ferah | 1 | ✓ | ✓ | 5,205 | 1,470 | 18 |
| Yüksek Sadakat | 1 | ✓ | ✓ | 2,701 | 181 | 11 |
| Digital Daggers | 1 |  | ✓ | 5,436 | 0 | 0 |
| Cryoshell | 1 |  | ✓ | 2,714 | 0 | 0 |
| The Cranberries | 1 |  | ✓ | 271,917 | 13,684 | 355 |
| Chevelle | 1 |  | ✓ | 101,267 | 3,731 | 113 |
| Funda Arar | 1 | ✓ | ✓ | 1,546 | 572 | 11 |
| Green Day | 1 |  | ✓ | 372,230 | 28,770 | 418 |
| Anıl Emre Daldal | 1 | ✓ | ✓ | 394 | 0 | 0 |
| Server Uraz | 1 |  | ✓ | 49 | 0 | 0 |
| bülow | 1 |  | ✓ | 4,277 | 0 | 0 |
| Norm Ender | 1 |  | ✓ | 119 | 0 | 0 |
