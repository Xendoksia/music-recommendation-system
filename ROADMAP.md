# Müzik öneri modeli: yol haritası

Son güncelleme: 2026-11-23 · Durum: **Parça 0–4 bitti, ayar taraması (4b) Kaggle'da sürüyor, ardından Parça 5**

## Amaç
Playlist Tracker uygulamasındaki "Senin için" önerileri şu an Deezer'ın benzer sanatçı listesine dayanıyor
(`C:\Users\Deniz\Desktop\spotify liste kontrol\playlist_tracker\music\recommend.py`, commit `80ae4a2`). Bu klasörde **kendi modelimizi**
eğitip Deezer ile aynı ölçütlerle karşılaştıracağız. Deezer'ı tamamen çıkarmak hedef değil: hangisi ya da hangi
karışım en iyi sonucu verirse uygulamaya o girecek.

Model, uygulamanın ihtiyacı olan tek şeyi üretecek: **bir sanatçıya benzeyen sanatçılar, sıralı.** Uygulamada yalnızca
`related()` fonksiyonunun kaynağı değişecek. Başlangıç sanatçıları, puanlama, çeşitlilik (SEED_DECAY) ve gerekçe
metni olduğu gibi kalacak.

## Çalışma kuralları
- Parça parça ilerlenir. Her parçanın sonunda durulur, kullanıcı onay verince sonrakine geçilir.
- Uygulamanın gerçek verisi `%APPDATA%\Playlist Tracker` klasöründe. Buradan **yalnızca okunur**, hiçbir şey yazılmaz.
- İndirilen büyük veri setleri `data/` altında durur, kodla karıştırılmaz.
- Uygulama metinlerinde noktalı virgül kullanılmaz, cümleler bağlaçla bağlanır.
- Uygulama reposuna yapılan commit'lerde Claude satırı olmaz.

## Elimizde olanlar
- `data/spotify_tracks.csv`: HF "spotify-tracks-dataset". 114.000 satır, 89.741 benzersiz şarkı, 114 tür, ses
  özellikleri var ama kullanıcı yok. Kullanıcının 270 şarkısının 108'i, 102 sanatçısının 68'i bu veride bulundu.
  Bunlar dinlemelerin %46'sını karşılıyor.
- Kullanıcının dinlemeleri: 752 çalma, 270 şarkı, 117 sanatçı (1–21 Kasım 2026). Ağırlık metalcore ve alternatif
  rock'ta, yanında Türkçe pop var (Göksel, Burcu Güneş, Hilal Bilen).
- Uygulamanın MusicBrainz önbelleği (`artwork/artist_info.json`): sanatçıların MusicBrainz id'leri. ListenBrainz
  verisi de MBID kullandığı için isim eşleştirme derdini büyük ölçüde çözer.
- Deezer'ın sonuçları: uygulamanın `artwork/related.json` önbelleğine yazılacak (her sanatçı için ilk 20 benzer
  sanatçı). Uygulama önerileri henüz hiç çalıştırmadığı için bu dosya 2026-11-23 itibarıyla yok. Parça 4'ten önce
  doldurulması gerekiyor.

## Parça 0: Hazırlık
- Klasör yapısı: `data/` (ham veri), `notebooks/` (keşif), `src/` (tekrar kullanılan kod), `models/` (çıktılar),
  `reports/` (sonuç tabloları).
- `requirements.txt`: pandas, numpy, scipy, implicit (ALS), gensim (item2vec), scikit-learn, matplotlib.
- `src/user_data.py`: uygulamanın `listening.db` dosyasını salt okunur açar, sanatçıları uygulamanın kuralıyla
  çıkarır (ortak şarkılar virgül ve feat. ile bölünür, "&" bölünmez) ve `fold` ile normalize eder. Göksel ile
  "Goksel" aynı sayılır.

## Parça 1: Kapsam ölçümü (karar kapısı)
Model kurmadan önce hangi veri setinin işe yarayacağını ölçmek:
- Adaylar: **ListenBrainz** (açık lisanslı dinleme dökümleri, MBID'li, çok büyük olduğu için alt küme alınır),
  **Last.fm 360K** (kullanıcı–sanatçı dinleme sayıları, 2008), **Last.fm 1K** (zaman damgalı dinlemeler).
- Her biri için ölçülecekler:
  - Kullanıcının 102 sanatçısından kaçı var, kaçı yeterli sinyale sahip (örneğin en az 50 dinleyici)?
  - Ayrıca Türkçe sanatçılarda kapsam nasıl?
  - Kullanıcının dinlemelerinin yüzde kaçı kapsanıyor?
- Çıktı: `reports/kapsam.md` tablosu ve hangi veri setiyle devam edileceği kararı.
- Kapsam düşük çıkarsa birleştirme yapılır (ListenBrainz ve Last.fm birlikte) ya da modelin bilmediği sanatçılar
  için Deezer'a düşülür (Parça 4'teki karma çözüm).

## Parça 2: Veri hazırlama
- Kullanıcı × sanatçı matrisi (dinleme sayıları). Çok az dinlenen sanatçılar ve çok az dinleme yapan kullanıcılar
  atılır.
- Ağırlıklandırma: log(1 + dinleme) ya da BM25 (çok dinleyen birkaç kullanıcı her şeyi domine etmesin).
- İsim → MBID eşlemesi, MBID yoksa `fold` ile normalize edilmiş isim.
- Değerlendirme için bölme: kullanıcıların bir kısmı test için ayrılır (her test kullanıcısının sanatçılarının %20'si
  gizlenir).

## Parça 3: Modeller
Basitten karmaşığa doğru, aynı değerlendirmeyle:
1. **Popülerlik** (taban çizgisi): herkese en çok dinlenenleri öner. Bunu geçemeyen model işe yaramıyor demektir.
2. **Birlikte dinlenme (item-item cosine):** aynı kişilerin dinlediği sanatçılar benzerdir. Basit ve güçlü bir taban.
3. **Implicit ALS:** matris ayrıştırma ile her sanatçıya bir vektör (64–128 boyut).
4. **item2vec:** Last.fm 1K ve ListenBrainz'deki dinleme sıralarından word2vec (aynı oturumda art arda dinlenenler).
- Hiperparametreler (boyut, düzenlileştirme, iterasyon) test kullanıcıları üzerinde seçilir.

## Parça 4: Deezer ile karşılaştırma
Deezer de bir "model" gibi aynı ölçütlerle değerlendirilir.

**Ölçüm 1. Kullanıcının kendi verisiyle tek tek gizleme (asıl ölçüt)**
- Kullanıcının bilinen sanatçılarından biri gizlenir. Kalanlar başlangıç olarak verilir ve gizlenen sanatçının
  öneri listesinde kaçıncı sırada çıktığına bakılır.
- Bu işlem her sanatçı için tekrarlanır. Metrikler HitRate@10, HitRate@20 ve MRR.
- Deezer ve modeller birebir aynı başlangıç sanatçılarıyla, aynı puanlama ve çeşitlilik kuralıyla çalıştırılır.

**Ölçüm 2. Zamana göre bölme**
- Belirli bir tarihten önceki dinlemeler başlangıç olarak verilir. Sonrasında **ilk kez** dinlenen sanatçılar
  öneride çıkmış mı diye bakılır.
- Veri şu an 3 haftalık olduğu için örnek az olacak. Zaman geçtikçe değeri artar.

**Ölçüm 3. Genel veri setindeki test kullanıcıları**
- Recall@K ve NDCG@K. Deezer burada ölçülemez (her test kullanıcısı için API çağrısı gerekir), yalnızca modeller
  kendi aralarında karşılaştırılır.

**Nitel ve pratik ölçütler**
- Türkçe sanatçılar listeye giriyor mu?
- Çeşitlilik: önerilerin ne kadarı aynı türde?
- Popülerlik yanlılığı: hep en ünlüler mi geliyor?
- Kapsam: kullanıcının sanatçılarından kaçı için öneri üretilebiliyor?
- Hız ve dosya boyutu.
- Kör karşılaştırma: iki listeyi kaynağını söylemeden kullanıcıya göster, hangisini beğendiğini sor.

**Karma çözümler de denenir**
- Ağırlıklı toplam: `α × model + (1 − α) × Deezer`, α değeri Ölçüm 1 ile seçilir.
- Reciprocal Rank Fusion: iki listedeki sıralar birleştirilir, ağırlık ayarı gerektirmez.
- Yedekleme: model sanatçıyı tanımıyorsa Deezer'a düşülür (Türkçe sanatçılar için önemli olabilir).

**Çıktı:** `reports/karsilastirma.md` (tablo ve örnek listeler) ve "Deezer / model / karma" kararı.

## Parça 5: Uygulamaya taşıma
- Kazanan model dışa aktarılır: `artist_vectors.npy` (float16) ve `artists.json` (isim, MBID, normalize isim).
  Hedef boyut 20 MB'ın altı. numpy uygulamanın paketinde zaten var.
- `recommend.py` içinde `related()` için kaynak seçimi eklenir: config'de `RECOMMEND_SOURCE = deezer | model | hybrid`.
  Varsayılanı Parça 4'teki karar belirler.
- Model dosyası kurulum paketine eklenir (`build.py`). Testler sahte vektörlerle yazılır.
- Deezer fotoğraf ve en iyi şarkılar için kalır. Model seçildiğinde bile öneri mantığı internetsiz çalışır.

## Parça 6: Uygulama içinde ölçüm
- Önerilen bir sanatçı sonraki 30 gün içinde dinlendi mi? Uygulama dinlemeleri zaten kaydettiği için bu otomatik
  ölçülür. "Son 30 günde 20 öneriden 4'ünü dinledin" gibi bir isabet oranı çıkar.
- Interleaving: "Senin için" listesine Deezer ve model önerileri karışık konur, hangisinden gelenlerin daha çok
  dinlendiği sayılır. Bu, karşılaştırmanın canlı sürümü olur.

## İleride (şimdilik kapsam dışı)
- **Kişisel sıralayıcı:** Parça 6'nın "dinlendi / dinlenmedi" etiketleriyle önerileri yeniden sıralayan küçük bir
  model (lojistik regresyon). Özellikler tür, saat ve kaç başlangıç sanatçısının önerdiği olabilir.
- **Şarkı düzeyinde öneri:** HF veri setinin ses özellikleriyle "bu şarkıya benzeyenler" kartı.
- **Ses embedding'i:** CLAP, MERT ya da Essentia modelleri. Ses klibi gerektirdiği için şimdilik değil.

## Kaldığımız yer
- 2026-11-23: Yol haritası hazırlandı. Deezer tabanlı öneri uygulamada Python tarafında hazır (`80ae4a2`). "Senin
  için" arayüzü (masaüstü ve telefon) henüz yapılmadı, o iş uygulamanın kendi ROADMAP'inde.
- 2026-11-23, **Parça 0 bitti.** Klasörler açıldı, `.venv` kuruldu (`requirements.txt`, Windows sertifika deposu
  için `truststore` eklendi). `src/user_data.py` uygulamanın `listening.db` dosyasını salt okunur açıyor ve 752
  çalmadan 102 sanatçı çıkarıyor. Deezer önbelleği (`related.json`) henüz yok.
- 2026-11-23, **Parça 1 bitti.** `src/coverage.py` çalıştırıldı ve sonuç `reports/kapsam.md` dosyasında. Kapsam
  (en az 50 dinleyicisi olan sanatçıların dinlemelerdeki payı) ListenBrainz'de %97, Last.fm 360K'da %64, Last.fm
  1K'da %57 çıktı. **Karar:** Ana kaynak ListenBrainz olacak. Last.fm 1K bırakıldı, 360K yedekte kaldı. Last.fm
  arşivleri `data/` altında duruyor. Tam ListenBrainz dökümü 246 GB olduğu için diske sığmıyor.
- 2026-11-23: Veri setlerine yer açmak için klasör `C:\Users\Deniz\Desktop` altından
  `D:\music recommandation system` konumuna taşındı (D:'de 228 GB boş yer var). `.venv` D:'de yeniden
  kuruldu. Uygulamanın verisi yine `%APPDATA%\Playlist Tracker` içinde ve yalnızca okunuyor.
- 2026-11-23, **Parça 2 başladı (Kaggle).** İndirme ve küçültme işi Kaggle'da yapılıyor, çünkü bağlantı yaklaşık
  3 MB/sn hız veriyor ve 29 GB'ı indirmek 3 saat sürerdi. Kaggle CLI `.venv` içinde kurulu, anahtar
  `~/.kaggle/access_token` dosyasında duruyor (hesap: xendoksia). Notebook'lar `kaggle/` altında.
  - `kaggle/explore`: İstatistik dökümünde kullanıcı başına sanatçılar `artists_all_time.jsonl` dosyasında
    (açılmış hâli 4,2 GB) ve bu dosya arşivin başında duruyor. Şarkı ve albüm tabloları 100 GB'tan fazla tuttuğu
    için okunmuyor. Günlük dökümler sunucuda yalnızca son yaklaşık 15 gün için tutuluyor ve bir gün yaklaşık 22
    milyon dinleme içeriyor.
  - `kaggle/prep`: `user_artists.parquet` (tüm zamanlar), `listens/*.parquet` (15 günlük sıralı dinlemeler) ve
    ortak sanatçı sözlüğü `artists.parquet` üretiliyor. Kullanıcının kendi verisi (`listening.db`) buluta
    çıkarılmıyor ve değerlendirme yerelde yapılacak.
  - Hazırlık bitti: 101.313 kullanıcının tüm zaman istatistiği (44,4 milyon satır) ve 9–23 Kasım arasındaki 15 günün
    126 milyon dinlemesi. Sanatçı sözlüğünde 4,4 milyon kayıt var. Çıktı Kaggle'da `xendoksia/listenbrainz-prep`
    notebook'unun çıktısı olarak duruyor ve sonraki notebook'lar onu doğrudan okuyor.
  - `kaggle/counts` sanatçı başına dinleyici sayılarını çıkardı ve `src/lb_coverage.py` eşleştirmeyi yerelde yaptı
    (`reports/kapsam_listenbrainz.md`). En az 50 dinleyici eşiğinde kapsam istatistikte %88, son 15 günde %75. En
    az 10 dinleyici eşiğinde kapsam istatistikte %99. **Türk pop zayıf:** Burcu Güneş, Pera, SELIN, Zeynep Dizdar
    ve Zeynep Casalini 50'nin altında kalıyor. İstatistik dökümü API'den çok daha düşük sayılar veriyor (Burcu
    Güneş için 928 yerine 41). Sebebi büyük olasılıkla dökümün her kullanıcı için yalnızca en çok dinlenen
    sanatçıları tutması, ama bu henüz doğrulanmadı. Bu yüzden Deezer'a düşen karma çözüm büyük olasılıkla
    gerekli olacak.
  - Görsel özet: `reports/genel_bakis.html` (https://claude.ai/artifact/MkuNa7CzVC9xWGWR4v45NJ).
- 2026-11-23, **Parça 2 bitti.** `kaggle/build` (Kaggle'da `xendoksia/listenbrainz-build`) eğitim verisini kurdu.
  Yerel denemesi `kaggle/build/test_build.py` ile yapılıyor.
  - Birleştirme: MBID'siz 3,7 milyon ad kaydının 1,07 milyonu aynı adlı MBID'li sanatçıya, 212 bini ortak şarkının
    ilk sanatçısına bağlandı.
  - Eleme: en az 10 dinleyicili sanatçılar ve en az 5 sanatçı dinlemiş kullanıcılar kaldı. Matris 85.183 kullanıcı
    × 276.112 sanatçıdan ve 37,2 milyon çiftten oluşuyor.
  - Bölme (tohum 42): 68.147 eğitim, 8.518 doğrulama ve 8.518 test kullanıcısı. Doğrulama ve test kullanıcılarının
    sanatçılarının %20'si gizlendi (1,48 milyon çift).
  - Oturumlar: yalnızca eğitim kullanıcılarından, 30 dakikalık boşlukla bölündü. 966.552 oturum ve 12,5 milyon
    öğe çıktı, ortalama uzunluk 12,9. Oturumlarda 211.376 sanatçı geçiyor.
  - Ağırlıklandırma (log ya da BM25) Parça 3'te hiperparametre olarak seçilecek ve ham sayılar saklandı.
  - Kullanıcının sanatçılarından 99'u matriste kaldı ve bunlar dinlemelerin %99'unu karşılıyor. 89'unun en az 50
    dinleyicisi var (%90). Birleştirme Türk sanatçılara biraz yaradı: SELIN 50'nin üstüne çıktı. Matriste hiç
    olmayanlar Shotgun Parade, Funda ve Furkan Kızılay.
- 2026-11-23, **Parça 3 başladı.** `kaggle/train/train.py` (Kaggle'da `xendoksia/listenbrainz-train`) çalışıyor.
  - Puanlama uygulamanın `score()` fonksiyonunun kopyası: en çok dinlenen 10 başlangıç sanatçısı, her birinin 20
    komşusu, sıraya göre pay ve SEED_DECAY = 0,5. Ölçütler 12'de (uygulamanın gösterdiği) ve 20'de recall, hit
    rate ve NDCG.
  - Ayarlar 2.000 doğrulama kullanıcısında seçiliyor (knn için 4, ALS için 6, item2vec için 3 ayar). Her ailenin
    en iyisi 8.518 test kullanıcısında bir kez ölçülüyor. Her sanatçının 50 komşusu `neighbours_<model>.npz`
    dosyasına yazılıyor.
  - Yerel deneme `kaggle/train/test_train.py` ile yapılıyor: türleri bilinen sahte veride üç model de popülerliği
    geçiyor ve komşuların %99–100'ü aynı türden çıkıyor. Bu deneme küçük veride 64 boyutlu ALS'nin aşırı
    öğrendiğini ve log ağırlığının BM25'ten iyi çalıştığını gösterdi, bu yüzden ALS taramasına log'lu ayarlar
    eklendi.
- 2026-11-23, **Parça 3 bitti.** Kaggle'daki eğitim yaklaşık 2,5 saat sürdü ve hatasız tamamlandı. Çıktılar
  `kaggle/train/output/` altına indirildi (`results.json`, `summary.json`, `neighbours_{knn,als,item2vec}.npz`,
  ALS ve item2vec vektörleri).
  - Test sonuçları (8.518 kullanıcı, 12'de): popülerlik recall 0,016 / hit 0,52 / NDCG 0,11. **knn (cosine, log)
    recall 0,069 / hit 0,79 / NDCG 0,32** ile en iyisi. ALS (256 boyut, BM25) recall 0,055 / hit 0,75 / NDCG 0,27.
    item2vec (pencere 20) recall 0,034 / hit 0,64 / NDCG 0,17 ve başlangıç sanatçılarının yalnızca %86'sını
    tanıyor.
  - Doğrulama ve test sonuçları birbirine çok yakın, yani aşırı öğrenme görünmüyor. BM25 knn'de log'dan belirgin
    biçimde kötü çıktı.
- 2026-11-23, **Parça 4 başladı.** `src/deezer_fetch.py` Deezer'ın benzer listelerini uygulamanın yöntemiyle
  `data/deezer_related.json` dosyasına çekti. Uygulamanın önbelleğine dokunulmadı. Deezer 102 sanatçının hepsini
  tanıyor ve en fazla 20 benzer veriyor. `src/compare.py` bütün kaynakları uygulamanın `score()` kuralıyla ölçtü ve
  sonuçları `reports/karsilastirma.md` dosyasına yazdı.
  - Ölçüm 1 (tek tek gizleme, 102 sanatçı): hit@12 değeri Deezer'da 0,059, knn'de 0,127, ALS'de 0,088 ve
    item2vec'te 0,098. hit@20 değeri Deezer'da 0,108, knn'de 0,206. knn ile Deezer arasındaki fark eşleştirilmiş
    işaret testinde 20'de anlamlı çıktı (13'e 3, p = 0,02), 12'de anlamlılığın sınırında kaldı (9'a 2, p = 0,07).
    ALS ile item2vec Deezer'dan anlamlı biçimde ayrılmıyor.
  - Ölçüm 2 (günlük zaman bölmesi, 16 günde 85 yeni sanatçı): sıralama aynı ama oranlar çok düşük (hit@20 knn'de
    0,082, Deezer'da 0,047).
  - Karma çözümlerin hiçbiri tek başına knn'i belirgin biçimde geçmedi. En yakını α = 0,75 ile knn ve Deezer
    karışımı (hit@12 0,157 ama MRR daha düşük). "knn, yoksa Deezer" knn ile aynı sonucu veriyor, çünkü knn'in
    tanımadığı 3 sanatçı hiç başlangıç sanatçısı olmuyor.
  - Nitel: knn bugünkü 12 önerinin 4'ünde Türk sanatçı veriyor, Deezer 2'sinde. Popülerlik yanlılığı ikisinde
    benzer. Deezer ile knn listelerinin yalnızca 3 sanatçısı ortak.
- 2026-11-23, **Parça 4 bitti.** Kör karşılaştırmada kullanıcı knn listesini (A) Deezer'a (B) tercih etti.
  **Karar:** knn birincil kaynak olacak ve tanımadığı sanatçılarda Deezer'a düşülecek (`RECOMMEND_SOURCE =
  hybrid` varsayılan). Deezer fotoğraf ve en iyi şarkılar için kalıyor. Gerekçe `reports/karsilastirma.md`
  dosyasının Karar bölümünde.
- 2026-11-23, **Parça 4b başladı (ayar taraması).** Kullanıcı önce ayarların yapılmasını, sonra Parça 5'e
  geçilmesini istedi. Türk sanatçı kapsamı bir hedef değil, kararlar genel ölçütlerle veriliyor.
  - `kaggle/tune/tune.py` (Kaggle'da `xendoksia/listenbrainz-tune`) knn'i iki yönden ayarlıyor. Popülerlik
    cezasında benzerlik `dot / (|i|^(1−β) · |j|^β)` ile hesaplanıyor (β 0,3–0,85). Shrinkage'da paydaya 0, 10 ya
    da 50 ekleniyor. implicit bunları sunmadığı için her sanatçının cosine'e göre en yakın 400 adayı yeniden
    puanlanıyor.
  - Uygulamanın ayarları da taranıyor: başlangıç sayısı (5–30), benzer sayısı (10–50) ve SEED_DECAY
    (0,25–0,75). Önce benzerlik, sonra puanlama ayarları seçiliyor ve en iyi birkaç benzerlik yeniden
    deneniyor. Ölçüt NDCG@12. Sonunda şu anki ayar ile en iyi ayar 8.518 test kullanıcısında karşılaştırılıyor.
  - Senin verinde yapılan ön deneme: 20 başlangıç sanatçısıyla hit@12 0,127'den 0,167'ye çıktı. Benzer sayısını
    50'ye çıkarmak sıralamayı bozdu.
  - Yerel deneme `kaggle/tune/test_tune.py` ile yapılıyor.
- **Sıradaki:** Tarama bitince en iyi ayarlarla Parça 4'ün karşılaştırması yeniden yapılacak, ardından Parça 5
  (uygulamaya taşıma) gelecek.
