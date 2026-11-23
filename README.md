# Artist Recommendation Model

**An artist similarity model trained on open ListenBrainz listening data. On my own listening history it finds
about twice as many of my artists as Deezer's "related artists" do, under the same scoring rules.**

The model answers one question: *for a given artist, which artists are most similar, best first?* A
recommendation app can use it to suggest new artists from the ones you already play. It was built as the
recommendation source for my listening app (Playlist Tracker), but the pipeline and the evaluation stand on their
own.

---

## Results

### On my own listening (leave one out)

Each of my 102 artists was hidden from my history in turn. The rest became the input, and I checked where the
hidden artist appeared in the suggestions. Every source ran through the same scoring code: the same 10 seed
artists, 20 similar artists per seed and the same diversity rule. Only the similarity lists differ.

| Source | Hit@12 | Hit@20 | MRR |
|---|---:|---:|---:|
| Popularity (baseline) | 0.010 | 0.029 | 0.004 |
| Deezer related artists | 0.059 | 0.108 | 0.032 |
| **Item-item kNN (this repo)** | **0.127** | **0.206** | **0.053** |
| Implicit ALS | 0.088 | 0.118 | 0.020 |
| item2vec | 0.098 | 0.137 | 0.040 |

- The kNN model beats Deezer in a paired sign test. At 20, kNN finds 13 hidden artists that Deezer misses, and
  Deezer finds 3 that kNN misses (p = 0.02).
- Blends of kNN and Deezer (weighted sum, reciprocal rank fusion) did not beat kNN alone. The chosen setup is kNN
  first, with Deezer as a fallback for artists the model does not know.
- In a blind A/B test of today's 12 suggestions, I picked the kNN list over Deezer's without knowing which was
  which.
- kNN suggests more Turkish artists (4 of 12, Deezer 2 of 12). Turkish pop was the weakest part of the data, so
  this mattered.

### On 8,518 held-out ListenBrainz users

20% of each test user's artists were hidden. Hyperparameters were chosen on a separate set of 2,000 validation
users, and each model family was scored once on the test users.

| Model | Recall@12 | Hit@12 | NDCG@12 |
|---|---:|---:|---:|
| Popularity | 0.016 | 0.519 | 0.111 |
| **kNN (cosine, log weights)** | **0.069** | **0.791** | **0.316** |
| ALS (256 factors, BM25) | 0.055 | 0.749 | 0.272 |
| item2vec (window 20) | 0.034 | 0.639 | 0.171 |

Validation and test scores are almost identical, so there is no sign of overfitting.

---

## How it works

```
ListenBrainz dumps ──► prep ──► build ──► train ──► neighbour lists ──► local comparison
   (Kaggle)            (Kaggle)  (Kaggle)  (Kaggle)    (.npz, 50 each)    (my listening.db)
```

1. **Data choice.** I measured how well each candidate data set covers my artists before building anything.
   ListenBrainz covered 97% of my listening, Last.fm 360K 64% and Last.fm 1K 57%.
2. **Download and preparation (Kaggle).** The full ListenBrainz dump is 246 GB, so only what is needed is
   streamed: each user's all-time artist counts from the statistics dump, and about 15 days of ordered listens
   from the daily dumps. The result is 101,313 users, 44.4 million user-artist rows and 126 million listens.
3. **Training data.** Listens without an artist MBID are merged into the MBID artist with the same normalised
   name, and collaborations go to their first artist. Artists need at least 10 listeners and users at least 5
   artists. The final matrix is 85,183 users × 276,112 artists with 37.2 million pairs. Users are split
   80/10/10 by seed 42, and 20% of each validation and test user's artists are hidden.
4. **Models.**
   - **Item-item kNN:** cosine similarity between artists' listener vectors (`implicit`).
   - **Implicit ALS:** matrix factorisation with 64 to 256 factors.
   - **item2vec:** word2vec over listening sessions split at 30-minute gaps (`gensim`).

   All models are tuned with log or BM25 weighting and compared with a popularity baseline.
5. **Comparison with Deezer (local).** Deezer's related artists are fetched the same way the app fetches them.
   Every source is then scored by the app's own ranking code on my listening history.

**Privacy:** my own listening history never leaves my computer. Kaggle only sees public ListenBrainz data, and
every evaluation on my own data runs locally.

---

## Repository layout

| Path | Contents |
|---|---|
| `kaggle/explore/` | First look at the ListenBrainz dump formats |
| `kaggle/prep/` | Streams and shrinks the dumps into parquet files |
| `kaggle/counts/` | Listeners per artist, for the local coverage check |
| `kaggle/build/` | Artist merging, filtering, user split, sessions |
| `kaggle/train/` | Trains and scores the models and exports 50 neighbours per artist |
| `kaggle/*/output/` | Small outputs from each Kaggle run (logs, `results.json`, summaries) |
| `kaggle/*/test_*.py` | Local tests on small synthetic data before pushing a kernel |
| `src/user_data.py` | Reads the app's listening database read-only |
| `src/coverage.py`, `src/lb_coverage.py` | Data set coverage of my artists |
| `src/deezer_fetch.py` | Deezer related artists for my artists |
| `src/compare.py` | Leave-one-out, by-day split, blends and qualitative checks |
| `reports/` | Coverage and comparison reports (in Turkish) |
| `ROADMAP.md` | The plan and a dated log of every step (in Turkish) |

---

## Reproducing

Requirements: Python 3.11 and a Kaggle account with the CLI token in `~/.kaggle/`.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# Kaggle: each kernel reads the previous one's output (kernel_sources)
.venv/Scripts/python kaggle/build/test_build.py     # local checks on synthetic data first
.venv/Scripts/python kaggle/train/test_train.py
.venv/Scripts/kaggle kernels push -p kaggle/prep
.venv/Scripts/kaggle kernels push -p kaggle/build
.venv/Scripts/kaggle kernels push -p kaggle/train
.venv/Scripts/kaggle kernels output <user>/listenbrainz-train -p kaggle/train/output

# Local comparison, from a Playlist Tracker data folder
export PLAYLIST_TRACKER_DATA="path/to/data"   # optional, defaults to %APPDATA%\Playlist Tracker
.venv/Scripts/python -m src.deezer_fetch
.venv/Scripts/python -m src.compare
```

The kernel ids in `kaggle/*/kernel-metadata.json` point to my account. Change them to your username before
pushing. The training run takes about 2.5 hours on a Kaggle CPU. Large files (`data/`, `*.npz`, `*.npy`,
`*.parquet`) are not in the repository and are recreated by the steps above.

---

## Limitations

- The personal evaluation is one person's 3 weeks of listening (765 plays, 102 artists). The kNN–Deezer gap is
  significant at 20 suggestions but only borderline at 12.
- The ListenBrainz statistics dump seems to keep only each user's top artists, so smaller artists have fewer
  listeners than ListenBrainz's own API reports. A few Turkish artists are still missing from the model.
- The model only knows artists, not songs. Song suggestions currently come from each suggested artist's top tracks.

## Next steps

- Ship the kNN neighbour lists in Playlist Tracker, with Deezer as a fallback.
- Measure in the app whether suggested artists are actually played within 30 days.
- Song-level recommendations, starting with a song coverage check on ListenBrainz.

## Data and credits

- [ListenBrainz](https://listenbrainz.org) data dumps by the MetaBrainz Foundation, released under CC0.
- [MusicBrainz](https://musicbrainz.org) for artist ids and countries.
- [Deezer API](https://developers.deezer.com) for the related artists used as the comparison baseline.
- [implicit](https://github.com/benfred/implicit) and [gensim](https://radimrehurek.com/gensim/) for the models.
