# MovieLens Implicit-Feedback Recommender (ALS)

A collaborative filtering recommender trained on real MovieLens interaction
data, built to reflect how recommenders actually work in production: implicit
signals rather than star ratings, ranking evaluation rather than rating-error
evaluation, and an explicit accounting of where the model breaks down.

This is the **training repo**. A separate demo repo
([`movielens-als-demo`](#)) reuses this repo's trained model to let a visitor
type in a few movies they like and get live recommendations — which also
doubles as a live demonstration of this model's main limitation, cold-start
(see [Limitations](#limitations)).

## Why implicit, not explicit

MovieLens ships explicit 1-5 star ratings, and it's tempting to build a
rating-prediction model (RMSE/MAE on held-out stars) because that's the
"obvious" dataset-native task. Most real recommenders don't have that luxury:
retail, streaming, and social platforms mostly observe *implicit* signals —
clicks, watches, purchases, dwell time — with no explicit negative feedback
at all. Someone who didn't watch a movie might dislike it, might not know it
exists, or might just not have gotten to it yet; those are very different
things, and implicit-feedback models are built around that ambiguity rather
than around it away.

So this project deliberately converts MovieLens into an implicit-feedback
problem: a rating **≥ 3.5** becomes a single positive interaction (strength
1.0); everything else is treated as *unknown*, not *negative*. That's a
harder and more realistic problem than rating prediction, and it's the one
most recommender roles actually involve.

## Model: Implicit ALS, implemented from scratch

The core algorithm is Alternating Least Squares for implicit feedback, from
[Hu, Koren & Volinsky (2008), "Collaborative Filtering for Implicit Feedback
Datasets"](https://ieeexplore.ieee.org/document/4781121). It's implemented
from scratch in [`src/als.py`](src/als.py) using NumPy/SciPy sparse matrices
— not the `implicit` package — specifically so this repo demonstrates
understanding of the algorithm rather than just calling a library. The
sparse-update trick used (computing `YᵀY` once per iteration and only
correcting for each user's nonzero items) is the same one production
libraries use internally, so the implementation is a faithful, if slower,
version of what a production system would run. `requirements.txt` notes
`implicit` as an optional drop-in for anyone who wants Cython-speed training
on the full 25M dataset.

Model:

```
confidence  c_ui = 1 + alpha * r_ui
preference  p_ui = 1 if r_ui > 0 else 0

minimize  sum_ui c_ui * (p_ui - x_u^T y_i)^2  +  lambda * (||X||² + ||Y||²)
```

solved by alternating closed-form ridge regressions for user factors `X` and
item factors `Y`. See the docstring in `src/als.py` for the full derivation
and the sparsity trick that makes it tractable at MovieLens scale.

## Evaluation methodology

We evaluate **ranking quality** — Precision@k, Recall@k, NDCG@k — not rating
error. For a "recommend items" task, what matters is whether the right items
land near the top of a list, not whether a numeric score is close to some
target; RMSE on implicit strength (0/1) isn't a meaningful thing to measure.

Two choices worth calling out explicitly, because they materially affect
results and are easy to get quietly wrong:

- **Time-based train/test split**, not a random split. We pick one global
  timestamp cutoff and train on everything before it, test on everything
  after. A random per-interaction split leaks future information into
  training and inflates offline metrics — it lets the model implicitly learn
  from data that "hasn't happened yet" relative to some test point. See
  `time_based_split` in `src/data.py`.
- **Warm-start-only evaluation.** Matrix factorization has no factors for a
  user or item it never saw in training, so test interactions involving
  unseen users/items are dropped before scoring (`restrict_test_to_known_users_items`).
  This keeps ranking-quality metrics from being contaminated by cold-start
  failure, which is measured and discussed separately below.
- **Popularity baseline, always reported alongside the model.** A number
  without a reference point isn't informative — every result table below
  reports ALS next to "recommend the same popular items to everyone."

### Why even these metrics aren't the whole story

Offline Precision/Recall/NDCG are computed against *historical* interactions
— they measure "did the model rank the movies this user watched anyway
highly," not "did the model surface movies the user wouldn't have found
otherwise." A recommender that mostly re-ranks popular blockbusters can score
well offline while adding little real value; conversely, a genuinely good
"discovery" recommendation that the user never would have manually
picked doesn't get credit unless they'd have watched it under the current
system too. Offline metrics are a necessary, cheap correctness check before
online A/B testing — not a substitute for it. (For that side of the
recommender lifecycle, see the
[marketplace A/B testing project](https://github.com/agnija-ashrita/marketplace-interference-ab-testing)
elsewhere in this portfolio.)

## Results

Running the full pipeline requires downloading the ~250MB MovieLens 25M
dataset, which needs network access this build environment didn't have.
**Numbers below are from a small synthetic smoke test** (300 users, 150
items, block-structured synthetic preferences) used to verify the pipeline
runs correctly end-to-end and that ALS meaningfully beats the baseline — they
are *not* the real MovieLens benchmark numbers, and shouldn't be read as
such. Run `scripts/train.py` against the real dataset (see Quickstart) and
replace this table with those numbers before treating this as a finished
result.

| k | Precision@k (ALS / baseline) | Recall@k (ALS / baseline) | NDCG@k (ALS / baseline) |
|---|---|---|---|
| 5  | 0.090 / 0.018 | 0.168 / 0.038 | 0.134 / 0.031 |
| 10 | 0.082 / 0.019 | 0.312 / 0.076 | 0.194 / 0.047 |
| 20 | 0.071 / 0.018 | 0.559 / 0.148 | 0.275 / 0.071 |

(Full test suite — 22 tests covering data transforms, metrics, and a
structure-recovery sanity check for the ALS model itself — passes; see
[Testing](#testing).)

## Limitations

- **Cold-start.** A brand-new user or item has no learned factor row. The
  demo repo's "pick a few movies you like" flow works around this with a
  *fold-in* approximation (`ImplicitALS.score_new_user`): it solves the same
  per-user ridge regression ALS uses internally, treating the typed-in movies
  as that user's only observed interactions against the already-trained item
  factors. This is a standard, reasonable approximation — but it's built from
  3-5 data points instead of a full interaction history, so it's noticeably
  less reliable than recommendations for users the model actually trained
  on. A production system would pair this with content-based features
  (genre, cast, text embeddings) to handle cold-start properly; that's the
  natural v2 extension, not implemented here.
- **Popularity/exposure bias.** Implicit feedback conflates "liked it" with
  "was shown it and engaged." Popular items get more interactions partly
  because they're popular, not purely because they're better matches — this
  is a known bias in implicit-feedback CF generally, not specific to this
  implementation.
- **No temporal drift modeling.** User taste changes over time; this model
  treats all training-period interactions as equally informative regardless
  of recency.
- **Offline-only evaluation**, as discussed above.

## Quickstart

```bash
git clone <this-repo>
cd movielens-als-recommender
pip install -r requirements.txt

# Download MovieLens (requires network). Use ml-latest-small for a fast
# local smoke test, or ml-25m for the real benchmark:
python -c "from src.data import download_movielens, ML_LATEST_SMALL_URL; \
           download_movielens('data', ML_LATEST_SMALL_URL)"

python scripts/train.py --dataset-dir data/ml-latest-small \
    --factors 64 --iterations 15
```

For the full 25M dataset, swap in `ML_25M_URL` — training will take
significantly longer since this implementation is pure NumPy/SciPy, not
the Cython-accelerated `implicit` package (see [Model](#model-implicit-als-implemented-from-scratch)).

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

22 tests across three files:
- `tests/test_data.py` — implicit conversion, id remapping, time-based split,
  warm-start filtering, sparse matrix construction
- `tests/test_evaluate.py` — Precision/Recall/NDCG correctness on hand-worked
  examples, popularity baseline
- `tests/test_als.py` — factor shapes, seen-item filtering, a
  block-structure-recovery sanity check (does the model actually learn
  latent structure, not just run without crashing), and cold-start fold-in
  behavior

## Project structure

```
├── src/
│   ├── als.py        # Implicit ALS model (from scratch)
│   ├── data.py        # Download, load, implicit conversion, time split
│   └── evaluate.py    # Precision@k / Recall@k / NDCG@k, popularity baseline
├── scripts/
│   └── train.py        # End-to-end CLI: load -> train -> evaluate -> save
├── tests/
│   ├── test_als.py
│   ├── test_data.py
│   └── test_evaluate.py
├── data/                # (gitignored) downloaded MovieLens files
├── artifacts/           # (gitignored) trained model + metrics.json
├── requirements.txt
└── README.md
```

## Reference

Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for
Implicit Feedback Datasets. *IEEE International Conference on Data Mining*.

MovieLens dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. The
MovieLens Datasets: History and Context. *ACM Transactions on Interactive
Intelligent Systems*. https://grouplens.org/datasets/movielens/
