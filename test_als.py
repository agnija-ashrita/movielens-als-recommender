import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.als import ImplicitALS


def make_block_interactions(n_users=40, n_items=40, n_blocks=4, seed=0):
    """Synthetic data with clear latent block structure: users and items are
    split into n_blocks groups, and users only interact with items in their
    own group. A working ALS model should recover this structure and rank
    same-block items above other-block items for held-out interactions.
    """
    rng = np.random.default_rng(seed)
    user_block = rng.integers(0, n_blocks, size=n_users)
    item_block = rng.integers(0, n_blocks, size=n_items)

    rows, cols = [], []
    for u in range(n_users):
        same_block_items = np.where(item_block == user_block[u])[0]
        # each user interacts with most items in their block
        chosen = rng.choice(
            same_block_items, size=max(1, int(len(same_block_items) * 0.8)), replace=False
        )
        for i in chosen:
            rows.append(u)
            cols.append(i)

    data = np.ones(len(rows))
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    return mat, user_block, item_block


def test_fit_produces_correctly_shaped_factors():
    mat, _, _ = make_block_interactions()
    model = ImplicitALS(factors=8, iterations=5, random_state=0)
    model.fit(mat)
    assert model.user_factors.shape == (mat.shape[0], 8)
    assert model.item_factors.shape == (mat.shape[1], 8)
    assert not np.isnan(model.user_factors).any()
    assert not np.isnan(model.item_factors).any()


def test_recommend_filters_already_seen_items():
    mat, _, _ = make_block_interactions()
    model = ImplicitALS(factors=8, iterations=5, random_state=0)
    model.fit(mat)
    user_id = 0
    seen = set(mat[user_id].indices)
    recs = model.recommend(user_id, mat, k=10, filter_already_seen=True)
    rec_items = {item for item, _ in recs}
    assert rec_items.isdisjoint(seen)


def test_model_recovers_block_structure_better_than_random():
    """Sanity check that the ALS implementation actually learns something:
    for held-out same-block items, the model should score them higher than
    a random item, on average, for most users.
    """
    mat, user_block, item_block = make_block_interactions(
        n_users=60, n_items=60, n_blocks=3, seed=1
    )
    model = ImplicitALS(factors=10, iterations=10, regularization=0.05, alpha=20, random_state=1)
    model.fit(mat)

    rng = np.random.default_rng(2)
    correct_direction = 0
    trials = 0
    for u in range(mat.shape[0]):
        same_block_items = np.where(item_block == user_block[u])[0]
        other_block_items = np.where(item_block != user_block[u])[0]
        if len(same_block_items) == 0 or len(other_block_items) == 0:
            continue
        same_item = rng.choice(same_block_items)
        other_item = rng.choice(other_block_items)
        score_same = model.item_factors[same_item] @ model.user_factors[u]
        score_other = model.item_factors[other_item] @ model.user_factors[u]
        correct_direction += int(score_same > score_other)
        trials += 1

    accuracy = correct_direction / trials
    assert accuracy > 0.7, f"expected model to prefer same-block items, got accuracy={accuracy:.2f}"


def test_score_new_user_cold_start_prefers_similar_items():
    """The fold-in cold-start scorer should rank items similar to the
    liked items above unrelated items, using only trained item factors.
    """
    mat, user_block, item_block = make_block_interactions(
        n_users=60, n_items=60, n_blocks=3, seed=3
    )
    model = ImplicitALS(factors=10, iterations=10, regularization=0.05, alpha=20, random_state=3)
    model.fit(mat)

    target_block = 0
    liked_items = np.where(item_block == target_block)[0][:3].tolist()
    scores = model.score_new_user(liked_items)

    same_block_other_items = [
        i for i in np.where(item_block == target_block)[0] if i not in liked_items
    ]
    other_block_items = np.where(item_block != target_block)[0]

    mean_same = scores[same_block_other_items].mean()
    mean_other = scores[other_block_items].mean()
    assert mean_same > mean_other


def test_empty_user_row_yields_zero_factor_no_crash():
    # A user with zero interactions should get an all-zero factor row and
    # not crash the linear solve.
    mat = sparse.csr_matrix(
        np.array([[1, 1, 0], [0, 0, 0]], dtype=np.float64)
    )
    model = ImplicitALS(factors=4, iterations=3, random_state=0)
    model.fit(mat)
    assert np.allclose(model.user_factors[1], 0.0)
