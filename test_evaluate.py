import numpy as np
import pytest
from scipy import sparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    held_out_items_by_user,
    popularity_baseline_recommend,
    evaluate_popularity_baseline,
)


def test_precision_at_k_all_hits():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_precision_at_k_no_hits():
    assert precision_at_k([1, 2, 3], {9, 10}, k=3) == 0.0


def test_precision_at_k_partial():
    # top-3 has 2 hits out of 3 relevant possible
    assert precision_at_k([1, 2, 3], {1, 9, 10}, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_full_recall():
    assert recall_at_k([1, 2, 3, 4], {1, 2}, k=4) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k([1, 9, 9, 9], {1, 2}, k=4) == pytest.approx(0.5)


def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_ndcg_perfect_ranking_is_one():
    # relevant items placed at the very top -> NDCG should be 1.0
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    relevant = {1, 2}
    perfect = ndcg_at_k([1, 2, 9, 9], relevant, k=4)
    worse = ndcg_at_k([9, 9, 1, 2], relevant, k=4)
    assert worse < perfect
    assert perfect == pytest.approx(1.0)


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_test_interactions_by_user_groups_correctly():
    # users 0 and 1 have held-out items; user 2 has none
    mat = sparse.csr_matrix(
        np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=np.float64)
    )
    mapping = held_out_items_by_user(mat)
    assert mapping[0] == {1, 2}
    assert mapping[1] == {0}
    assert 2 not in mapping


def test_popularity_baseline_ranks_by_count():
    # item 0 interacted with by 3 users, item 1 by 1 user, item 2 by 0
    train = sparse.csr_matrix(
        np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float64)
    )
    top = popularity_baseline_recommend(train, k=3)
    assert top[0] == 0  # most popular item first


def test_evaluate_popularity_baseline_runs_end_to_end():
    train = sparse.csr_matrix(
        np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.float64)
    )
    test = sparse.csr_matrix(
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.float64)
    )
    metrics = evaluate_popularity_baseline(train, test, k_values=(2,))
    assert "precision@2" in metrics
    assert "recall@2" in metrics
    assert "ndcg@2" in metrics
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
