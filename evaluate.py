"""
Ranking evaluation for implicit-feedback recommenders.

We evaluate top-k ranking quality (Precision@k, Recall@k, NDCG@k), not
RMSE/MAE on predicted scores -- for a "recommend items" task, what
matters is whether the right items land in the top of the list, not
whether the raw score is numerically close to some target. See README
"Evaluation methodology" for the full argument, including why even
these offline metrics are not the whole story.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    relevances = relevances[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top_k = recommended[:k]
    gains = np.array([1.0 if item in relevant else 0.0 for item in top_k])
    dcg = _dcg_at_k(gains, k)
    ideal_gains = np.ones(min(len(relevant), k))
    idcg = _dcg_at_k(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def held_out_items_by_user(test_matrix: sparse.csr_matrix) -> dict[int, set[int]]:
    """Map user_idx -> set of held-out item_idx they interacted with."""
    result: dict[int, set[int]] = defaultdict(set)
    coo = test_matrix.tocoo()
    for u, i in zip(coo.row, coo.col):
        result[int(u)].add(int(i))
    return result


def evaluate_model(
    model,
    train_matrix: sparse.csr_matrix,
    test_matrix: sparse.csr_matrix,
    k_values: tuple[int, ...] = (5, 10, 20),
    max_users: int | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute mean Precision@k / Recall@k / NDCG@k over users with held-out data.

    Only users present in `test_matrix` (i.e. with at least one held-out
    positive interaction) are scored -- users with nothing held out
    contribute no signal to ranking metrics by definition.

    max_users subsamples users for speed on very large datasets; pass
    None to evaluate on all eligible users.
    """
    user_relevant = held_out_items_by_user(test_matrix)
    eligible_users = list(user_relevant.keys())

    if max_users is not None and len(eligible_users) > max_users:
        rng = np.random.default_rng(random_state)
        eligible_users = list(
            rng.choice(eligible_users, size=max_users, replace=False)
        )

    max_k = max(k_values)
    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    for user_idx in eligible_users:
        relevant = user_relevant[user_idx]
        recommended = [
            item for item, _ in model.recommend(
                user_idx, train_matrix, k=max_k, filter_already_seen=True
            )
        ]
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(recommended, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(recommended, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(recommended, relevant, k))

    return {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}


def popularity_baseline_recommend(
    train_matrix: sparse.csr_matrix, k: int
) -> list[int]:
    """Most-popular-items ranking, used as the baseline every model must beat."""
    item_counts = np.asarray(train_matrix.sum(axis=0)).ravel()
    top_idx = np.argsort(-item_counts)[:k]
    return top_idx.tolist()


def evaluate_popularity_baseline(
    train_matrix: sparse.csr_matrix,
    test_matrix: sparse.csr_matrix,
    k_values: tuple[int, ...] = (5, 10, 20),
    max_users: int | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """Same protocol as evaluate_model, but recommending the same popular
    items to every user. A learned model should clearly beat this --
    reporting it is what makes the results honest rather than just a
    number with no reference point.
    """
    user_relevant = held_out_items_by_user(test_matrix)
    eligible_users = list(user_relevant.keys())

    if max_users is not None and len(eligible_users) > max_users:
        rng = np.random.default_rng(random_state)
        eligible_users = list(
            rng.choice(eligible_users, size=max_users, replace=False)
        )

    max_k = max(k_values)
    global_top = popularity_baseline_recommend(train_matrix, max_k)

    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    for user_idx in eligible_users:
        relevant = user_relevant[user_idx]
        seen = set(train_matrix[user_idx].indices)
        recommended = [item for item in global_top if item not in seen]
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(recommended, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(recommended, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(recommended, relevant, k))

    return {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}
