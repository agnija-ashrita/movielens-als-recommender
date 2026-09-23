import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    build_interaction_matrix,
    restrict_test_to_known_users_items,
    time_based_split,
    to_implicit,
)


def make_fake_ratings() -> pd.DataFrame:
    # userId, movieId, rating, timestamp (seconds, increasing)
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "movieId": [10, 20, 30, 10, 40, 10, 20, 30, 40],
            "rating": [5.0, 2.0, 4.0, 3.5, 1.0, 5.0, 3.0, 4.5, 2.5],
            "timestamp": [100, 101, 102, 103, 104, 105, 106, 107, 108],
        }
    )


def test_to_implicit_filters_by_threshold():
    ratings = make_fake_ratings()
    positive, maps = to_implicit(ratings, positive_threshold=3.5)
    # ratings >= 3.5: (1,10,5.0) (1,30,4.0) (2,10,3.5) (3,10,5.0) (3,30,4.5)
    assert len(positive) == 5
    assert (positive["rating"] >= 3.5).all()


def test_to_implicit_remaps_ids_densely():
    ratings = make_fake_ratings()
    positive, maps = to_implicit(ratings, positive_threshold=3.5)
    # dense indices should start at 0 and be contiguous
    assert sorted(positive["user_idx"].unique()) == list(range(positive["user_idx"].nunique()))
    assert sorted(positive["item_idx"].unique()) == list(range(positive["item_idx"].nunique()))
    # round-trip via idx_to_*_id
    for _, row in positive.iterrows():
        assert maps.idx_to_user_id[int(row["user_idx"])] == row["userId"]
        assert maps.idx_to_item_id[int(row["item_idx"])] == row["movieId"]


def test_time_based_split_respects_global_cutoff():
    ratings = make_fake_ratings()
    positive, _ = to_implicit(ratings, positive_threshold=0.0)  # keep everything
    train, test = time_based_split(positive, test_fraction=0.3)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) + len(test) == len(positive)


def test_restrict_test_drops_unknown_users_and_items():
    train = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [0, 1, 0]})
    test = pd.DataFrame(
        {
            "user_idx": [0, 2, 0],   # user 2 never seen in train
            "item_idx": [0, 0, 5],   # item 5 never seen in train
        }
    )
    filtered = restrict_test_to_known_users_items(train, test)
    # only row (user 0, item 0) survives
    assert len(filtered) == 1
    assert filtered.iloc[0]["user_idx"] == 0
    assert filtered.iloc[0]["item_idx"] == 0


def test_build_interaction_matrix_shape_and_values():
    df = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [0, 1, 1]})
    mat = build_interaction_matrix(df, n_users=2, n_items=2)
    assert mat.shape == (2, 2)
    dense = mat.toarray()
    assert dense[0, 0] == 1.0
    assert dense[0, 1] == 1.0
    assert dense[1, 0] == 0.0
    assert dense[1, 1] == 1.0
