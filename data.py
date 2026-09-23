"""
Download, load, and prepare MovieLens data as implicit feedback.

MovieLens ships explicit 1-5 star ratings. We convert to implicit,
positive-only signal because that's the realistic case for most
production recommenders (clicks, watches, purchases -- not star
ratings), and it's a deliberately harder, more honest problem than
predicting a rating: see README "Why implicit, not explicit".

Conversion rule: a rating >= POSITIVE_THRESHOLD counts as one positive
implicit interaction (strength 1.0). Lower ratings are dropped, not
treated as negative -- implicit ALS models absence of interaction as
"unknown", not "disliked", so folding low ratings in as explicit
negatives would misrepresent the implicit setting.
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy import sparse

ML_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

POSITIVE_THRESHOLD = 3.5  # rating >= this -> positive implicit interaction


def download_movielens(dest_dir: str | Path, url: str = ML_25M_URL, force: bool = False) -> Path:
    """Download and unzip a MovieLens release into dest_dir. Idempotent.

    Requires network access. If you're running this in an offline sandbox,
    download manually from https://grouplens.org/datasets/movielens/ and
    unzip into dest_dir instead.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_name = url.rsplit("/", 1)[-1]
    zip_path = dest_dir / zip_name
    extracted_name = zip_name.replace(".zip", "")
    extracted_path = dest_dir / extracted_name

    if extracted_path.exists() and not force:
        return extracted_path

    if not zip_path.exists() or force:
        print(f"Downloading {url} -> {zip_path}")
        urlretrieve(url, zip_path)

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    return extracted_path


def load_ratings(dataset_dir: str | Path) -> pd.DataFrame:
    """Load ratings.csv from an extracted MovieLens directory."""
    path = Path(dataset_dir) / "ratings.csv"
    df = pd.read_csv(path)
    expected = {"userId", "movieId", "rating", "timestamp"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"ratings.csv missing expected columns: {missing}")
    return df


def load_movies(dataset_dir: str | Path) -> pd.DataFrame:
    """Load movies.csv (id -> title/genres) from an extracted MovieLens directory."""
    path = Path(dataset_dir) / "movies.csv"
    return pd.read_csv(path)


@dataclass
class IdMaps:
    user_id_to_idx: dict
    idx_to_user_id: np.ndarray
    item_id_to_idx: dict
    idx_to_item_id: np.ndarray


def to_implicit(
    ratings: pd.DataFrame, positive_threshold: float = POSITIVE_THRESHOLD
) -> tuple[pd.DataFrame, IdMaps]:
    """Filter to positive interactions and remap user/item ids to dense indices.

    Returns the filtered interactions (with dense user_idx/item_idx columns
    added) and the id maps needed to convert back to original MovieLens ids.
    """
    positive = ratings[ratings["rating"] >= positive_threshold].copy()

    user_ids = np.sort(positive["userId"].unique())
    item_ids = np.sort(positive["movieId"].unique())

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    positive["user_idx"] = positive["userId"].map(user_id_to_idx)
    positive["item_idx"] = positive["movieId"].map(item_id_to_idx)

    maps = IdMaps(
        user_id_to_idx=user_id_to_idx,
        idx_to_user_id=user_ids,
        item_id_to_idx=item_id_to_idx,
        idx_to_item_id=item_ids,
    )
    return positive, maps


def time_based_split(
    interactions: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by GLOBAL timestamp cutoff, not a random per-row split.

    Random splitting leaks future information into training (the model
    could implicitly learn from interactions that "haven't happened yet"
    relative to a test point) and overstates offline metrics. A single
    global time cutoff is the more honest, harder, more realistic
    evaluation setup: train on the past, evaluate on the future -- see
    README "Evaluation methodology".
    """
    cutoff = interactions["timestamp"].quantile(1 - test_fraction)
    train = interactions[interactions["timestamp"] < cutoff]
    test = interactions[interactions["timestamp"] >= cutoff]
    return train, test


def build_interaction_matrix(
    df: pd.DataFrame, n_users: int, n_items: int
) -> sparse.csr_matrix:
    """Build a (n_users x n_items) sparse binary interaction matrix."""
    rows = df["user_idx"].to_numpy()
    cols = df["item_idx"].to_numpy()
    data = np.ones(len(df), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def restrict_test_to_known_users_items(
    train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Drop test interactions for users/items never seen in training.

    A model trained via matrix factorization has no factors for a user or
    item it never saw -- including such rows in evaluation would silently
    conflate cold-start failure with ranking-quality failure. We measure
    cold-start separately and explicitly (see README), so here we keep the
    warm-start evaluation clean.
    """
    known_users = set(train["user_idx"].unique())
    known_items = set(train["item_idx"].unique())
    return test[
        test["user_idx"].isin(known_users) & test["item_idx"].isin(known_items)
    ]
