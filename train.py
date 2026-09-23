#!/usr/bin/env python3
"""
End-to-end pipeline: download (if needed) -> load -> convert to implicit
-> time-based split -> train ALS -> evaluate vs. popularity baseline ->
save model + metrics.

Usage
-----
    python scripts/train.py --dataset-dir data/ml-25m --factors 64 \
        --regularization 0.05 --alpha 40 --iterations 15

For a quick local smoke test without the full 25M dataset, point
--dataset-dir at ml-latest-small instead (see README "Quickstart").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.als import ImplicitALS
from src.data import (
    build_interaction_matrix,
    load_ratings,
    restrict_test_to_known_users_items,
    time_based_split,
    to_implicit,
)
from src.evaluate import evaluate_model, evaluate_popularity_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=str, required=True,
                    help="Path to an extracted MovieLens directory containing ratings.csv")
    p.add_argument("--positive-threshold", type=float, default=3.5)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--regularization", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=40.0)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--max-eval-users", type=int, default=5000,
                    help="Subsample users for evaluation speed; None-equivalent via -1")
    p.add_argument("--artifacts-dir", type=str, default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading ratings from {args.dataset_dir} ...")
    ratings = load_ratings(args.dataset_dir)
    print(f"  {len(ratings):,} raw ratings")

    positive, maps = to_implicit(ratings, positive_threshold=args.positive_threshold)
    n_users = len(maps.idx_to_user_id)
    n_items = len(maps.idx_to_item_id)
    print(f"  {len(positive):,} positive implicit interactions "
          f"({n_users:,} users, {n_items:,} items) after threshold >= {args.positive_threshold}")

    train_df, test_df = time_based_split(positive, test_fraction=args.test_fraction)
    test_df = restrict_test_to_known_users_items(train_df, test_df)
    print(f"  train: {len(train_df):,} interactions | "
          f"test (warm-start only): {len(test_df):,} interactions")

    train_matrix = build_interaction_matrix(train_df, n_users, n_items)
    test_matrix = build_interaction_matrix(test_df, n_users, n_items)

    print(f"Training ALS (factors={args.factors}, reg={args.regularization}, "
          f"alpha={args.alpha}, iterations={args.iterations}) ...")
    model = ImplicitALS(
        factors=args.factors,
        regularization=args.regularization,
        alpha=args.alpha,
        iterations=args.iterations,
        random_state=args.seed,
    )
    model.fit(train_matrix, verbose=True)

    max_eval_users = None if args.max_eval_users == -1 else args.max_eval_users
    k_values = tuple(args.k_values)

    print("Evaluating ALS model ...")
    als_metrics = evaluate_model(
        model, train_matrix, test_matrix, k_values=k_values,
        max_users=max_eval_users, random_state=args.seed,
    )
    print("Evaluating popularity baseline ...")
    baseline_metrics = evaluate_popularity_baseline(
        train_matrix, test_matrix, k_values=k_values,
        max_users=max_eval_users, random_state=args.seed,
    )

    results = {
        "config": vars(args),
        "n_users": n_users,
        "n_items": n_items,
        "n_train_interactions": len(train_df),
        "n_test_interactions": len(test_df),
        "als_metrics": als_metrics,
        "popularity_baseline_metrics": baseline_metrics,
        "wall_clock_seconds": time.time() - t0,
    }

    print("\n=== Results (ALS vs. popularity baseline) ===")
    for k in k_values:
        print(f"  k={k:>3}  "
              f"precision  ALS={als_metrics[f'precision@{k}']:.4f}  base={baseline_metrics[f'precision@{k}']:.4f}  |  "
              f"recall  ALS={als_metrics[f'recall@{k}']:.4f}  base={baseline_metrics[f'recall@{k}']:.4f}  |  "
              f"ndcg  ALS={als_metrics[f'ndcg@{k}']:.4f}  base={baseline_metrics[f'ndcg@{k}']:.4f}")

    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    model_path = artifacts_dir / "als_model.npz"
    np.savez(
        model_path,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        idx_to_user_id=maps.idx_to_user_id,
        idx_to_item_id=maps.idx_to_item_id,
        factors=model.factors,
        regularization=model.regularization,
        alpha=model.alpha,
    )
    print(f"Saved model -> {model_path}")


if __name__ == "__main__":
    main()
