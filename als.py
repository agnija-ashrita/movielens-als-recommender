"""
Implicit-feedback Alternating Least Squares (ALS).

Implements the weighted matrix factorization model from:
    Hu, Y., Koren, Y., & Volinsky, C. (2008).
    "Collaborative Filtering for Implicit Feedback Datasets."
    IEEE International Conference on Data Mining (ICDM).

Why implemented from scratch rather than using the `implicit` package:
this project is meant to demonstrate understanding of the algorithm, not
just call a library. The math and the sparse-update trick below are the
same ones production libraries (e.g. benfred/implicit) use internally.

Model
-----
For each observed implicit signal r_ui >= 0 (e.g. binary "watched"):
    confidence  c_ui = 1 + alpha * r_ui
    preference  p_ui = 1 if r_ui > 0 else 0

We minimize, over user factors X (n_users x k) and item factors Y
(n_items x k):

    sum_ui c_ui * (p_ui - x_u^T y_i)^2  +  lambda * (||X||_F^2 + ||Y||_F^2)

Solved by alternating least squares. Each user/item update is a
closed-form ridge regression:

    x_u = (Y^T Y + Y^T (C_u - I) Y + lambda * I)^-1 Y^T C_u p_u

The key efficiency trick (used here, same as the reference paper and
production implementations): Y^T Y is computed ONCE per iteration
(k x k, cheap). For each user, only their nonzero items contribute the
"(C_u - I)" correction, so per-user work is O(nnz_u * k^2) instead of
O(n_items * k^2). This is what makes ALS tractable at MovieLens-25M
scale (tens of millions of interactions, tens of thousands of items).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class ImplicitALS:
    """Implicit-feedback matrix factorization via Alternating Least Squares.

    Parameters
    ----------
    factors : int
        Dimensionality of the latent factor space (k).
    regularization : float
        L2 regularization strength (lambda).
    alpha : float
        Confidence scaling. confidence = 1 + alpha * interaction_strength.
    iterations : int
        Number of full ALS sweeps (user update + item update = 1 iteration).
    random_state : int | None
        Seed for reproducible factor initialization.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 40.0,
        iterations: int = 15,
        random_state: int | None = 42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, interactions: sparse.csr_matrix, verbose: bool = False) -> "ImplicitALS":
        """Fit the model.

        Parameters
        ----------
        interactions : scipy.sparse.csr_matrix, shape (n_users, n_items)
            Nonzero entries are implicit signal strength (e.g. 1.0 for a
            positive interaction). Zero/absent entries are treated as
            "unknown", not "negative" -- this is the core idea of implicit
            feedback ALS: absence of a signal is low-confidence, not
            necessarily disliked.
        """
        if not sparse.isspmatrix_csr(interactions):
            interactions = interactions.tocsr()

        n_users, n_items = interactions.shape
        rng = np.random.default_rng(self.random_state)

        # Small random init keeps early predictions near zero.
        self.user_factors = rng.normal(0, 0.01, size=(n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, size=(n_items, self.factors))

        # Confidence-weighted interactions, stored sparse: C - I is what
        # actually enters the per-row update (see module docstring).
        conf = interactions.copy().astype(np.float64)
        conf.data = self.alpha * conf.data  # this is (c_ui - 1)

        conf_t = conf.T.tocsr()  # item -> user view, for the item update

        reg_eye = self.regularization * np.eye(self.factors)

        for it in range(self.iterations):
            self.user_factors = self._als_step(
                conf, self.item_factors, reg_eye
            )
            self.item_factors = self._als_step(
                conf_t, self.user_factors, reg_eye
            )
            if verbose:
                print(f"[ALS] iteration {it + 1}/{self.iterations} done")

        return self

    @staticmethod
    def _als_step(
        conf_minus_one: sparse.csr_matrix,
        fixed_factors: np.ndarray,
        reg_eye: np.ndarray,
    ) -> np.ndarray:
        """One half-sweep: solve for `solving_factors` given `fixed_factors`.

        conf_minus_one has shape (n_solving, n_fixed) and holds
        alpha * r_ui (i.e. c_ui - 1) at nonzero entries.
        """
        n_solving = conf_minus_one.shape[0]
        k = fixed_factors.shape[1]

        YtY = fixed_factors.T @ fixed_factors  # (k, k), shared across all rows
        new_factors = np.zeros((n_solving, k))

        indptr = conf_minus_one.indptr
        indices = conf_minus_one.indices
        data = conf_minus_one.data

        for row in range(n_solving):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                # No observed interactions: ridge solution is 0 given the
                # zero right-hand side -- skip the solve.
                continue

            item_idx = indices[start:end]
            c_minus_1 = data[start:end]  # alpha * r_ui, i.e. c_ui - 1

            Y_u = fixed_factors[item_idx]  # (nnz, k)
            # Y^T (C_u - I) Y, computed only over the nonzero rows
            weighted = Y_u * c_minus_1[:, None]
            A = YtY + Y_u.T @ weighted + reg_eye
            # Y^T C_u p_u = sum over nonzero items of c_ui * y_i
            # (since p_ui = 1 exactly where c_ui - 1 is stored)
            b = (Y_u * (c_minus_1 + 1.0)[:, None]).sum(axis=0)

            new_factors[row] = np.linalg.solve(A, b)

        return new_factors

    def recommend(
        self,
        user_id: int,
        interactions: sparse.csr_matrix,
        k: int = 10,
        filter_already_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Top-k item recommendations for a known user id (matrix row index)."""
        scores = self.item_factors @ self.user_factors[user_id]
        if filter_already_seen:
            seen = interactions[user_id].indices
            scores = scores.copy()
            scores[seen] = -np.inf
        top_idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx]

    def score_new_user(
        self,
        liked_item_ids: list[int],
        regularization: float | None = None,
    ) -> np.ndarray:
        """Cold-start scoring for a brand-new user given a handful of liked items.

        This is NOT collaborative filtering in the usual sense -- the user
        has no row in the trained factor matrix. Instead we solve the same
        ridge regression the ALS update uses, treating the liked items as
        that user's only observed positive interactions against the
        already-trained item factors. This is a standard "fold-in" /
        cold-start approximation, and it is exactly the mechanism the demo
        app uses. It is fundamentally limited by having only a few
        interactions to work with -- see README "Limitations".
        """
        if self.item_factors is None:
            raise RuntimeError("Model is not fit yet.")
        reg = self.regularization if regularization is None else regularization
        k = self.factors
        Y_u = self.item_factors[liked_item_ids]  # (n_liked, k)
        c_minus_1 = np.full(len(liked_item_ids), self.alpha)
        YtY = self.item_factors.T @ self.item_factors
        weighted = Y_u * c_minus_1[:, None]
        A = YtY + Y_u.T @ weighted + reg * np.eye(k)
        b = (Y_u * (c_minus_1 + 1.0)[:, None]).sum(axis=0)
        x_new = np.linalg.solve(A, b)
        return self.item_factors @ x_new
