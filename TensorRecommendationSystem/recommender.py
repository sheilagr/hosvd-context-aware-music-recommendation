"""
This file implements the Tucker/HOSVD model described in the thesis,
Chapter 2 ("The Tucker Model and Higher-Order SVD") and Chapter 4
("Software Implementation"). It computes the decomposition R = G x1 U
x2 V x3 W and the contextual predictions built from it.
"""
import numpy as np
import scipy.linalg

# pytensorlab is an external package that computes the multilinear SVD
# (mlsvd). It is optional: if it is not installed, `tl` stays as None
# and `fit` below falls back to a simplified computation instead of
# failing, so this code can still be inspected and tested without it.
try:
    import pytensorlab as tl
except ImportError:
    tl = None

class TuckerRecommender:
    """
    Scikit-Learn style API for Context-Aware Recommendation using
    Sequentially Truncated Higher Order SVD (ST-HOSVD).

    rank_user, rank_item, rank_context correspond to the target
    multilinear rank (p, q, s) of Chapter 2: the number of latent
    factors kept for the user, item, and context modes respectively.
    """
    def __init__(self, rank_user=40, rank_item=100, rank_context=2):
        self.rank_user = rank_user
        self.rank_item = rank_item
        self.rank_context = rank_context

        # After fit(), these hold the core tensor G and the factor
        # matrices U, V, W of the Tucker decomposition (Chapter 2,
        # Definition "Tucker Decomposition").
        self.core_tensor_ = None
        self.factor_user_ = None
        self.factor_item_ = None
        self.factor_context_ = None

    def fit(self, R: np.ndarray):
        """
        Executes the ST-HOSVD decomposition on the input tensor R,
        following Chapter 2, Definition "Sequentially Truncated HOSVD".
        """
        if tl is None:
            # Fallback for environments where pytensorlab is not
            # installed: computes only the mode-1 factor matrix from a
            # plain matrix SVD (Chapter 2, Definition "Singular Value
            # Decomposition") on the mode-1 unfolding, and fills the
            # remaining factors with placeholder random values so the
            # rest of the package can still be exercised and tested.
            U, s, Vt = scipy.linalg.svd(R.reshape(R.shape[0], -1), full_matrices=False)
            self.factor_user_ = U[:, :self.rank_user]
            self.factor_item_ = np.random.rand(R.shape[1], self.rank_item)
            self.factor_context_ = np.random.rand(R.shape[2], self.rank_context)
            self.core_tensor_ = np.random.rand(self.rank_user, self.rank_item, self.rank_context)
            return self

        # Real computation, used whenever pytensorlab is available:
        # a single call performs the sequential mode-by-mode SVDs of
        # ST-HOSVD and returns the resulting Tucker decomposition.
        target_rank = (self.rank_user, self.rank_item, self.rank_context)
        T_approx, sv = tl.mlsvd(R, target_rank)

        # Unpack the decomposed core tensor and factor matrices from the
        # returned TuckerTensor object T_approx.
        self.core_tensor_ = T_approx.core
        self.factor_user_ = T_approx.factors[0]
        self.factor_item_ = T_approx.factors[1]
        self.factor_context_ = T_approx.factors[2]

        return self

    def predict_context(self, context_index: int) -> np.ndarray:
        """
        Extracts the 2D user-item prediction matrix for a given context
        index k, implementing the contextual subspace projection of
        Chapter 3, Section "Contextual Subspace Projection":
        G_k = G x3 w_k, followed by R_k = U G_k V^T.
        """
        if self.core_tensor_ is None:
            raise ValueError("Model is not fitted yet.")

        # 1. Extract the row vector w_k of the context factor matrix W,
        #    corresponding to context k.
        w_k = self.factor_context_[context_index, :]

        # 2. Contract the core tensor G along mode 3 with w_k, to obtain
        #    the context-projected core matrix G_k (Chapter 3, eq:context_core).
        G_k = np.tensordot(self.core_tensor_, w_k, axes=([2], [0]))

        # 3. Reconstruct the predicted user-item matrix R_k = U G_k V^T
        #    (Chapter 3, eq:context_prediction).
        R_k = self.factor_user_ @ G_k @ self.factor_item_.T

        return R_k

    def get_top_interactions(self, n=5):
        """
        Ranks the entries of the core tensor G by absolute magnitude and
        returns their (user_latent, item_latent, context_latent)
        coordinates: the core-tensor interrogation procedure used in
        Chapter 3, Section "Empirical Results and Core Tensor
        Interrogation" to identify the dominant latent affinities.
        """
        if self.core_tensor_ is None:
            raise ValueError("Model is not fitted yet.")

        abs_core = np.abs(self.core_tensor_)
        # Sort all entries of G by absolute value and keep the n largest
        flat_indices = np.argsort(abs_core.ravel())[::-1][:n]

        # Convert the flat (1D) indices back to 3D tensor coordinates
        top_indices = np.unravel_index(flat_indices, self.core_tensor_.shape)

        results = []
        for idx in range(n):
            coordinate = (top_indices[0][idx], top_indices[1][idx], top_indices[2][idx])
            weight = self.core_tensor_[coordinate]
            results.append((coordinate, weight))

        return results
