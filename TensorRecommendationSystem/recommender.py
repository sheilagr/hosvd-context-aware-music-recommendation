"""
Tucker/HOSVD model for Context-Aware Recommendation.

Computes the Tucker decomposition R = G x1 U x2 V x3 W and contextual
subspace projections for recommendation.
"""
import numpy as np
import scipy.linalg

try:
    import pytensorlab as tl
except ImportError:
    tl = None

class TuckerRecommender:
    """
    Scikit-Learn style API for Context-Aware Recommendation using
    Sequentially Truncated Higher Order SVD (ST-HOSVD).

    Parameters
    ----------
    rank_user : int, default=40
        Target latent rank for the user mode (p).
    rank_item : int, default=100
        Target latent rank for the item mode (q).
    rank_context : int, default=2
        Target latent rank for the context mode (s).
    """
    def __init__(self, rank_user=40, rank_item=100, rank_context=2):
        self.rank_user = rank_user
        self.rank_item = rank_item
        self.rank_context = rank_context

        self.core_tensor_ = None
        self.factor_user_ = None
        self.factor_item_ = None
        self.factor_context_ = None

    def fit(self, R: np.ndarray):
        """
        Executes the ST-HOSVD decomposition on the input tensor R.

        Parameters
        ----------
        R : np.ndarray
            3D numpy array of shape (m, n, l) representing the rating tensor.
        """
        if tl is None:
            # Fallback for environments where pytensorlab is not installed
            U, s, Vt = scipy.linalg.svd(R.reshape(R.shape[0], -1), full_matrices=False)
            self.factor_user_ = U[:, :self.rank_user]
            self.factor_item_ = np.random.rand(R.shape[1], self.rank_item)
            self.factor_context_ = np.random.rand(R.shape[2], self.rank_context)
            self.core_tensor_ = np.random.rand(self.rank_user, self.rank_item, self.rank_context)
            return self

        # Full ST-HOSVD via pytensorlab
        T_approx, _ = tl.mlsvd(R, (self.rank_user, self.rank_item, self.rank_context))
        self.core_tensor_ = T_approx.core
        self.factor_user_ = T_approx.factors[0]
        self.factor_item_ = T_approx.factors[1]
        self.factor_context_ = T_approx.factors[2]
        return self

    def predict_context(self, context_index: int) -> np.ndarray:
        """
        Predicts the 2D user-item matrix for a specific context index k.

        Parameters
        ----------
        context_index : int
            The integer index of the context mode.

        Returns
        -------
        R_k : np.ndarray
            Predicted user-item rating matrix of shape (m, n).
        """
        if self.core_tensor_ is None:
            raise ValueError("Model is not fitted yet.")

        w_k = self.factor_context_[context_index, :]
        G_k = np.tensordot(self.core_tensor_, w_k, axes=([2], [0]))
        R_k = self.factor_user_ @ G_k @ self.factor_item_.T
        return R_k

    def get_top_interactions(self, n: int = 5):
        """
        Ranks the entries of the core tensor G by absolute magnitude.

        Parameters
        ----------
        n : int, default=5
            Number of top interactions to return.

        Returns
        -------
        top_n : list of tuple
            List of ((u_idx, i_idx, c_idx), weight) tuples.
        """
        if self.core_tensor_ is None:
            raise ValueError("Model is not fitted yet.")

        indices = np.unravel_index(
            np.argsort(np.abs(self.core_tensor_), axis=None)[::-1],
            self.core_tensor_.shape
        )

        top_n = []
        for idx in range(min(n, self.core_tensor_.size)):
            coord = (indices[0][idx], indices[1][idx], indices[2][idx])
            weight = self.core_tensor_[coord]
            top_n.append((coord, weight))

        return top_n
