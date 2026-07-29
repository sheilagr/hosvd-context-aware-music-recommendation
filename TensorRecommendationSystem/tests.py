"""
Automatic checks for TensorRecommendationSystem. Each test below verifies
one mathematical property claimed in the thesis rather than one specific
numerical outcome. Run all of them with:

    python -m unittest tests

See README.md, Section 4, for a plain-language explanation of what each
test checks and why it matters mathematically.
"""
import unittest
import numpy as np
import pandas as pd
from ingestion import TensorizationModule
from recommender import TuckerRecommender

class TestTensorRecommendationSystem(unittest.TestCase):

    def setUp(self):
        # Small synthetic dataset used by several tests below: 3 users,
        # 3 items, 2 contexts, 5 observed (user, item, context, rating) rows.
        data = {
            'user': ['u1', 'u1', 'u2', 'u3', 'u3'],
            'item': ['i1', 'i2', 'i1', 'i3', 'i2'],
            'context': ['c1', 'c2', 'c1', 'c1', 'c2'],
            'rating': [10, 5, 20, 15, 8]
        }
        self.df = pd.DataFrame(data)

    def test_tensorization_module(self):
        # Checks that TensorizationModule builds a tensor of the correct
        # shape and applies the log(1+x) rescaling from Chapter 3.
        ingestor = TensorizationModule()
        tensor = ingestor.fit_transform(self.df)

        # We have 3 unique users, 3 unique items, 2 unique contexts
        self.assertEqual(tensor.shape, (3, 3, 2))

        # Verify logarithmic scaling
        u1_idx = ingestor.user_map_['u1']
        i1_idx = ingestor.item_map_['i1']
        c1_idx = ingestor.context_map_['c1']

        expected_val = np.log1p(10)
        self.assertAlmostEqual(tensor[u1_idx, i1_idx, c1_idx], expected_val)

    def test_recommender_fit_predict(self):
        # Checks that predict_context returns a matrix of the exact
        # user x item shape expected from eq:context_prediction, on a
        # random tensor with a small target rank (to exercise truncation).
        tensor = np.random.rand(10, 10, 4)

        model = TuckerRecommender(rank_user=3, rank_item=3, rank_context=2)
        model.fit(tensor)

        # Test predictions for context 0
        pred_matrix = model.predict_context(0)

        # The predicted matrix should be users x items
        self.assertEqual(pred_matrix.shape, (10, 10))

    def test_factor_orthogonality(self):
        # Checks the defining property of the HOSVD (Chapter 2, Definition
        # "Higher-Order Singular Value Decomposition"): the factor matrices
        # U, V, W must be orthonormal, i.e. U^T U = I, V^T V = I, W^T W = I.
        tensor = np.random.rand(10, 10, 4)

        model = TuckerRecommender(rank_user=3, rank_item=3, rank_context=2)
        model.fit(tensor)

        # Determine if pytensorlab was used (real computation) or the
        # scipy-based fallback (mock), since the check differs slightly.
        from recommender import tl

        if tl is not None:
            # Ensure orthogonality of all modes: U^T @ U = I
            for idx, (factor, rank) in enumerate([
                (model.factor_user_, 3),
                (model.factor_item_, 3),
                (model.factor_context_, 2)
            ]):
                dot_product = factor.T @ factor
                identity = np.eye(rank)
                np.testing.assert_allclose(
                    dot_product,
                    identity,
                    atol=1e-10,
                    err_msg=f"Factor matrix for mode {idx+1} is not orthonormal!"
                )
        else:
            # Under the fallback, factor_user_ comes from a plain scipy
            # SVD and must still be orthonormal by the SVD definition.
            dot_product = model.factor_user_.T @ model.factor_user_
            identity = np.eye(3)
            np.testing.assert_allclose(
                dot_product,
                identity,
                atol=1e-10,
                err_msg="Mocked factor_user_ is not orthonormal!"
            )

    def test_get_top_interactions(self):
        # Checks that get_top_interactions returns coordinates within the
        # core tensor's bounds, matching the stored weights, and correctly
        # sorted by decreasing absolute magnitude.
        tensor = np.random.rand(10, 10, 4)
        model = TuckerRecommender(rank_user=3, rank_item=3, rank_context=2)
        model.fit(tensor)

        top_n = model.get_top_interactions(n=5)

        # Verify it returns n elements
        self.assertEqual(len(top_n), 5)

        # Verify coordinates match expected dimension shape and weights match core values
        for coord, weight in top_n:
            self.assertEqual(len(coord), 3)
            self.assertTrue(0 <= coord[0] < 3)
            self.assertTrue(0 <= coord[1] < 3)
            self.assertTrue(0 <= coord[2] < 2)
            self.assertEqual(model.core_tensor_[coord], weight)

        # Verify they are actually sorted in descending order of absolute value
        absolute_weights = [abs(w) for c, w in top_n]
        self.assertEqual(absolute_weights, sorted(absolute_weights, reverse=True))


if __name__ == '__main__':
    unittest.main()
