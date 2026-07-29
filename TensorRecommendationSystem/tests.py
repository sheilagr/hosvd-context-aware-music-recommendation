"""
Automated unit tests for TensorRecommendationSystem.
"""
import unittest
import numpy as np
import pandas as pd
from TensorRecommendationSystem.ingestion import TensorizationModule
from TensorRecommendationSystem.recommender import TuckerRecommender

class TestTensorRecommendationSystem(unittest.TestCase):

    def setUp(self):
        data = {
            'user': ['u1', 'u1', 'u2', 'u3', 'u3'],
            'item': ['i1', 'i2', 'i1', 'i3', 'i2'],
            'context': ['c1', 'c2', 'c1', 'c1', 'c2'],
            'rating': [10, 5, 20, 15, 8]
        }
        self.df = pd.DataFrame(data)

    def test_tensorization_module(self):
        ingestor = TensorizationModule()
        tensor = ingestor.fit_transform(self.df)

        self.assertEqual(tensor.shape, (3, 3, 2))

        u1_idx = ingestor.user_map_['u1']
        i1_idx = ingestor.item_map_['i1']
        c1_idx = ingestor.context_map_['c1']

        expected_val = np.log1p(10)
        self.assertAlmostEqual(tensor[u1_idx, i1_idx, c1_idx], expected_val)

    def test_recommender_fit_predict(self):
        tensor = np.random.rand(10, 10, 4)

        model = TuckerRecommender(rank_user=3, rank_item=3, rank_context=2)
        model.fit(tensor)

        pred_matrix = model.predict_context(0)
        self.assertEqual(pred_matrix.shape, (10, 10))

    def test_factor_orthogonality(self):
        tensor = np.random.rand(10, 10, 4)

        model = TuckerRecommender(rank_user=3, rank_item=3, rank_context=2)
        model.fit(tensor)

        from TensorRecommendationSystem.recommender import tl

        if tl is not None:
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
            dot_product = model.factor_user_.T @ model.factor_user_
            identity = np.eye(3)
            np.testing.assert_allclose(
                dot_product,
                identity,
                atol=1e-10,
                err_msg="Mocked factor_user_ is not orthonormal!"
            )

    def test_get_top_interactions(self):
        tensor = np.random.rand(10, 10, 4)
        model = TuckerRecommender(rank_user=3, rank_item=3, rank_context=2)
        model.fit(tensor)

        top_n = model.get_top_interactions(n=5)
        self.assertEqual(len(top_n), 5)

        for coord, weight in top_n:
            self.assertEqual(len(coord), 3)
            self.assertTrue(0 <= coord[0] < 3)
            self.assertTrue(0 <= coord[1] < 3)
            self.assertTrue(0 <= coord[2] < 2)
            self.assertEqual(model.core_tensor_[coord], weight)

        absolute_weights = [abs(w) for c, w in top_n]
        self.assertEqual(absolute_weights, sorted(absolute_weights, reverse=True))

if __name__ == '__main__':
    unittest.main()
