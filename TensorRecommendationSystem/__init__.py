"""
TensorRecommendationSystem: A Python library for context-aware recommendation via Higher-Order SVD (HOSVD) and Tucker decomposition.
"""
from .ingestion import TensorizationModule
from .recommender import TuckerRecommender

__version__ = "0.1.0"
__all__ = ["TensorizationModule", "TuckerRecommender"]
