"""
Data ingestion module for converting tabular interaction data into dense 3D tensors.
"""
import pandas as pd
import numpy as np

class TensorizationModule:
    """
    Converts a tabular DataFrame of (user, item, context, rating) rows
    into a dense 3D tensor R of shape (m, n, l), suitable for multilinear
    decomposition algorithms.
    """
    def __init__(self, user_col='user', item_col='item', context_col='context', rating_col='rating'):
        self.user_col = user_col
        self.item_col = item_col
        self.context_col = context_col
        self.rating_col = rating_col

        self.user_map_ = {}
        self.item_map_ = {}
        self.context_map_ = {}

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Maps categorical IDs to contiguous integer indices and constructs the dense tensor R
        with log(1+x) playcount rescaling.
        """
        users = df[self.user_col].unique()
        items = df[self.item_col].unique()
        contexts = df[self.context_col].unique()

        self.user_map_ = {u: i for i, u in enumerate(users)}
        self.item_map_ = {itm: i for i, itm in enumerate(items)}
        self.context_map_ = {c: i for i, c in enumerate(contexts)}

        tensor_shape = (len(users), len(items), len(contexts))
        R = np.zeros(tensor_shape)

        user_indices = df[self.user_col].map(self.user_map_).values
        item_indices = df[self.item_col].map(self.item_map_).values
        context_indices = df[self.context_col].map(self.context_map_).values
        ratings = np.log1p(df[self.rating_col].values)

        R[user_indices, item_indices, context_indices] = ratings

        return R
