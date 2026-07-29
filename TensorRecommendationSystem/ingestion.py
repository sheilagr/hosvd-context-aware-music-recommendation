"""
This file builds the third-order tensor R described in the thesis,
Chapter 3, Section "The User-Item-Context Tensor Formulation".

The input is a plain table with one row per observed interaction
(one column for the user, one for the item, one for the context, one
for the rating/playcount). The output is a dense three-dimensional
array R with shape (m, n, l): m users, n items, l contexts.
"""
import pandas as pd
import numpy as np

class TensorizationModule:
    """
    Converts a tabular DataFrame of (user, item, context, rating) rows
    into the dense tensor R in R^(m x n x l) used throughout the thesis,
    suitable as input to the Tucker/HOSVD decomposition (TuckerRecommender).
    """
    def __init__(self, user_col='user', item_col='item', context_col='context', rating_col='rating'):
        self.user_col = user_col
        self.item_col = item_col
        self.context_col = context_col
        self.rating_col = rating_col

        # These dictionaries record the mapping from the original
        # (arbitrary) user/item/context labels to the contiguous
        # 0..m-1 / 0..n-1 / 0..l-1 indices used as tensor coordinates.
        self.user_map_ = {}
        self.item_map_ = {}
        self.context_map_ = {}

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Maps categorical IDs to contiguous indices and builds the dense
        tensor R, applying the log(1+x) rescaling of raw counts used in
        Chapter 3 (log1p keeps the tensor entries non-negative and
        compresses the heavy-tailed distribution of raw play counts).
        """
        # Create mappings: one contiguous integer index per distinct
        # user / item / context label.
        users = df[self.user_col].unique()
        items = df[self.item_col].unique()
        contexts = df[self.context_col].unique()

        self.user_map_ = {u: i for i, u in enumerate(users)}
        self.item_map_ = {itm: i for i, itm in enumerate(items)}
        self.context_map_ = {c: i for i, c in enumerate(contexts)}

        # Initialize the dense tensor R with shape (m, n, l)
        tensor_shape = (len(users), len(items), len(contexts))
        R = np.zeros(tensor_shape)

        # Map each row's categorical IDs to their tensor coordinates
        user_indices = df[self.user_col].map(self.user_map_).values
        item_indices = df[self.item_col].map(self.item_map_).values
        context_indices = df[self.context_col].map(self.context_map_).values
        ratings = np.log1p(df[self.rating_col].values)

        # Populate the tensor in one vectorized assignment (NumPy advanced
        # indexing) instead of looping row by row over the DataFrame.
        R[user_indices, item_indices, context_indices] = ratings

        return R
