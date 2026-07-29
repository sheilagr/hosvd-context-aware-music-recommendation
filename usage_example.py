"""
Worked usage example from Chapter 4, Section "A Worked Usage Example".

Reproduces the five steps of Code 8 in the thesis using a small synthetic
dataset (6 users, 5 items, 2 contexts, 14 observed interactions) so that
every intermediate result can be displayed in full and compared against
the values reported in the thesis text.

Requires pytensorlab to produce the real HOSVD decomposition. If it is
not installed, the script exits with an error rather than falling back
to placeholder values, because the purpose of this example is exact
reproducibility of the numbers in Chapter 4.

Run from the repository root folder:
    python usage_example.py
"""
import sys
import numpy as np
import pandas as pd

try:
    import pytensorlab
except ImportError:
    print("ERROR: pytensorlab is required to reproduce this example.")
    print("Install it with:  pip install pytensorlab")
    sys.exit(1)

from TensorRecommendationSystem import TensorizationModule, TuckerRecommender


# Step 1. Raw interaction table: 6 users, 5 items, 2 contexts
df = pd.DataFrame({
    'user':    ['alice', 'alice', 'bob', 'bob', 'carol', 'carol',
                'dave', 'dave', 'erin', 'erin', 'frank', 'frank',
                'alice', 'bob'],
    'item':    ['song_a', 'song_b', 'song_a', 'song_c', 'song_b', 'song_d',
                'song_c', 'song_e', 'song_a', 'song_d', 'song_b', 'song_e',
                'song_e', 'song_d'],
    'context': ['Morning', 'Evening', 'Morning', 'Evening', 'Morning', 'Evening',
                'Morning', 'Evening', 'Morning', 'Evening', 'Morning', 'Evening',
                'Evening', 'Morning'],
    'rating':  [5, 2, 3, 7, 4, 1, 6, 2, 3, 5, 2, 8, 4, 3]
})

print("Step 1: Raw interaction table")
print(df.to_string(index=False))
print()


# Step 2. Build the dense tensor R
ingestor = TensorizationModule()
R = ingestor.fit_transform(df)

print("Step 2: Tensorization")
print(f"  Tensor shape: {R.shape}")
print(f"  User map:    {ingestor.user_map_}")
print(f"  Item map:    {ingestor.item_map_}")
print(f"  Context map: {ingestor.context_map_}")

alice_idx = ingestor.user_map_['alice']
song_a_idx = ingestor.item_map_['song_a']
morning_idx = ingestor.context_map_['Morning']
val = R[alice_idx, song_a_idx, morning_idx]
print(f"  R[alice, song_a, Morning] = log(1+5) = {val:.6f}")
print()


# Step 3. Fit the Tucker/HOSVD model at rank (3, 3, 2)
model = TuckerRecommender(rank_user=3, rank_item=3, rank_context=2)
model.fit(R)

print("Step 3: Tucker/HOSVD decomposition")
print(f"  Core tensor shape:    {model.core_tensor_.shape}")
print(f"  factor_user_ shape:   {model.factor_user_.shape}")
print(f"  factor_item_ shape:   {model.factor_item_.shape}")
print(f"  factor_context_ shape:{model.factor_context_.shape}")

# Verify orthonormality of factor_user_
UtU = model.factor_user_.T @ model.factor_user_
print(f"  U^T U (should be I):\n{np.array2string(UtU, precision=10)}")
print()


# Step 4. Predict the user-item matrix for the Morning context
morning = ingestor.context_map_['Morning']
R_morning = model.predict_context(morning)

print("Step 4: Contextual prediction for Morning")
print(f"  Predicted matrix shape: {R_morning.shape}")
print(f"  R_morning[alice, song_a] = {R_morning[alice_idx, song_a_idx]:.4f}")
print()


# Step 5. Interrogate the core tensor for dominant latent interactions
top_3 = model.get_top_interactions(n=3)

print("Step 5: Top 3 core tensor entries by absolute magnitude")
for rank, (coord, weight) in enumerate(top_3, 1):
    print(f"  {rank}. g[{coord[0]},{coord[1]},{coord[2]}] = {weight:.4f}")
