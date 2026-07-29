# Code Companion for the Thesis

This folder contains the code used to produce the results described in the thesis *"The Higher Order SVD and its Application to Context-Aware Music Recommendation Systems."* It is provided so that every result in the thesis can be reproduced and verified step by step.

## 1. What is in this folder

- **`TensorRecommendationSystem/`** — a self-contained Python package that implements the Tucker decomposition and the context-based recommendation logic described in Chapter 4.
- **`hosvd_practical.ipynb`** — a Jupyter notebook that downloads the Last.fm dataset used in Chapter 3, builds the tensor, computes the decomposition, and reproduces all the numerical results reported there, including the empirical verification of the HOSVD error bound.
- **`usage_example.py`** — a script that reproduces the worked usage example of Chapter 4: it builds a small synthetic tensor (6 users, 5 items, 2 contexts), fits the Tucker/HOSVD model, and prints every intermediate result reported in the thesis.
- **`analytical_examples/`** — three scripts that recompute the numerical examples of Chapter 2 (the rank-3 / border-rank-2 tensor and its convergent sequence) so the arithmetic can be checked independently of the thesis text.
- **`requirements.txt`** — the list of external dependencies with the exact versions used to produce the thesis results.

## 2. How to run this code

1. Install Python 3.10 or later.
2. Open a terminal and navigate to this folder.
3. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the unit tests:
   ```
   cd TensorRecommendationSystem
   python -m unittest tests
   ```
   A line of dots followed by `OK` confirms that all the properties described in Section 3 below were verified.
6. Open the main notebook:
   ```
   jupyter notebook hosvd_practical.ipynb
   ```
   Each cell can be run in order with Shift+Enter. The first run downloads the ~640 MB Last.fm dataset, which may take a few minutes.
7. Run the Chapter 4 usage example:
   ```
   python usage_example.py
   ```
   This prints the tensor shape, factor matrix dimensions, predicted user-item matrix for the Morning context, and the top-3 core tensor entries. The output should match the values reported in Chapter 4.

## 3. What each unit test verifies

The file `TensorRecommendationSystem/tests.py` contains four tests:

- **`test_tensorization_module`** — confirms that raw (user, item, context, rating) rows are correctly converted into the tensor with the $\log(1+x)$ rescaling described in Chapter 3.
- **`test_recommender_fit_predict`** — confirms that extracting a prediction for a given context returns a matrix of the expected user × item dimensions, corresponding to $\tilde{R}_k = U G_k V^T$ in Chapter 3.
- **`test_factor_orthogonality`** — verifies the orthonormality of the factor matrices ($U^TU = I$, $V^TV = I$, $W^TW = I$) to a precision of $10^{-10}$, the defining property of the HOSVD. When `pytensorlab` is not installed the test applies this check only to $U$ (which comes from a standard SVD); the remaining factors are placeholders in that case (see Chapter 4).
- **`test_get_top_interactions`** — confirms that the code correctly identifies and sorts the largest entries of the core tensor $\mathcal{G}$ by absolute magnitude, used in Chapter 3 to identify dominant latent interactions.

## 4. Where the Chapter 3 numbers come from

Every number quoted in Chapter 3 (Recall@10, NDCG@10, singular value spectra, retained-energy percentages, and the HOSVD error-bound verification) was produced by running `hosvd_practical.ipynb` end to end with a fixed random seed. Running the notebook again reproduces these numbers exactly.

## 5. Authorship

All code in this folder was written by the thesis author. The only external dependencies are the general-purpose packages listed in `requirements.txt`.
