# Code Companion for the Thesis

This folder contains the code used to produce the results described in the thesis *"The Higher Order SVD and its Application to Context-Aware Music Recommendation Systems."* (Universidad Carlos III de Madrid - UC3M).

It is provided so that every result in the thesis can be reproduced and verified step by step.

---

## 1. What is in this folder

- **`TensorRecommendationSystem/`** — a self-contained Python package that implements the Tucker decomposition and the context-based recommendation logic described in Chapter 4.
- **`hosvd_practical.ipynb`** — a Jupyter notebook that downloads the Last.fm dataset used in Chapter 3, builds the tensor, computes the decomposition, and reproduces all numerical results.
- **`usage_example.py`** — a script that reproduces the worked usage example of Chapter 4: it builds a small synthetic tensor (6 users, 5 items, 2 contexts), fits the Tucker/HOSVD model, and prints every intermediate result reported in the thesis.
- **`analytical_examples/`** — three scripts that recompute the numerical examples of Chapter 2 (the rank-3 / border-rank-2 tensor and its convergent sequence).
- **`requirements.txt`** — the list of external dependencies.

---

## 2. How to Run This Code (by Operating System)

###  macOS (using Homebrew)

1. **Install Python 3 via Homebrew**:
   ```bash
   brew install python git
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run unit tests**:
   ```bash
   python -m unittest TensorRecommendationSystem.tests
   ```

5. **Run the Chapter 4 usage example**:
   ```bash
   python usage_example.py
   ```

6. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook hosvd_practical.ipynb
   ```

---

### 🐧 Linux / Ubuntu

1. **Install Python 3, venv & Git**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip git
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies and run**:
   ```bash
   pip install -r requirements.txt
   python usage_example.py
   ```

---

### 𝄪 Windows

1. **Install Python 3 & Git**:
   - Download Python 3 from [python.org](https://www.python.org/downloads/) (check **"Add Python to PATH"**).
   - Download Git from [git-scm.com](https://git-scm.com/).

2. **Create and activate a virtual environment in PowerShell / Command Prompt**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies and run**:
   ```cmd
   pip install -r requirements.txt
   python usage_example.py
   ```

---

## 3. License and Attribution

**Copyright (c) 2026 Sheila Gallardo Redondo, Universidad Carlos III de Madrid (UC3M)**  
Licensed under the [MIT License](LICENSE).
