# k-Means Capstone (CS5100, Spring 2026)

Reproduction of **MacQueen (1967)**, *Some Methods for Classification and
Analysis of Multivariate Observations*, extended with a comparison between
random initialization and **k-means++** seeding (Arthur & Vassilvitskii,
2007).

The project implements k-means from scratch in Python, runs 30 seeds per
configuration on two real public datasets (Iris, Wine) plus two synthetic
supplementary datasets, and records WCSS, iterations, runtime, silhouette,
ARI, and a best-hit stability metric.

---

## Project layout

```
.
├── README.md                     # this file
├── requirements.txt              # pinned minimum versions
├── src/
│   ├── __init__.py
│   └── kmeans.py                 # KMeans class (random / k-means++)
│                                 # + macqueen_online_kmeans() online variant
└── experiments/
    └── run_experiments.py        # loads Iris + Wine from OpenML,
                                  # runs 30 seeds per config, writes outputs
```

Running the experiment script creates `results/` and `figures/` folders
with the CSV outputs and PNG plots for that run. These are regenerated
every time the script is executed, so the numbers in the CSVs and the
visuals in the PNGs may differ slightly from run to run (runtime is
wall-clock; WCSS, iterations, silhouette, and ARI are deterministic
given the seed).

---

## Requirements

Python 3.9 or newer with:

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
```

---

## Setup

From the project root:

```bash
cd /path/to/kmeans-capstone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `source .venv/bin/activate` line is the one you run every time you open
a new terminal for this project. To leave the virtual environment, run
`deactivate`.

If you prefer not to use a virtual environment, you can also install the
packages directly into your system Python:

```bash
python3 -m pip install -r requirements.txt
```

---

## Running the experiments

From the project root with the venv activated:

```bash
python experiments/run_experiments.py
```

This will:

1. Fetch **Iris** (OpenML id 61) and **Wine** (OpenML id 187) from
   https://www.openml.org. If OpenML is unreachable, it falls back to the
   bit-identical UCI copies bundled with scikit-learn.
2. Run batch k-means with 30 random seeds for each of `random` and
   `k-means++` on every dataset — 240 runs total.
3. Create `results/per_run.csv` and `results/summary.csv`.
4. Create the PNGs under `figures/`.

Expected runtime: well under one minute on a laptop.

A short summary table prints at the end.

---

## Datasets

| Dataset    | Source                                                                                                   | Offline fallback               |
|------------|----------------------------------------------------------------------------------------------------------|--------------------------------|
| Iris       | OpenML dataset **id = 61**, version 1 — https://www.openml.org/d/61 (Fisher 1936 / UCI)                  | `sklearn.datasets.load_iris()` |
| Wine       | OpenML dataset **id = 187**, version 1 — https://www.openml.org/d/187 (Forina et al. / UCI)              | `sklearn.datasets.load_wine()` |
| blobs-easy | Synthetic — `make_blobs(n_samples=500, centers=4, cluster_std=1.0, n_features=2, random_state=0)`        | n/a                            |
| blobs-hard | Synthetic — `make_blobs(n_samples=1500, centers=15, cluster_std=1.5, n_features=10, random_state=0)`     | n/a                            |

The synthetic sets are used only as supplementary illustrations. All
quantitative claims about real data in the report come from Iris and Wine.

---

## Library usage

The `KMeans` class can be used directly:

```python
import numpy as np
from src.kmeans import KMeans

X = np.load("your_data.npy")

km = KMeans(n_clusters=3, init="k-means++", random_state=0)
res = km.fit(X)

print(res.labels)       # cluster assignments
print(res.centers)      # final cluster centers
print(res.inertia)      # WCSS
print(res.n_iter)       # iterations to convergence
print(res.runtime_sec)  # wall-clock seconds
```

Swap `init="k-means++"` for `init="random"` to compare.

For the single-pass online variant that matches MacQueen's 1967 procedure:

```python
from src.kmeans import macqueen_online_kmeans
res = macqueen_online_kmeans(X, k=3, random_state=0)
```
---
## References

- Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding.* SODA, 1027–1035.
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179–188.
- Forina, M., et al. (1991). *PARVUS.* Source of the UCI Wine Recognition dataset.
- Harris, C. R., et al. (2020). *Array programming with NumPy.* Nature, 585(7825), 357–362.
- MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.* 5th Berkeley Symposium.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
- Sculley, D. (2010). *Web-scale k-means clustering.* WWW, 1177–1178.
- Vanschoren, J., et al. (2014). *OpenML: networked science in machine learning.* SIGKDD Explorations, 15(2), 49–60.
