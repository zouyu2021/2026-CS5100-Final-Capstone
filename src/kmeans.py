from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

def _init_random(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:

    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    return X[idx].copy()


def _init_kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:

    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)

    first = rng.integers(0, n)
    centers[0] = X[first]

    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0.0:
            centers[i] = X[rng.integers(0, n)]
        else:
            probs = closest_sq / total
            next_idx = rng.choice(n, p=probs)
            centers[i] = X[next_idx]
            new_sq = np.sum((X - centers[i]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_sq)

    return centers


@dataclass
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int
    converged: bool
    runtime_sec: float
    inertia_history: List[float] = field(default_factory=list)


class KMeans:

    def __init__(
        self,
        n_clusters: int,
        init: str = "random",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        if init not in ("random", "k-means++"):
            raise ValueError(f"unknown init strategy: {init}")
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state


    @staticmethod
    def _pairwise_sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - C[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _seed(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.init == "random":
            return _init_random(X, self.n_clusters, rng)
        return _init_kmeans_pp(X, self.n_clusters, rng)


    def fit(self, X: np.ndarray) -> KMeansResult:
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        centers = self._seed(X, rng)
        history: List[float] = []
        converged = False

        t0 = time.perf_counter()
        for it in range(1, self.max_iter + 1):
            dists_sq = self._pairwise_sq_dist(X, centers)
            labels = np.argmin(dists_sq, axis=1)
            inertia = float(dists_sq[np.arange(len(X)), labels].sum())
            history.append(inertia)

            new_centers = centers.copy()
            for j in range(self.n_clusters):
                mask = labels == j
                if mask.any():
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    far_idx = int(np.argmax(dists_sq[np.arange(len(X)), labels]))
                    new_centers[j] = X[far_idx]

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift <= self.tol:
                converged = True
                break

        runtime = time.perf_counter() - t0

        dists_sq = self._pairwise_sq_dist(X, centers)
        labels = np.argmin(dists_sq, axis=1)
        inertia = float(dists_sq[np.arange(len(X)), labels].sum())

        return KMeansResult(
            labels=labels,
            centers=centers,
            inertia=inertia,
            n_iter=it,
            converged=converged,
            runtime_sec=runtime,
            inertia_history=history,
        )


def macqueen_online_kmeans(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> KMeansResult:
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    order = rng.permutation(len(X))
    X_stream = X[order]

    centers = X_stream[:k].copy()
    counts = np.ones(k, dtype=np.int64)
    labels = np.empty(len(X_stream), dtype=np.int64)
    labels[:k] = np.arange(k)

    t0 = time.perf_counter()
    for t in range(k, len(X_stream)):
        x = X_stream[t]
        diffs = centers - x
        j = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        labels[t] = j
        counts[j] += 1
        centers[j] = centers[j] + (x - centers[j]) / counts[j]
    runtime = time.perf_counter() - t0

    out_labels = np.empty_like(labels)
    out_labels[order] = labels

    diffs = X[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    inertia = float(dists_sq[np.arange(len(X)), out_labels].sum())

    return KMeansResult(
        labels=out_labels,
        centers=centers,
        inertia=inertia,
        n_iter=1,
        converged=True,
        runtime_sec=runtime,
        inertia_history=[inertia],
    )
