import time

import numpy as np


def init_random(X, k, rng):

    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    return X[idx].copy()


def init_kmeans_pp(X, k, rng):

    n, d = X.shape
    centers = np.zeros((k, d))

    first = rng.integers(0, n)
    centers[0] = X[first]

    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0:

            centers[i] = X[rng.integers(0, n)]
        else:
            probs = closest_sq / total
            next_idx = rng.choice(n, p=probs)
            centers[i] = X[next_idx]

            new_sq = np.sum((X - centers[i]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_sq)

    return centers


class KMeansResult:

    def __init__(self, labels, centers, inertia, n_iter, converged,
                 runtime_sec, inertia_history):
        self.labels = labels
        self.centers = centers
        self.inertia = inertia
        self.n_iter = n_iter
        self.converged = converged
        self.runtime_sec = runtime_sec
        self.inertia_history = inertia_history


class KMeans:
    def __init__(self, n_clusters, init="random", max_iter=300,
                 tol=1e-4, random_state=None):
        if init not in ("random", "k-means++"):
            raise ValueError("init must be 'random' or 'k-means++'")
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def pairwise_sq_dist(self, X, C):

        diff = X[:, None, :] - C[None, :, :]
        return np.sum(diff * diff, axis=2)

    def seed(self, X, rng):

        if self.init == "random":
            return init_random(X, self.n_clusters, rng)
        return init_kmeans_pp(X, self.n_clusters, rng)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        centers = self.seed(X, rng)
        history = []
        converged = False

        start = time.perf_counter()
        for it in range(1, self.max_iter + 1):

            dists_sq = self.pairwise_sq_dist(X, centers)
            labels = np.argmin(dists_sq, axis=1)
            inertia = float(dists_sq[np.arange(len(X)), labels].sum())
            history.append(inertia)

            new_centers = centers.copy()
            for j in range(self.n_clusters):
                mask = labels == j
                if mask.any():
                    new_centers[j] = X[mask].mean(axis=0)
                else:

                    point_dists = dists_sq[np.arange(len(X)), labels]
                    far_idx = int(np.argmax(point_dists))
                    new_centers[j] = X[far_idx]

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift <= self.tol:
                converged = True
                break

        runtime = time.perf_counter() - start

        dists_sq = self.pairwise_sq_dist(X, centers)
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


def macqueen_online_kmeans(X, k, random_state=None):

    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    order = rng.permutation(len(X))
    X_stream = X[order]

    centers = X_stream[:k].copy()
    counts = np.ones(k, dtype=int)
    labels = np.zeros(len(X_stream), dtype=int)
    labels[:k] = np.arange(k)

    start = time.perf_counter()
    for t in range(k, len(X_stream)):
        x = X_stream[t]

        diffs = centers - x
        j = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        labels[t] = j
        counts[j] += 1

        centers[j] = centers[j] + (x - centers[j]) / counts[j]
    runtime = time.perf_counter() - start


    out_labels = np.zeros_like(labels)
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
