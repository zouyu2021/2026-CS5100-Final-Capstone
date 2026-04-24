import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_iris, load_wine, make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kmeans import KMeans, macqueen_online_kmeans


RESULTS_DIR = ROOT / "results"
FIGS_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)


def fetch_iris_public():

    try:
        bunch = fetch_openml(name="iris", version=1, as_frame=False,
                             parser="liac-arff")
        X = np.asarray(bunch.data, dtype=np.float64)

        _, y = np.unique(bunch.target, return_inverse=True)
        source = "openml (id=iris, v1)"
    except Exception as exc:
        print(f"  [iris] OpenML fetch failed ({exc}); using sklearn copy")
        b = load_iris()
        X, y = b.data.astype(np.float64), b.target
        source = "sklearn bundled UCI copy"
    return X, y, source


def fetch_wine_public():

    try:
        bunch = fetch_openml(name="wine", version=1, as_frame=False,
                             parser="liac-arff")
        X = np.asarray(bunch.data, dtype=np.float64)
        _, y = np.unique(bunch.target, return_inverse=True)
        source = "openml (id=wine, v1)"
    except Exception as exc:
        print(f"  [wine] OpenML fetch failed ({exc}); using sklearn copy")
        b = load_wine()
        X, y = b.data.astype(np.float64), b.target
        source = "sklearn bundled UCI copy"
    return X, y, source


def load_datasets(verbose=True):

    datasets = {}

    X, y, src = fetch_iris_public()
    if verbose:
        print(f"  iris   : {X.shape} from {src}")
    datasets["iris"] = (StandardScaler().fit_transform(X), y, 3)

    X, y, src = fetch_wine_public()
    if verbose:
        print(f"  wine   : {X.shape} from {src}")
    datasets["wine"] = (StandardScaler().fit_transform(X), y, 3)


    X_easy, y_easy = make_blobs(n_samples=500, centers=4, cluster_std=1.0,
                                n_features=2, random_state=0)
    datasets["blobs_easy"] = (X_easy, y_easy, 4)

    X_hard, y_hard = make_blobs(n_samples=1500, centers=15, cluster_std=1.5,
                                n_features=10, random_state=0)
    datasets["blobs_hard"] = (X_hard, y_hard, 15)

    return datasets


def run_once(X, k, init, seed):

    model = KMeans(n_clusters=k, init=init, random_state=seed,
                   max_iter=300, tol=1e-4)
    res = model.fit(X)


    n_unique = len(np.unique(res.labels))
    if n_unique > 1:
        sil = float(silhouette_score(X, res.labels))
    else:
        sil = float("nan")

    return {
        "inertia": res.inertia,
        "n_iter": res.n_iter,
        "converged": res.converged,
        "runtime_sec": res.runtime_sec,
        "silhouette": sil,
        "n_unique_clusters": n_unique,
        "inertia_history": res.inertia_history,
        "labels": res.labels,
    }


def run_sweep(n_seeds=30):

    datasets = load_datasets()
    per_run_rows = []

    for name, (X, y, k) in datasets.items():
        for init in ("random", "k-means++"):
            for seed in range(n_seeds):
                r = run_once(X, k, init, seed)
                ari = float(adjusted_rand_score(y, r["labels"]))
                per_run_rows.append({
                    "dataset": name,
                    "init": init,
                    "seed": seed,
                    "k": k,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "inertia": r["inertia"],
                    "n_iter": r["n_iter"],
                    "converged": r["converged"],
                    "runtime_sec": r["runtime_sec"],
                    "silhouette": r["silhouette"],
                    "ari_vs_true_labels": ari,
                    "n_unique_clusters": r["n_unique_clusters"],
                })
    df = pd.DataFrame(per_run_rows)
    df.to_csv(RESULTS_DIR / "per_run.csv", index=False)

    summary_rows = []
    for (ds, init), group in df.groupby(["dataset", "init"]):
        summary_rows.append({
            "dataset": ds,
            "init": init,
            "inertia_mean": group["inertia"].mean(),
            "inertia_std": group["inertia"].std(),
            "inertia_min": group["inertia"].min(),
            "inertia_max": group["inertia"].max(),
            "n_iter_mean": group["n_iter"].mean(),
            "n_iter_std": group["n_iter"].std(),
            "runtime_mean": group["runtime_sec"].mean(),
            "runtime_std": group["runtime_sec"].std(),
            "silhouette_mean": group["silhouette"].mean(),
            "silhouette_std": group["silhouette"].std(),
            "ari_mean": group["ari_vs_true_labels"].mean(),
            "ari_std": group["ari_vs_true_labels"].std(),
        })
    agg = pd.DataFrame(summary_rows)

    best_hit_rows = []
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        best = ds_df["inertia"].min()
        for init in ("random", "k-means++"):
            sub = ds_df[ds_df["init"] == init]
            frac = float((sub["inertia"] <= best * 1.001).mean())
            best_hit_rows.append({"dataset": ds, "init": init,
                                  "frac_at_best": frac})
    best_hit_df = pd.DataFrame(best_hit_rows)
    agg = agg.merge(best_hit_df, on=["dataset", "init"])

    agg.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return df, agg


def figure_convergence(dataset_name, X, k, n_seeds=20):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, init in zip(axes, ["random", "k-means++"]):
        for seed in range(n_seeds):
            model = KMeans(n_clusters=k, init=init, random_state=seed)
            res = model.fit(X)
            ax.plot(range(1, len(res.inertia_history) + 1),
                    res.inertia_history,
                    alpha=0.5, linewidth=1)
        ax.set_title(f"{dataset_name} -- {init}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("WCSS (inertia)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Convergence of WCSS over {n_seeds} seeds")
    fig.tight_layout()
    path = FIGS_DIR / f"convergence_{dataset_name}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_inertia_distribution(df):

    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4), sharey=False)
    for ax, ds in zip(axes, datasets):
        data = [df[(df["dataset"] == ds) & (df["init"] == init)]["inertia"].values
                for init in ("random", "k-means++")]
        ax.boxplot(data, tick_labels=["random", "k-means++"])
        ax.set_title(ds)
        ax.set_ylabel("final WCSS")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Final WCSS across 30 random seeds")
    fig.tight_layout()
    path = FIGS_DIR / "inertia_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_blobs_clusters(X, k):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, init in zip(axes, ["random", "k-means++"]):
        model = KMeans(n_clusters=k, init=init, random_state=0)
        res = model.fit(X)
        ax.scatter(X[:, 0], X[:, 1], c=res.labels, s=10, cmap="tab10")
        ax.scatter(res.centers[:, 0], res.centers[:, 1],
                   marker="X", c="black", s=120, edgecolors="white")
        ax.set_title(f"{init}\nWCSS={res.inertia:.1f}, iter={res.n_iter}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Clustering of the synthetic blobs dataset (seed=0)")
    fig.tight_layout()
    path = FIGS_DIR / "blobs_clusters.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_macqueen_illustration(X, k):

    batch = KMeans(n_clusters=k, init="random", random_state=0).fit(X)
    online = macqueen_online_kmeans(X, k=k, random_state=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pairs = [("batch (Lloyd)", batch), ("online (MacQueen 1967)", online)]
    for ax, (name, res) in zip(axes, pairs):
        ax.scatter(X[:, 0], X[:, 1], c=res.labels, s=10, cmap="tab10")
        ax.scatter(res.centers[:, 0], res.centers[:, 1],
                   marker="X", c="black", s=120, edgecolors="white")
        ax.set_title(f"{name}\nWCSS={res.inertia:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Batch vs MacQueen-style online k-means (blobs, seed=0)")
    fig.tight_layout()
    path = FIGS_DIR / "macqueen_online_vs_batch.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path



def main():
    print("Running k-means experiments ...")
    df, agg = run_sweep(n_seeds=30)

    print("\n=== Summary (mean +/- std over 30 seeds) ===")
    for _, row in agg.iterrows():
        print(
            f"  {row['dataset']:>15s} | {row['init']:>10s} | "
            f"WCSS={row['inertia_mean']:9.3f}+/-{row['inertia_std']:6.3f} "
            f"[min={row['inertia_min']:9.3f}, max={row['inertia_max']:9.3f}] | "
            f"iter={row['n_iter_mean']:5.2f}+/-{row['n_iter_std']:4.2f} | "
            f"sil={row['silhouette_mean']:.3f} | "
            f"ARI={row['ari_mean']:.3f} | "
            f"best-hit={row['frac_at_best']:.2f} | "
            f"t={row['runtime_mean']*1000:.2f}ms"
        )

    datasets = load_datasets()
    for name, (X, y, k) in datasets.items():
        figure_convergence(name, X, k, n_seeds=15)
    figure_inertia_distribution(df)
    blobs_X, _, blobs_k = datasets["blobs_easy"]
    figure_blobs_clusters(blobs_X, blobs_k)
    figure_macqueen_illustration(blobs_X, blobs_k)

    print(f"\nSaved per_run.csv and summary.csv under {RESULTS_DIR}")
    print(f"Saved figures under {FIGS_DIR}")


if __name__ == "__main__":
    main()
