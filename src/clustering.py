#type: ignore

import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

import duckdb as ddb
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import faiss

import importlib
_sfc = importlib.import_module("simple-flow-clustering")
Region = _sfc.Region

load_dotenv()
CLUSTERING_PATH = os.getenv("CLUSTERING_PATH")

MAX_CLUSTERS = 50


def load_region_vectors(region: Region) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed clustering vectors from the Hive-partitioned parquets
    for all departamentos in the given region, pivot them into a feature matrix.

    Returns:
        client_ids: array of unique client ids
        vectors: 2D float32 array of shape (n_clients, n_features)
    """
    parquet_glob = str(Path(CLUSTERING_PATH) / "**" / "*.parquet")
    deps = tuple(region.departamentos)

    df = ddb.query(f"""
        SELECT id, mes, dia_semana, hora, valor
        FROM read_parquet('{parquet_glob}', hive_partitioning=true)
        WHERE departamento IN {deps}
    """).fetchdf()

    if df.empty:
        raise FileNotFoundError(f"No data found for region {region.code}")

    df['feature_key'] = df['mes'].astype(str) + '_' + df['dia_semana'].astype(str) + '_' + df['hora'].astype(str)
    pivot = df.pivot_table(index='id', columns='feature_key', values='valor', aggfunc='mean')
    pivot = pivot.fillna(0.0)

    client_ids = pivot.index.values
    vectors = np.ascontiguousarray(pivot.values, dtype=np.float32)

    return client_ids, vectors


def compute_elbow_curve(vectors: np.ndarray, max_k: int = MAX_CLUSTERS, n_iter: int = 50, seed: int = 42) -> dict[int, float]:
    """
    Run K-Means for k = 2 .. max_k and return the inertia (sum of squared
    distances to the nearest centroid) for each k.
    """
    d = vectors.shape[1]
    inertias: dict[int, float] = {}

    for k in range(2, max_k + 1):
        kmeans = faiss.Kmeans(d, k, niter=n_iter, seed=seed, verbose=False)
        kmeans.train(vectors)

        # Compute squared L2 distances to nearest centroid
        distances, _ = kmeans.index.search(vectors, 1)
        inertia = float(np.sum(distances))
        inertias[k] = inertia
        print(f"  k={k:2d}  inertia={inertia:.4f}")

    return inertias


def plot_elbow(region: Region, inertias: dict[int, float], pdf: PdfPages):
    """Plot the elbow curve for a single region."""
    ks = sorted(inertias.keys())
    vals = [inertias[k] for k in ks]

    plt.figure(figsize=(12, 5))
    plt.plot(ks, vals, "o-", linewidth=2)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (sum of squared distances)")
    plt.title(f"Elbow curve – {region.code}")
    plt.xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def main():
    results_dir = Path("results") / "clustering_elbow"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[int, float]] = {}

    for region in Region:
        print(f"\n=== Region: {region.code} ===")
        client_ids, vectors = load_region_vectors(region)
        print(f"  {len(client_ids)} clients, feature dim {vectors.shape[1]}")

        inertias = compute_elbow_curve(vectors, max_k=MAX_CLUSTERS)
        all_results[region.code] = inertias

    # Per-region elbow plots
    pdf_path = results_dir / "elbow_curves.pdf"
    with PdfPages(pdf_path) as pdf:
        for region in Region:
            plot_elbow(region, all_results[region.code], pdf)

        # Summary page with all regions overlaid
        plt.figure(figsize=(12, 6))
        for region in Region:
            inertias = all_results[region.code]
            ks = sorted(inertias.keys())
            vals = [inertias[k] for k in ks]
            plt.plot(ks, vals, "o-", linewidth=2, label=region.code)
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia (sum of squared distances)")
        plt.title("Elbow curves – all regions")
        plt.xticks(range(2, MAX_CLUSTERS + 1))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"\nElbow curves saved to {pdf_path}")


if __name__ == "__main__":
    main()
