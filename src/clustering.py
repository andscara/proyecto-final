#type: ignore

import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

import json
import duckdb as ddb
from matplotlib import pyplot as plt
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


def plot_from_json(json_path: str):
    """
    Given the path to a JSON file saved by main(), generate elbow curve plots
    as a PNG image in the same directory.
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        all_results: dict[str, dict[str, float]] = json.load(f)

    output_dir = json_path.parent

    # One PNG per region
    for region_code, inertias in all_results.items():
        ks = sorted(inertias.keys(), key=int)
        vals = [inertias[k] for k in ks]
        ks_int = [int(k) for k in ks]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ks_int, vals, "o-", linewidth=2)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia (sum of squared distances)")
        ax.set_title(f"Elbow curve – {region_code}")
        ax.set_xticks(ks_int)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out = output_dir / f"elbow_{region_code}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  {out}")

    # Summary PNG with all regions overlaid
    fig, ax = plt.subplots(figsize=(12, 6))
    for region_code, inertias in all_results.items():
        ks = sorted(inertias.keys(), key=int)
        vals = [inertias[k] for k in ks]
        ks_int = [int(k) for k in ks]
        ax.plot(ks_int, vals, "o-", linewidth=2, label=region_code)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (sum of squared distances)")
    ax.set_title("Elbow curves – all regions")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = output_dir / "elbow_all_regions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


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

    # Save results as JSON
    json_path = results_dir / "elbow_results.json"
    # JSON keys must be strings, convert int keys
    json_data = {
        region_code: {str(k): v for k, v in inertias.items()}
        for region_code, inertias in all_results.items()
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate plots from the saved JSON
    plot_from_json(str(json_path))


if __name__ == "__main__":
    main()
