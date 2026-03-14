#type: ignore

# Fix duplicate libomp crash (PyTorch + faiss-cpu both bundle OpenMP on macOS)
# check for mac os and set env vars to avoid crash. See
import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

from datetime import datetime
from tabnanny import check
import time
import duckdb as ddb
from matplotlib import pyplot as plt
from numpy import c_
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from forecasting.autoformer.autoformer import Autoformer
from forecasting.autoformer.prediction_window import PredictionWindow
from forecasting.autoformer.trainer import Trainer
from torch.utils import data
from forecasting.autoformer.data_loader import data_splitter
from pathlib import Path
from dotenv import load_dotenv
import horizon as h
import numpy as np
from typing import List
import faiss
import json
from enum import Enum


load_dotenv()
PATH = os.getenv("DATA_PATH")
WINDOW_SIZE = 24*7*2 # 2 weeks
HORIZON = h.Horizon(type = h.HorizonType.HOUR, length=24*7) # 1 week
BATCH_SIZE = 64
LABEL_LEN = WINDOW_SIZE

EXOG_COLS = ['temp_max', 'temp_min', 'temp_media']

CLUSTERING_PATH = os.getenv("CLUSTERING_PATH")
CLUSTERING_RESULTS_PATH = Path('results') / 'clustering' / 'clustering_assignments.json'

class Region(Enum):
    NORTH = ("NORTH", ["ARTIGAS", "SALTO", "RIVERA", "TACUAREMBO", "CERRO LARGO"])
    SOUTH = ("SOUTH", ["SAN JOSE", "COLONIA", "CANELONES", "FLORES", "FLORIDA", "SORIANO"])
    EAST = ("EAST", ["MALDONADO", "ROCHA", "TREINTA Y TRES", "LAVALLEJA"])
    WEST = ("WEST", ["PAYSANDU","RIO NEGRO", "DURAZNO"])
    MONTEVIDEO = ("MONTEVIDEO", ["MONTEVIDEO"])

    def __init__(self, code: str, departamentos: list[str]):
        self.code = code
        self.departamentos = departamentos

def print_test_metrics(
    predictions: List[PredictionWindow],
    prefix: str,
    pdf: PdfPages,
):
    # Calculate the error metrics for the aggregated predictions and print them in the pdf as text, using the global_prediction_windows to calculate the error metrics for each window and then averaging them to get the global error metrics
    errors = PredictionWindow.calculate_error_metrics(predictions)
    plt.figure(figsize=(10, 6))
    metrics_pattern = f"""
    {prefix} MSE: {errors['mse']:.4f} 
    {prefix} MIN MSE: {errors['min_mse']:.4f} 
    {prefix} MAX MSE: {errors['max_mse']:.4f}\n
    {prefix} MAE: {errors['mae']:.4f} 
    {prefix} MIN MAE: {errors['min_mae']:.4f} 
    {prefix} MAX MAE: {errors['max_mae']:.4f}\n
    {prefix} MAPE: {errors['mape']:.2f} 
    {prefix} MIN MAPE: {errors['min_mape']:.2f} 
    {prefix} MAX MAPE: {errors['max_mape']:.2f}\n
    """

    plt.text(0.1, 0.5, metrics_pattern, fontsize=12)
    plt.title(f"{prefix} Error Metrics", fontsize=16)
    plt.axis('off')
    pdf.savefig()
    plt.close()


def build_client_vectors(con: ddb.DuckDBPyConnection, region: Region) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed clustering vectors from the Hive-partitioned parquets
    for all departamentos in the given region, pivot them into a feature matrix.

    Returns:
        client_ids: array of unique client ids
        vectors: 2D float32 array of shape (n_clients, n_features)
    """
    parquet_glob = str(Path(CLUSTERING_PATH) / "**" / "*.parquet")
    deps = tuple(region.departamentos)

    df = con.query(f"""
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


def run_kmeans(vectors: np.ndarray, n_clusters: int, n_iter: int = 50, seed: int = 42) -> np.ndarray:
    """
    Run K-Means clustering using FAISS.

    Args:
        vectors: float32 array of shape (n_samples, n_features)
        n_clusters: number of clusters to create
        n_iter: number of K-Means iterations
        seed: random seed for reproducibility

    Returns:
        assignments: array of cluster labels for each sample
    """
    d = vectors.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, seed=seed, verbose=True)
    kmeans.train(vectors)

    # Assign each vector to the nearest centroid
    _, assignments = kmeans.index.search(vectors, 1)
    return assignments.flatten()


def save_clustering_results(
    all_assignments: dict[str, tuple[np.ndarray, np.ndarray]],
    region_clusters: dict[Region, int],
):
    """Save clustering assignments for all regions to a JSON file."""
    data = {}
    for region, n_clusters in region_clusters.items():
        client_ids, assignments = all_assignments[region.code]
        data[region.code] = {
            "n_clusters": n_clusters,
            "client_ids": [int(cid) if isinstance(cid, (np.integer,)) else cid for cid in client_ids.tolist()],
            "assignments": assignments.tolist(),
        }
    CLUSTERING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLUSTERING_RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Clustering assignments saved to {CLUSTERING_RESULTS_PATH}")


def load_clustering_results() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load clustering assignments from the JSON file. Returns dict region_code -> (client_ids, assignments)."""
    with open(CLUSTERING_RESULTS_PATH) as f:
        data = json.load(f)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region_code, info in data.items():
        client_ids = np.array(info["client_ids"])
        assignments = np.array(info["assignments"])
        result[region_code] = (client_ids, assignments)
    print(f"Clustering assignments loaded from {CLUSTERING_RESULTS_PATH}")
    return result


def get_cluster_timeseries(con: ddb.DuckDBPyConnection, client_ids: np.ndarray, assignments: np.ndarray) -> dict[int, pd.DataFrame]:
    """
    For each cluster, aggregate the hourly time series of all clients
    belonging to that cluster into a single summed time series,
    joined with temperature data.

    Returns:
        dict mapping cluster_id -> DataFrame with columns (dia, hora, agg_valor, temp_max, temp_min, temp_media)
    """
    cluster_map = pd.DataFrame({'id': client_ids, 'cluster': assignments})
    unique_clusters = sorted(cluster_map['cluster'].unique())

    cluster_ts: dict[int, pd.DataFrame] = {}
    # Load cluster_map into DuckDB for joining
    con.register('cluster_map', cluster_map)
    for cluster_id in unique_clusters:
        ids_in_cluster = cluster_map[cluster_map['cluster'] == cluster_id]['id'].tolist()
        query = f"""
        SELECT e.dia, e.hora, SUM(agg_valor) AS agg_valor,
               AVG((temp_max + 15) / 65) AS temp_max,
               AVG((temp_min + 15) / 65) AS temp_min,
               AVG((temp_media + 15) / 65) AS temp_media
        FROM (
            SELECT departamento, dia, hora, SUM(valor) AS agg_valor
            FROM read_parquet('{PATH}') i inner JOIN cluster_map cm ON i.id = cm.id
            WHERE cm.cluster = {cluster_id}
            GROUP BY departamento, dia, hora
        ) e INNER JOIN temperatura_departamento t ON e.dia = t.dia AND e.departamento = t.departamento
        GROUP BY e.dia, e.hora
        ORDER BY e.dia, e.hora
        """
        ts_df = con.execute(query).fetchdf()
        cluster_ts[cluster_id] = ts_df
        print(f"Cluster {cluster_id}: {len(ids_in_cluster)} clients, {len(ts_df)} timesteps")

    return cluster_ts


def train_and_predict_cluster(
    cluster_id: int,
    cluster_data: pd.DataFrame,
    region: Region,
    train: bool,
    execution_id: str,
) -> tuple[np.ndarray, np.ndarray, List[PredictionWindow]]:
    """Train/predict Autoformer for a single cluster. Returns predictions without generating PDFs."""
    print(f"\n--- Processing {region.code} / Cluster {cluster_id} ({len(cluster_data)} records) ---")

    all_dataset, train_dataset, val_dataset, test_dataset = data_splitter(
        df=cluster_data,
        windows_size=WINDOW_SIZE,
        horizon=HORIZON,
        label_len=LABEL_LEN,
        stride=24,
        target_col_name="agg_valor",
        scale=True,
        exog_cols=EXOG_COLS
    )

    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    seq_len = WINDOW_SIZE
    pred_len = HORIZON.length
    model = Autoformer(
        seq_len=seq_len,
        label_len=LABEL_LEN,
        pred_len=pred_len,
        c_out=1,
        enc_in=1,
        dec_in=1,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        e_layers=3,
        d_layers=2,
        dropout=0,
        factor=2,
        d_mark=8
    )
    trainer = Trainer(
        model=model,
        window_stride_in_days=1,
        all_dataset=all_dataset,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        seq_len=seq_len,
        label_len=LABEL_LEN,
        pred_len=pred_len,
        output_attention=False,
        device_name=os.getenv("DEVICE_NAME")
    )

    checkpoint_path = Path("checkpoints") / execution_id / f"{region.code}" / f"cluster_{cluster_id}"
    patience = 50
    lr = 0.00001
    train_epochs = 300
    setting = 'patience_{}_lr_{}_epochs_{}'.format(patience, lr, train_epochs)
    path = checkpoint_path / setting
    if not os.path.exists(path):
        os.makedirs(path)

    if train:
        print(f"Starting training for {region.code} / cluster {cluster_id}...")
        trainer.train(
            patience=patience,
            verbose=True,
            learning_rate=lr,
            train_epochs=train_epochs,
            checkpoint_path=path
        )
        print("Training finished.")

    print(f"Running predictions for {region.code} / cluster {cluster_id}...")
    plots_path = Path('results') / 'clustering' / f'{region.code}' / f'cluster_{cluster_id}_graficas.pdf'
    plots_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(plots_path) as pdf:
        trainer.predict_series(
            pdf=pdf,
            checkpoint_path=path,
            rolling_step=0,
            load=not train
        )
        y_preds_flat, y_reals_flat = trainer.predict_series(
            pdf=pdf,
            checkpoint_path=path,
            rolling_step=24 * 1,
            load=not train
        )
        prediction_windows: List[PredictionWindow] = trainer.predict_windows(
            checkpoint_path=path,
            rolling_step=0,
            load=not train
        )
        Trainer.plot_prediction_windows(
            pdf=pdf,
            prediction_windows=prediction_windows,
            rolling_step=0
        )
    print(f"Cluster PDF saved: {plots_path}")

    return y_preds_flat, y_reals_flat, prediction_windows


def main(
    train: bool = True,
    run_clustering: bool = True,
    region_clusters: dict[Region, int] = None
):
    if region_clusters is None:
        region_clusters = {region: 5 for region in Region}

    execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Execution ID: {execution_id}")

    con = ddb.connect(database=os.getenv("DB_PATH"))

    # ── Phase 1: K-Means clustering (or load from file) ──
    all_assignments: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if run_clustering:
        for region, n_clusters in region_clusters.items():
            print(f"\n{'='*60}")
            print(f"  Region: {region.code} ({n_clusters} clusters)")
            print(f"{'='*60}")

            print("Building client feature vectors...")
            client_ids, vectors = build_client_vectors(con, region)
            print(f"Built vectors for {len(client_ids)} clients, feature dimension: {vectors.shape[1]}")

            print(f"Running FAISS K-Means with {n_clusters} clusters...")
            assignments = run_kmeans(vectors, n_clusters)
            print(f"Clustering complete. Cluster distribution:")
            unique, counts = np.unique(assignments, return_counts=True)
            for c, cnt in zip(unique, counts):
                print(f"  Cluster {c}: {cnt} clients")

            all_assignments[region.code] = (client_ids, assignments)

        save_clustering_results(all_assignments, region_clusters)
    else:
        all_assignments = load_clustering_results()

    # ── Phase 2: Build time series + training/prediction for all regions ──
    region_results: dict[Region, list[tuple[np.ndarray, np.ndarray, List[PredictionWindow]]]] = {}

    for region in region_clusters:
        client_ids, assignments = all_assignments[region.code]

        print(f"\nBuilding aggregated time series for {region.code}...")
        cluster_ts = get_cluster_timeseries(con, client_ids, assignments)

        cluster_results = []
        for cluster_id, cluster_data in cluster_ts.items():
            result = train_and_predict_cluster(cluster_id, cluster_data, region, train, execution_id)
            cluster_results.append(result)

        region_results[region] = cluster_results

    con.close()

    # ── Phase 2: Generate PDFs ──
    global_prediction_windows: List[PredictionWindow] = []
    global_y_preds_list = []
    global_y_reals_list = []

    for region, cluster_results in region_results.items():
        region_prediction_windows: List[PredictionWindow] = []
        region_y_preds_list = []
        region_y_reals_list = []

        # Generate per-cluster PDF and aggregate into region
        for y_preds, y_reals, prediction_windows in cluster_results:
            region_y_preds_list.append(y_preds)
            region_y_reals_list.append(y_reals)

            if len(region_prediction_windows) == 0:
                region_prediction_windows = prediction_windows
            else:
                region_prediction_windows = [
                    pw_region.aggregate(pw_cluster)
                    for pw_region, pw_cluster in zip(region_prediction_windows, prediction_windows)
                ]

        # Aggregate region into global
        global_y_preds_list.append(np.array(region_y_preds_list).sum(axis=0))
        global_y_reals_list.append(np.array(region_y_reals_list).sum(axis=0))

        if len(global_prediction_windows) == 0:
            global_prediction_windows = region_prediction_windows
        else:
            global_prediction_windows = [
                pw_global.aggregate(pw_region)
                for pw_global, pw_region in zip(global_prediction_windows, region_prediction_windows)
            ]

        # Region PDF: sum of predictions across clusters
        region_y_preds = np.array(region_y_preds_list).sum(axis=0).flatten()
        region_y_reals = np.array(region_y_reals_list).sum(axis=0).flatten()

        plots_path = Path('results') / 'clustering' / f'{region.code}' / 'graficas.pdf'
        plots_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(plots_path) as pdf:
            print_test_metrics(
                predictions=region_prediction_windows,
                prefix=f"{region.code}",
                pdf=pdf
            )
            Trainer.plot_and_print_ys(
                pdf=pdf,
                y_preds_flat=region_y_preds,
                y_reals_flat=region_y_reals,
                rolling_step=24 * 1
            )
            Trainer.plot_prediction_windows(
                pdf=pdf,
                prediction_windows=region_prediction_windows,
                rolling_step=0
            )
        print(f"Region PDF saved: {plots_path}")

    # Global PDF: sum of predictions across all regions
    global_y_preds = np.array(global_y_preds_list).sum(axis=0).flatten()
    global_y_reals = np.array(global_y_reals_list).sum(axis=0).flatten()

    plots_path = Path('results') / 'clustering' / 'global' / 'graficas.pdf'
    plots_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(plots_path) as pdf:
        print_test_metrics(
            predictions=global_prediction_windows,
            prefix="Global",
            pdf=pdf
        )
        Trainer.plot_and_print_ys(
            pdf=pdf,
            y_preds_flat=global_y_preds,
            y_reals_flat=global_y_reals,
            rolling_step=24 * 1
        )
        Trainer.plot_prediction_windows(
            pdf=pdf,
            prediction_windows=global_prediction_windows,
            rolling_step=0
        )
    print(f"Global PDF saved: {plots_path}")


if __name__ == "__main__":
    main(
        train=True,
        run_clustering=True,
        region_clusters={
            Region.NORTH: 5,
            Region.SOUTH: 5,
            Region.EAST: 5,
            Region.WEST: 5,
            Region.MONTEVIDEO: 5,
        }
    )
