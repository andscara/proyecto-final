#type: ignore

# Fix duplicate libomp crash (PyTorch + faiss-cpu both bundle OpenMP on macOS)
# check for mac os and set env vars to avoid crash. See
import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

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
from enum import Enum


load_dotenv()
PATH = os.getenv("DATA_PATH")
WINDOW_SIZE = 24*7*2 # 2 weeks
HORIZON = h.Horizon(type = h.HorizonType.HOUR, length=24*7) # 1 week
BATCH_SIZE = 64
LABEL_LEN = WINDOW_SIZE

EXOG_COLS = ['temp_max', 'temp_min', 'temp_media']

N_CLUSTERS = 5

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
    Build a feature vector per client (id) by pivoting the average consumption
    values across (month, day_of_week, hour) into a flat vector.

    Returns:
        client_ids: array of unique client ids
        vectors: 2D float32 array of shape (n_clients, n_features) suitable for FAISS
    """
    query = f"""
        SELECT id, month(dia) AS mes, dayofweek(dia) AS dia_semana, hora, AVG(valor) AS valor
        FROM (
            SELECT id, dia, hora,
                COALESCE(valor / NULLIF(SUM(valor) OVER (PARTITION BY id, dia), 0), 0) as valor
            FROM read_parquet('{PATH}')
            where departamento in {tuple(region.departamentos)}
        )
        GROUP BY id, mes, dia_semana, hora
        ORDER BY id, mes, dia_semana, hora;
    """
    df = con.execute(query).fetchdf()

    # Pivot: each row becomes a client, columns are (mes, dia_semana, hora) combinations
    df['feature_key'] = df['mes'].astype(str) + '_' + df['dia_semana'].astype(str) + '_' + df['hora'].astype(str)
    pivot = df.pivot_table(index='id', columns='feature_key', values='valor', aggfunc='mean')
    pivot = pivot.fillna(0.0)

    client_ids = pivot.index.values
    vectors = np.ascontiguousarray(pivot.values, dtype=np.float32)

    # Normalize vectors (L2) so that K-Means uses cosine-like distances
    #faiss.normalize_L2(vectors)

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


def main(
    train: bool = True,
    n_clusters: int = N_CLUSTERS
):
    con = ddb.connect(database=os.getenv("DB_PATH"))

    # Step 1: Build feature vectors per client and run FAISS K-Means
    print("Building client feature vectors...")
    client_ids, vectors = build_client_vectors(con, Region.MONTEVIDEO)
    print(f"Built vectors for {len(client_ids)} clients, feature dimension: {vectors.shape[1]}")

    print(f"Running FAISS K-Means with {n_clusters} clusters...")
    assignments = run_kmeans(vectors, n_clusters)
    print(f"Clustering complete. Cluster distribution:")
    unique, counts = np.unique(assignments, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f"  Cluster {c}: {cnt} clients")

    # Step 2: Get aggregated time series per cluster
    print("Building aggregated time series per cluster...")
    cluster_ts = get_cluster_timeseries(con, client_ids, assignments)
    con.close()

    # Step 3: Train/predict Autoformer per cluster (same pattern as simple-flow.py)
    y_preds_flat_results = []
    y_reals_flat_results = []
    global_prediction_windows: List[PredictionWindow] = []

    for cluster_id, cluster_data in cluster_ts.items():
        print(f"\n--- Processing Cluster {cluster_id} ({len(cluster_data)} records) ---")

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

        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False
        )
        val_dataloader = data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )
        test_dataloader = data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )

        seq_len = WINDOW_SIZE
        pred_len = HORIZON.length
        def create_model():
            return Autoformer(
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
            model_factory=create_model,
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

        checkpoint_path = Path("checkpoints") / f"cluster_{cluster_id}"
        patience = 50
        lr = 0.00001
        train_epochs = 300
        setting = 'patience_{}_lr_{}_epochs_{}'.format(patience, lr, train_epochs)
        path = checkpoint_path / setting
        if not os.path.exists(path):
            os.makedirs(path)

        if train:
            print(f"Starting training for cluster {cluster_id}...")
            trainer.train(
                patience=patience,
                verbose=True,
                learning_rate=lr,
                train_epochs=train_epochs,
                checkpoint_path=path
            )
            print("Training finished.")

        print("Starting testing...")
        plots_path = Path('results') / path / f'graficas.pdf'
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
            y_preds_flat_results.append(y_preds_flat)
            y_reals_flat_results.append(y_reals_flat)
            cluster_prediction_windows: List[PredictionWindow] = trainer.predict_windows(
                checkpoint_path=path,
                rolling_step=0,
                load=not train
            )
            if len(global_prediction_windows) == 0:
                global_prediction_windows = cluster_prediction_windows
            else:
                global_prediction_windows = [
                    pw_global.aggregate(pw_cluster)
                    for pw_global, pw_cluster in zip(global_prediction_windows, cluster_prediction_windows)
                ]
            Trainer.plot_prediction_windows(
                pdf=pdf,
                prediction_windows=cluster_prediction_windows,
                rolling_step=0
            )

    # Step 4: Global metrics across all clusters
    plots_path = Path('results') / 'clustering' / f'graficas.pdf'
    plots_path.parent.mkdir(parents=True, exist_ok=True)
    y_preds_flat = np.sum(np.array(y_preds_flat_results), axis=0).flatten()
    y_reals_flat = np.sum(np.array(y_reals_flat_results), axis=0).flatten()
    with PdfPages(plots_path) as pdf:
        print_test_metrics(
            predictions=global_prediction_windows,
            prefix=f"clustering - Global",
            pdf=pdf
        )
        Trainer.plot_and_print_ys(
            pdf=pdf,
            y_preds_flat=y_preds_flat,
            y_reals_flat=y_reals_flat,
            rolling_step=24 * 1
        )
        Trainer.plot_prediction_windows(
            pdf=pdf,
            prediction_windows=global_prediction_windows,
            rolling_step=0
        )


if __name__ == "__main__":
    main(train=True)
