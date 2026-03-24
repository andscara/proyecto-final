from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import json
import os
import platform

import duckdb as ddb
import numpy as np
import pandas as pd

from forecasting.autoformer.data_loader import data_splitter
from dataclasses import dataclass
from forecasting.autoformer.data_loader import WindowsDataset
from forecasting.autoformer.experiment_configuration import ExperimentConfiguration

@dataclass
class ExperimentGroup:
    """Data class for grouping experiments."""
    name: str
    full_dataset: WindowsDataset
    train_dataset: WindowsDataset
    val_dataset: WindowsDataset
    test_dataset: WindowsDataset
    raw_df: pd.DataFrame = None  # full time series DataFrame, used by SARIMAXRunner


class BaseExperimentHandler(ABC):

    def __init__(
        self, 
        db_path: str, 
        data_path: str,
        exp_config: ExperimentConfiguration
    ):
        self._db_path = db_path
        self._data_path = data_path
        self._exp_config = exp_config

    @abstractmethod
    def has_next(self) -> bool:
        """
        Check if there are more experiments to run.
        """
        ...

    @abstractmethod
    def next_experiment_group(self) -> ExperimentGroup:
        """
        Get the next experiment group to run.
        """
        ...

    @abstractmethod
    def use_exogenous(self) -> bool:
        ...

class Region(Enum):
    NORTH = ("NORTH", ["ARTIGAS", "SALTO", "RIVERA", "TACUAREMBO", "CERRO LARGO"])
    SOUTH = ("SOUTH", ["SAN JOSE", "COLONIA", "CANELONES", "FLORES", "FLORIDA", "SORIANO"])
    EAST = ("EAST", ["MALDONADO", "ROCHA", "TREINTA Y TRES", "LAVALLEJA"])
    WEST = ("WEST", ["PAYSANDU","RIO NEGRO", "DURAZNO"])
    MONTEVIDEO = ("MONTEVIDEO", ["MONTEVIDEO"])

    def __init__(self, code: str, departamentos: list[str]):
        self.code = code
        self.departamentos = departamentos

class RegionsExperimentHandler(BaseExperimentHandler):

    def __init__(self, db_path: str, data_path: str, exp_config: ExperimentConfiguration):
        super().__init__(db_path, data_path, exp_config)
        self._region_index = 0

    def has_next(self) -> bool:
        return self._region_index < Region.__len__()
    
    def next_experiment_group(self) -> ExperimentGroup:
        region = list(Region)[self._region_index]
        self._region_index += 1
        query = f"""
        select e.dia, e.hora, SUM(agg_valor) as agg_valor, AVG((temperatura + 15) / 65) as temp_media
        from (
            select departamento, dia, hora, SUM(valor) as agg_valor
            from read_parquet('{self._data_path}')
            where departamento in {tuple(region.departamentos)}
            group by departamento, dia, hora
        ) e inner join temp_departamento t on e.dia=t.dia and e.hora=t.hora and t.departamento = e.departamento
        group by e.dia, e.hora
        order by e.dia, e.hora
        """

        con = ddb.connect(database=self._db_path)
        ts_agg_region = con.execute(query).fetchdf()
        print(f"Cantidad de registros totales en todos los departamentos agregados: {len(ts_agg_region)}")
        con.close()
        print ("Creating datasets...")
        all_dataset, train_dataset, val_dataset, test_dataset = data_splitter(
            df=ts_agg_region,
            exp_config=self._exp_config
        )
        print ("Datasets created.")
        return ExperimentGroup(
            name=f"Region {region.code}",
            full_dataset=all_dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            raw_df=ts_agg_region,
        )
    
    def use_exogenous(self) -> bool:
        return True
    
class CountryExperimentHandler(BaseExperimentHandler):

    def __init__(self, db_path: str, data_path: str, exp_config: ExperimentConfiguration):
        super().__init__(db_path, data_path, exp_config)
        self._has_next = True

    def has_next(self) -> bool:
        has_next = self._has_next
        self._has_next = False
        return has_next
    
    def next_experiment_group(self) -> ExperimentGroup:
        query = f"""
        select
            e.dia,
            e.hora,
            SUM(e.valor) as agg_valor,
            AVG((t.temperatura + 15) / 65) as temp_media
        from read_parquet('{self._data_path}') e
        inner join temp_departamento t on e.dia=t.dia and e.hora=t.hora and t.departamento = e.departamento
        group by e.dia, e.hora
        order by e.dia, e.hora
        """

        con = ddb.connect(database=self._db_path)
        ts_agg_region = con.execute(query).fetchdf()
        print(f"Cantidad de registros totales en todo el pais: {len(ts_agg_region)}")
        con.close()
        print ("Creating datasets...")
        print ("Experiment configuration:", self._exp_config)
        all_dataset, train_dataset, val_dataset, test_dataset = data_splitter(
            df=ts_agg_region,
            exp_config=self._exp_config
        )
        print ("Datasets created.")
        return ExperimentGroup(
            name=f"Country",
            full_dataset=all_dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            raw_df=ts_agg_region,
        )
    
    def use_exogenous(self) -> bool:
        return True

class ExperimentType(Enum):
    REGIONS = "regions"
    COUNTRY = "country"
    DEPARTAMENTS = "departments"
    REGION_CLUSTERING = "region_clustering"


class ClusteringExperimentHandler(BaseExperimentHandler):
    """
    Iterates over every (region, cluster) pair, yielding one ExperimentGroup
    per cluster.

    If run_clustering=True, runs FAISS K-Means for each region and saves the
    assignments to clustering_results_path. Otherwise loads existing assignments
    from that file.
    """

    def __init__(
        self,
        db_path: str,
        data_path: str,
        exp_config: ExperimentConfiguration,
        clustering_path: str,
        clustering_results_path: str,
        run_clustering: bool = False,
        n_clusters_per_region: dict[str, int] | None = None,
    ):
        super().__init__(db_path, data_path, exp_config)
        self._clustering_path = clustering_path
        self._clustering_results_path = Path(clustering_results_path)
        self._run_clustering = run_clustering
        self._n_clusters_per_region = n_clusters_per_region or {r.code: 5 for r in Region}

        # Fix duplicate libomp crash on macOS (PyTorch + faiss-cpu both bundle OpenMP)
        if platform.system() == "Darwin":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["OMP_NUM_THREADS"] = "1"

        # Build the flat list of (region, cluster_id, cluster_df) to iterate
        self._groups: list[tuple[str, int, pd.DataFrame]] = []
        self._index = 0
        self._setup()

    # ------------------------------------------------------------------
    # Setup: optionally run K-Means, then build time series per cluster
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        con = ddb.connect(database=self._db_path)
        try:
            if self._run_clustering:
                assignments = self._run_kmeans_all_regions(con)
                self._save_assignments(assignments)
            else:
                assignments = self._load_assignments()

            for region_code, (client_ids, cluster_labels) in assignments.items():
                cluster_map = pd.DataFrame({"id": client_ids, "cluster": cluster_labels})
                con.register("cluster_map", cluster_map)
                n_clusters = int(cluster_labels.max()) + 1
                for cluster_id in range(n_clusters):
                    ts_df = self._get_cluster_timeseries(con, cluster_id)
                    print(f"{region_code} / cluster {cluster_id}: {len(ts_df)} timesteps")
                    self._groups.append((region_code, cluster_id, ts_df))
        finally:
            con.close()

    def _run_kmeans_all_regions(self, con: ddb.DuckDBPyConnection) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        import faiss
        assignments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for region in Region:
            n_clusters = self._n_clusters_per_region.get(region.code, 5)
            print(f"\nBuilding client vectors for {region.code}...")
            parquet_glob = str(Path(self._clustering_path) / "**" / "*.parquet")
            deps = tuple(region.departamentos)
            df = con.query(f"""
                SELECT id, mes, dia_semana, hora, valor
                FROM read_parquet('{parquet_glob}', hive_partitioning=true)
                WHERE departamento IN {deps}
            """).fetchdf()
            if df.empty:
                raise FileNotFoundError(f"No clustering vectors found for region {region.code}")
            df["feature_key"] = df["mes"].astype(str) + "_" + df["dia_semana"].astype(str) + "_" + df["hora"].astype(str)
            pivot = df.pivot_table(index="id", columns="feature_key", values="valor", aggfunc="mean").fillna(0.0)
            client_ids = pivot.index.values
            vectors = np.ascontiguousarray(pivot.values, dtype=np.float32)
            print(f"  {len(client_ids)} clients, {vectors.shape[1]} features. Running K-Means (k={n_clusters})...")
            kmeans = faiss.Kmeans(vectors.shape[1], n_clusters, niter=50, seed=42, verbose=True)
            kmeans.train(vectors)
            _, labels = kmeans.index.search(vectors, 1)
            assignments[region.code] = (client_ids, labels.flatten())
            unique, counts = np.unique(labels, return_counts=True)
            for c, cnt in zip(unique, counts):
                print(f"  Cluster {c}: {cnt} clients")
        return assignments

    def _save_assignments(self, assignments: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        data = {}
        for region_code, (client_ids, labels) in assignments.items():
            data[region_code] = {
                "n_clusters": int(labels.max()) + 1,
                "client_ids": [int(cid) if isinstance(cid, np.integer) else cid for cid in client_ids.tolist()],
                "assignments": labels.tolist(),
            }
        self._clustering_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._clustering_results_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Clustering assignments saved to {self._clustering_results_path}")

    def _load_assignments(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        with open(self._clustering_results_path) as f:
            raw = json.load(f)
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for region_code, info in raw.items():
            result[region_code] = (
                np.array(info["client_ids"]),
                np.array(info["assignments"]),
            )
        return result

    def _get_cluster_timeseries(self, con: ddb.DuckDBPyConnection, cluster_id: int) -> pd.DataFrame:
        query = f"""
        SELECT e.dia, e.hora,
               SUM(agg_valor) AS agg_valor,
               AVG((temp_max  + 15) / 65) AS temp_max,
               AVG((temp_min  + 15) / 65) AS temp_min,
               AVG((temp_media + 15) / 65) AS temp_media
        FROM (
            SELECT departamento, dia, hora, SUM(valor) AS agg_valor
            FROM read_parquet('{self._data_path}') i
            INNER JOIN cluster_map cm ON i.id = cm.id
            WHERE cm.cluster = {cluster_id}
            GROUP BY departamento, dia, hora
        ) e
        INNER JOIN temperatura_departamento t
            ON e.dia = t.dia AND e.departamento = t.departamento
        GROUP BY e.dia, e.hora
        ORDER BY e.dia, e.hora
        """
        return con.execute(query).fetchdf()

    # ------------------------------------------------------------------
    # BaseExperimentHandler interface
    # ------------------------------------------------------------------

    def has_next(self) -> bool:
        return self._index < len(self._groups)

    def next_experiment_group(self) -> ExperimentGroup:
        region_code, cluster_id, ts_df = self._groups[self._index]
        self._index += 1

        all_dataset, train_dataset, val_dataset, test_dataset = data_splitter(
            df=ts_df,
            exp_config=self._exp_config,
        )
        return ExperimentGroup(
            name=f"clustering/{region_code}/cluster_{cluster_id}",
            full_dataset=all_dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            raw_df=ts_df,
        )

    def use_exogenous(self) -> bool:
        return True


def experiment_factory(
    experiment_type: ExperimentType,
    db_path: str,
    data_path: str,
    exp_config: ExperimentConfiguration,
    clustering_path: str | None = None,
    clustering_results_path: str | None = None,
    run_clustering: bool = False,
    n_clusters_per_region: dict[str, int] | None = None,
) -> BaseExperimentHandler:
    if experiment_type == ExperimentType.REGIONS:
        return RegionsExperimentHandler(db_path, data_path, exp_config)
    elif experiment_type == ExperimentType.COUNTRY:
        return CountryExperimentHandler(db_path, data_path, exp_config)
    elif experiment_type == ExperimentType.REGION_CLUSTERING:
        if clustering_path is None or clustering_results_path is None:
            raise ValueError(
                "clustering_path and clustering_results_path are required for REGION_CLUSTERING"
            )
        return ClusteringExperimentHandler(
            db_path=db_path,
            data_path=data_path,
            exp_config=exp_config,
            clustering_path=clustering_path,
            clustering_results_path=clustering_results_path,
            run_clustering=run_clustering,
            n_clusters_per_region=n_clusters_per_region,
        )
    else:
        raise ValueError(f"Experiment type {experiment_type} not supported.")


