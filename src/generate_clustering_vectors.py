#type: ignore

"""
Pre-compute clustering feature vectors per departamento and save them as
individual parquet files so they don't need to be rebuilt every run.

Output layout (Hive-style partitioning):
    <DB_PATH parent>/clustering_vectors/departamento=<DEPARTAMENTO>/data.parquet

Each parquet contains columns: id, mes, dia_semana, hora, valor
(the raw query output, one row per client/time-slot combination).
Readable with: read_parquet('.../clustering_vectors/**/*.parquet', hive_partitioning=true)
"""

import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

import duckdb as ddb
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
PATH = os.getenv("DATA_PATH")

# All departamentos across every region
DEPARTAMENTOS = [
    "ARTIGAS", "SALTO", "RIVERA", "TACUAREMBO", "CERRO LARGO",
    "SAN JOSE", "COLONIA", "CANELONES", "FLORES", "FLORIDA", "SORIANO",
    "MALDONADO", "ROCHA", "TREINTA Y TRES", "LAVALLEJA",
    "PAYSANDU", "RIO NEGRO", "DURAZNO",
    "MONTEVIDEO",
]


def main():
    con = ddb.connect(database=os.getenv("DB_PATH"))
    con.execute("SET preserve_insertion_order = false;")

    output_dir = Path(os.getenv("DATA_PATH")).parent / "clustering_vectors"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dep in DEPARTAMENTOS:
        print(f"Processing {dep}...")
        dep_dir = output_dir / f"departamento={dep}"
        dep_dir.mkdir(parents=True, exist_ok=True)
        out_path = dep_dir / "data.parquet"

        query = f"""
            COPY (
                SELECT id, month(dia) AS mes, dayofweek(dia) AS dia_semana, hora, AVG(valor) AS valor
                FROM (
                    SELECT id, dia, hora,
                        COALESCE(valor / NULLIF(SUM(valor) OVER (PARTITION BY id, dia), 0), 0) as valor
                    FROM read_parquet('{PATH}')
                    WHERE departamento = '{dep}'
                )
                GROUP BY id, mes, dia_semana, hora
                ORDER BY id, mes, dia_semana, hora
            ) TO '{out_path}' (FORMAT PARQUET);
        """
        con.execute(query)
        print(f"  -> {out_path}")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
