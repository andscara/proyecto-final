# Proyecto Final

Proyecto de forecasting de series temporales de consumo eléctrico utilizando modelos Autoformer, Linear y SARIMAX.

## Requisitos

- Python 3.x
- [uv](https://docs.astral.sh/uv/) como gestor de paquetes

## Configuración

Crear un archivo `.env` dentro de la carpeta `src/` con las siguientes variables:

```env
DATA_PATH=<path a la carpeta con los archivos parquet de las series temporales>
CLUSTERING_PATH=<path a la carpeta con las series de dimensionalidad reducida para clusterización>
DB_PATH=<path a la base de datos DuckDB con los datos de temperatura>
DEVICE_NAME=cuda  # o "cpu" si no se dispone de GPU
```

## Ejecución

El punto de entrada del proyecto es `src/simple-flow.py`. Se ejecuta con `uv run` desde la raíz del proyecto:

```bash
uv run src/simple-flow.py <config> <experiment_type>
```

- `<config>`: path a un archivo de configuración TOML dentro de `configs/`. Por ejemplo: `configs/autoformer_day_ahead_temp.toml`
- `<experiment_type>`: tipo de experimento, uno de: `country`, `regions`, `region_clustering`

### Ejemplo

```bash
uv run src/simple-flow.py configs/autoformer_day_ahead_temp.toml country
```

### Configuraciones disponibles

Los archivos TOML en `configs/` definen los distintos experimentos:

- `autoformer_day_ahead_temp.toml` / `autoformer_day_ahead_no_temp.toml`
- `autoformer_weekly_temp.toml` / `autoformer_weekly_no_temp.toml`
- `linear_day_ahead_temp_fixed.toml` / `linear_day_ahead_temp_learned.toml` / `linear_day_ahead_no_temp.toml`
- `linear_weekly_temp_fixed.toml` / `linear_weekly_temp_learned.toml` / `linear_weekly_no_temp.toml`
- `baseline_day_ahead.toml` / `baseline_weekly.toml`
- `sarimax_day_ahead.toml` / `sarimax_weekly.toml`

## Carpeta `duckbench/`

Contiene utilidades escritas en Go que se utilizaron para la limpieza y preparación de los datos previo al entrenamiento.
