import logging
from data_source import ParquetDataSource
from horizon import Horizon, HorizonType
from input import Input
from storage.storage import FileSystemStorage
from profiling import ProfilingManager, SomProfiler
from forecasting.forecast_manager import ForecastManager
from forecasting.arima_model import ARIMAModel, AutoARIMAModelId
from forecasting.forecast_model import ForecastModelID
import runner


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    input_config = Input(
        data_source=ParquetDataSource(file_path="goiener_post_df.parquet"),
        horizon=Horizon(type=HorizonType.HOUR, length=24)
    )

    storage = FileSystemStorage(input=input_config, base_path="storage")

    profiling_manager = ProfilingManager(logger=logger, storage=storage)
    som_profiler = SomProfiler(
        rows=3,
        cols=3,
        learning_rate=0.5,
        sigma=None,
        epochs=10,
        batch_size=32,
        device="cpu"
    )
    profiling_manager._profilers.append(som_profiler)

    forecast_manager = ForecastManager(logger=logger)
    arima_model_id = AutoARIMAModelId(
        id="arima_mstl_24_168",
        season_length=[24, 24 * 7]
    )
    arima_model = ARIMAModel(logger=logger, model_id=arima_model_id)
    forecast_manager._forecast_models.append(arima_model)

    logger.info("Starting training pipeline with hardcoded parameters")
    runner.train(
        input=input_config,
        logger=logger,
        storage=storage,
        profiling_manager=profiling_manager,
        forecast_manager=forecast_manager
    )
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()