from logging import Logger
from forecasting.forecast_result import ForecastPredictionResult
from input import PredictionInput, Input
from horizon import Horizon, HorizonType
from metrics import ForecastMetricType
from storage.storage import BaseStorage
import pandas as pd
from dataclasses import dataclass
from forecast_model import ForecastModel, ForecastModelID
from statsforecast.models import MSTL, AutoARIMA
from statsforecast import StatsForecast
import pickle
import numpy as np

class ARIMAModel(ForecastModel):

    def __init__(self, logger: Logger, model_id: ForecastModelID):
        self.logger = logger
        self.model_id = model_id

    def fit(
        self,
        data: pd.DataFrame,
        input: Input
    ) -> dict[ForecastMetricType, float]:
        mstl = MSTL(
            season_length=[24, 24 * 7],
            trend_forecaster=AutoARIMA()
        )
        freq = 'h'
        if input.horizon.type == HorizonType.DAY:
            freq ='D'
        elif input.horizon.type == HorizonType.WEEK:
            freq = 'W'
        sf = StatsForecast(
            models=[mstl],
            freq=freq,
        )

        horizon_length = input.horizon.length
        cv_results = sf.cross_validation(
            df=data,
            h=horizon_length,
            step_size=horizon_length,
            n_windows=10,
            id_col='cluster',
            time_col='timestamp',
            target_col='kWh'
        )

        y_true = cv_results['kWh'].values
        y_pred = cv_results['MSTL'].values

        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)

        self.logger.info(f"ARIMA model cross-validation - MAE: {mae:.4f}, MSE: {mse:.4f}")

        self.model = sf.fit(data, data, id_col='cluster', time_col="timestamp", target_col="kWh")
        self.logger.info("ARIMA model fitted successfully")

        return {
            ForecastMetricType.MAE: float(mae),
            ForecastMetricType.MSE: float(mse)
        }

    def predict(
        self,
        data: pd.DataFrame,
        input: Input
    ) -> ForecastPredictionResult:
        horizon_length = input.horizon.length
        forecasts = self.model.predict(h=horizon_length)
        predictions = forecasts['MSTL'].tolist()
        return ForecastPredictionResult(predictions=predictions)

    def save_model(self, storage: BaseStorage) -> None:
        model_key = f"models/{self.model_id.id}/arima_model.pkl"
        model_bytes = pickle.dumps(self.model)
        storage.save(model_key, model_bytes)
        self.logger.info(f"ARIMA model saved to storage with key: {model_key}")

    def load_model(self, storage: BaseStorage) -> None:
        model_key = f"models/{self.model_id.id}/arima_model.pkl"
        model_bytes = storage.load(model_key)
        self.model = pickle.loads(model_bytes)
        self.logger.info(f"ARIMA model loaded from storage with key: {model_key}")
