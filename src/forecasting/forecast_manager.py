from typing import TypeAlias
from collections import defaultdict
from logging import Logger
from forecasting.forecast_model import ForecastModel, ForecastModelID
import pandas as pd

from input import Input
from metrics import ForecastMetricType

ForecastFitResult: TypeAlias = dict[ForecastModelID, dict[ForecastMetricType, float]]

class ForecastManager:
    
    def __init__(self, logger: Logger):
        self._logger = logger
        self._forecast_models : list[ForecastModel] = []

    def fit(
        self, 
        input: Input, 
        data: pd.DataFrame
    ) -> ForecastFitResult:
        results: ForecastFitResult = defaultdict(lambda: dict[ForecastMetricType, float]())
        for model in self._forecast_models:
            self._logger.info(f"Fitting model {model.model_id}")
            model_results = model.fit(data, input)
            results[model.model_id] = model_results
            self._logger.info(f"Model {model.model_id} fit completed")
        return results


