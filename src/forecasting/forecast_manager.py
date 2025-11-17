from typing import TypeAlias
from collections import defaultdict
from logging import Logger
from forecasting.forecast_model import ForecastModel, ForecastModelID
import pandas as pd

from input import Input
from metrics import ForecastMetricType

ForecastFitResult: TypeAlias = dict[str, dict[ForecastModelID, dict[ForecastMetricType, float]]]

class ForecastManager:
    
    def __init__(self, logger: Logger):
        self._logger = logger
        self._forecast_models : list[ForecastModel] = []

    def fitAll(
        self, 
        input: Input, 
        cluster_names: list[str],
        average_data: pd.DataFrame
    ) -> ForecastFitResult:
        results: ForecastFitResult = defaultdict(lambda: defaultdict(lambda: dict[ForecastMetricType, float]()))
        for cluster_name in cluster_names:
            self._logger.info(f"Fitting models for cluster: {cluster_name}")
            cluster_data = average_data[average_data['cluster'] == cluster_name]
            for model in self._forecast_models:
                self._logger.info(f"Fitting model {model.model_id}")
                model_results = model.fit(cluster_data, input)
                results[cluster_name][model.model_id] = model_results
                self._logger.info(f"Model {model.model_id} fit completed")
        return results


