from abc import ABC, abstractmethod
from logging import Logger
from forecasting.forecast_result import ForecastPredictionResult
from input import PredictionInput, FitInput
from metrics import ForecastMetricType
from storage.storage import BaseStorage
import pandas as pd
from dataclasses import dataclass

@dataclass
class ForecastModelID:
    id: str

class ForecastModel(ABC):

    def __init__(self, logger: Logger, model_id: ForecastModelID):
        self.logger = logger
        self.model_id = model_id

    @abstractmethod
    def fit(
        self, 
        data: pd.DataFrame, 
        input: FitInput
    ) -> dict[ForecastMetricType, float]:
        """Fit the forecasting model to the provided data."""
        ...

    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame, 
        input: PredictionInput
    ) -> ForecastPredictionResult:
        """Predict future values for a given number of steps."""
        ...

    @abstractmethod
    def save_model(self, storage: BaseStorage) -> None:
        """Save the model to the provided storage."""
        ...

    @abstractmethod
    def load_model(self, storage: BaseStorage) -> None:
        """Load the model from the provided storage."""
        ...