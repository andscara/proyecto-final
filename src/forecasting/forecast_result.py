from dataclasses import dataclass

from forecasting.forecast_model import ForecastModelID
from metrics import ForecastMetricType


@dataclass
class ForecastFitResult:
    results: dict[ForecastMetricType, tuple[ForecastModelID, float]]

@dataclass
class ForecastPredictionResult:
    """
    Class representing the result of a forecasting prediction operation.
    Attributes:
        predictions (list[float]): List of predicted values.
    """
    predictions: list[float]
