from dataclasses import dataclass



@dataclass
class ForecastPredictionResult:
    """
    Class representing the result of a forecasting prediction operation.
    Attributes:
        predictions (list[float]): List of predicted values.
    """
    predictions: list[float]
