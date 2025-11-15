from dataclasses import dataclass

from forecasting.forecast_model import ForecastModelID

@dataclass
class GlobalModel:
    models: list[ForecastModelID]
    