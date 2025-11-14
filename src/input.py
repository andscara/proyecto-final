from dataclasses import dataclass
from data_source import DataSource
from horizon import Horizon


@dataclass
class FitInput:
    """
    Class representing input configuration to train the forecasting model.
    Attributes:
        data_source (DataSource): Data source instance to read data from.
        horizon (Horizon): Forecasting horizon configuration.
    """
    data_source : DataSource
    horizon : Horizon

@dataclass
class PredictionInput:
    """
    Class representing input configuration to make predictions with the forecasting model.
    Attributes:
        data_source (DataSource): Data source instance to read data from.
        horizon (Horizon): Forecasting horizon configuration.
    """
    data_source : DataSource
    horizon : Horizon