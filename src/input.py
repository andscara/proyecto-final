from dataclasses import dataclass
from data_source import DataSource
from horizon import Horizon


@dataclass
class Input:
    """
    Class representing input configuration to train the forecasting model.
    Attributes:
        data_source (DataSource): Data source instance to read data from.
        horizon (Horizon): Forecasting horizon configuration.
    """
    data_source : DataSource
    horizon : Horizon

    def __str__(self):
        return f"{self.data_source}_{self.horizon}"