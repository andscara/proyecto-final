from abc import ABC, abstractmethod
from logging import Logger
import random
from forecasting.forecast_result import ForecastPredictionResult
from input import Input
from metrics import ForecastMetricType
from storage.storage import BaseStorage
import pandas as pd
from dataclasses import dataclass

@dataclass
class ForecastModelID:
    id: str

class ForecastModel(ABC):

    def __init__(
            self, 
            logger: Logger, 
            model_id: ForecastModelID,
            seed: int = 42
        ):
        self.logger = logger
        self.model_id = model_id
        self.seed = seed

    @abstractmethod
    def fit(
        self, 
        data: pd.DataFrame, 
        input: Input
    ) -> dict[ForecastMetricType, float]:
        """Fit the forecasting model to the provided data."""
        ...

    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame, 
        metric: ForecastMetricType,
        input: Input
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

    def get_train_val_dfs(
        self, 
        data: pd.DataFrame, 
        input: Input,
        validation_last_possible_window: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        We want to make sure the cross validation splits are consistent across models, so we will
        be moving the validation window splitting logic here, with a few considerations:
        1. We will assume the data rows represent the same logic type unit as the input.horizon type. For example, hourly data for hourly horizon.
        2. The validation window should not overlap with the training data.
        3. The validation window should be of length input.forecast_horizon.
        4. The validation window will only start from the second half of the data to ensure enough
           training data is available.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: train and validation dataframes.
        """
        # Add checks
        total_length = len(data)
        forecast_horizon = input.horizon.length
        if total_length < 2 * forecast_horizon:
                raise ValueError("Not enough data to create train and validation splits based on the forecast horizon.")   

        # Get a random split point for the validation window in the second half of the data
        random.seed(self.seed)
        if validation_last_possible_window:
            split_start = total_length - forecast_horizon
            split_end = total_length
        else:
            split_start = random.randint(total_length // 2, total_length - forecast_horizon)
            split_end = split_start + forecast_horizon
        val_df = data.iloc[split_start:split_end]
        train_df = data.iloc[:split_start]
        return train_df, val_df
        