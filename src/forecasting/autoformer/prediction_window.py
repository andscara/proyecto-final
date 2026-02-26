from dataclasses import dataclass
from datetime import datetime
from typing import List, Any
import numpy as np

@dataclass
class PredictionWindow:
    """Data class for storing window prediction results with dates."""
    history: tuple[List[datetime], np.ndarray[Any, np.float32]]
    predictions: tuple[List[datetime], np.ndarray[Any, np.float32]]
    real_values: tuple[List[datetime], np.ndarray[Any, np.float32]]

    def __post_init__(self):
        # Ensure that the dates and values have the same length for history, predictions, and real_values
        assert len(self.history[0]) == len(self.history[1]), "History dates and values must have the same length."
        assert len(self.predictions[0]) == len(self.predictions[1]), "Prediction dates and values must have the same length."
        assert len(self.real_values[0]) == len(self.real_values[1]), "Real value dates and values must have the same length."
        # Ensure the lenght of predictions and real_values are the same
        assert len(self.predictions[0]) == len(self.real_values[0]), "Prediction dates and real value dates must have the same length."

    def aggregate(
        self,
        other: PredictionWindow
    ):
        """Aggregate another PredictionWindow with this one."""
        # Ensure that dates in history, predictions, and real_values are the same
        assert self.history[0] == other.history[0], "History dates must be the same for aggregation."
        assert self.predictions[0] == other.predictions[0], "Prediction dates must be the same for aggregation."
        assert self.real_values[0] == other.real_values[0], "Real value dates must be the same for aggregation."

        # Aggregate history
        new_history_dates = self.history[0]
        new_history_values = np.sum([self.history[1], other.history[1]])

        # Aggregate predictions
        new_predictions_dates = self.predictions[0]
        new_predictions_values = np.sum([self.predictions[1], other.predictions[1]])

        # Aggregate real values
        new_real_values_dates = self.real_values[0]
        new_real_values_values = np.sum([self.real_values[1], other.real_values[1]])

        return PredictionWindow(
            history=(new_history_dates, new_history_values),
            predictions=(new_predictions_dates, new_predictions_values),
            real_values=(new_real_values_dates, new_real_values_values)
        )
    