from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import pandas as pd
from . import ProfilerID, SegmentationResult

T = TypeVar('T', bound=ProfilerID)

class Profiler[T](ABC):

    def __init__(self, id: T):
        super().__init__()
        self.id = id

    @abstractmethod
    def profile(self, data: pd.DataFrame) -> SegmentationResult:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    def _calculate_average_data(
            self, 
            data: pd.DataFrame, 
            mapping: dict[str, str]
    ) -> pd.DataFrame:
        """Calculate the average data for each cluster in the mapping."""
        new_data = data.copy()
        new_data["cluster"] = new_data["customer_id"].map(mapping)
        result = new_data.groupby(["cluster", "timestamp"]).agg({"kwh": "mean"}) # type: ignore
        del new_data
        result.reset_index(inplace=True)
        return result