from dataclasses import dataclass
from abc import ABC, abstractmethod
from profiling.segmentation_result import SegmentationResult
import pandas as pd

@dataclass
class ProfilerID:
    id: str

class Profiler(ABC):
    
    def __init__(self, id: ProfilerID):
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