from abc import ABC, abstractmethod
from dataclasses import asdict
import pandas as pd
from input import Input
from profiling.profiler import ProfilerID
from profiling.profiling_result import SegmentationResult
from results import TrainingResult
from pathlib import Path
import json

class BaseStorage(ABC):

    def __init__(self, input: Input) -> None:
        super().__init__()
        self._input = input


    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the storage."""
        ...

    @abstractmethod
    def save(self, key: str, data: bytes) -> None:
        """Save data to storage with the given key."""
        ...

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load data from storage using the given key."""
        ...
    
    @abstractmethod
    def save_training_result(self, training_result: TrainingResult) -> None:
        """Save the training result to storage."""
        ...

    @abstractmethod
    def load_training_result(self) -> TrainingResult:
        """Load the training result from storage."""
        ...

    @abstractmethod
    def initialize(self) -> None:
        """Initialize storage with the given input configuration."""
        ...

    @abstractmethod
    def save_segmentation_result(
        self, 
        segmentation_result: SegmentationResult
    ) -> None:
        """Save the segmentation result to storage."""
        ...

    @abstractmethod
    def get_segmentation_result(
        self,
        profiler_id: ProfilerID
    ) -> SegmentationResult | None:
        """Get the segmentation result for a given profiler ID."""
        ...

    
class FileSystemStorage(BaseStorage):
    
    def __init__(self, input: Input, base_path: str):
        super().__init__(input)
        self._root = Path(base_path) / str(input)
        self._initialized = False
    
    def clear(self) -> None:
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        # Delete all files inside the base path, recursively
        for item in self._root.glob("**/*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
    
    def save(self, key: str, data: bytes) -> None:
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        file_path = self._root / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        file_path = self._root / key
        with open(file_path, "rb") as f:
            return f.read()
        
    def save_training_result(self, training_result: TrainingResult) -> None:
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        json_str = json.dumps(asdict(training_result))
        self.save("training_result.json", json_str.encode("utf-8"))

    def load_training_result(self) -> TrainingResult:
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        data = self.load("training_result.json")
        json_str = data.decode("utf-8")
        dict_data = json.loads(json_str)
        return TrainingResult(**dict_data)
    
    def initialize(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def save_segmentation_result(
        self,
        segmentation_result: SegmentationResult
    ) -> None:
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        df_path = self._root / "segmentation" / str(segmentation_result.id) / "average_data.parquet"
        map_path = self._root / "segmentation" / str(segmentation_result.id) / "mapping.json"
        # If both files exist, skip saving
        if df_path.exists() and map_path.exists():
            return
        segmentation_result.average_data.to_parquet(df_path)
        with open(map_path, "w") as f:
            json.dump(segmentation_result.mapping, f)

    def get_segmentation_result(
        self,
        profiler_id: ProfilerID
    ) -> SegmentationResult | None:
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        df_path = self._root / "segmentation" / str(profiler_id) / "average_data.parquet"
        map_path = self._root / "segmentation" / str(profiler_id) / "mapping.json"
        if not df_path.exists() or not map_path.exists():
            return None
        average_data = pd.read_parquet(df_path) # type: ignore
        with open(map_path, "r") as f:
            mapping = json.load(f)
        return SegmentationResult(
            id=profiler_id,
            mapping=mapping,
            average_data=average_data
        )


        