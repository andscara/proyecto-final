from abc import ABC, abstractmethod

from results import TrainingResult

class BaseStorage(ABC):

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

    