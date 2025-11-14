from dataclasses import dataclass
from abc import ABC, abstractmethod
from profiling.profiling_result import Profile
import pandas as pd

@dataclass
class ProfilerID:
    id: str

class Profiler(ABC):
    
    def __init__(self, id: ProfilerID):
        super().__init__()
        self.id = id

    @abstractmethod
    def profile(self, data: pd.DataFrame) -> Profile:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass