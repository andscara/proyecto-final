from abc import ABC, abstractmethod

class Profiler(ABC):
    @abstractmethod
    def profile(self, data) -> dict[str, str]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass