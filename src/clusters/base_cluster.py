from abc import ABC, abstractmethod


class BaseCluster(ABC):
    
    @abstractmethod
    def create_clusters(self, data_path: str) -> dict[str, str]:
        """
        Creates clusters from the input data.
        
        Args:
            data_path (str): The path to the input data file.
        
        Returns:
            dict[str, str]: A dictionary mapping meter id with cluster id.
        """
        raise NotImplementedError("Subclasses must implement this method")