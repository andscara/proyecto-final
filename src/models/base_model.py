from abc import ABC, abstractmethod

from metrics import Metrics


class BaseModel(ABC):

    @abstractmethod
    def transform(self, data_path: str) -> str:
        """
        Transforms the input data according to the model's specifications.
        
        Args:
            data_path (str): The path to the input data file.
        
        Returns:
            str: The path to the transformed data file.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def train(self, data_path: str) -> None:
        """
        Trains the model using the provided data.
        
        Args:
            data_path (str): The path to the training data file.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def predict(self, data_path: str) -> Metrics:
        """
        Makes predictions using the trained model on the provided data.
        
        Args:
            data_path (str): The path to the data file for predictions.

        Returns:
            Metrics: The evaluation metrics for the predictions.
        """
        raise NotImplementedError("Subclasses must implement this method")