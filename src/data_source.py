import pandas as pd
from abc import ABC, abstractmethod

class DataSource(ABC):
    """
    Abstract base class for data sources.
    Methods:
        read_data: Abstract method to read data from the source.
    """
    @abstractmethod
    def read_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def __str__(self):
        pass

class ParquetDataSource(DataSource):
    """
    Data source class for reading data from a Parquet file.
    Attributes:
        file_path (str): Path to the Parquet file.
    Methods:
        read_data: Reads data from the Parquet file and returns it as a DataFrame.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_data(self) -> pd.DataFrame:
        return pd.read_parquet(self.file_path)

    def __str__(self):
        return self.file_path