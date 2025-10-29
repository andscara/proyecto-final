from input import Input


class Normalizer:

    def __init__(self, input: Input):
        self.input = input

    def normalize(self) -> tuple[str, str]:
        """
        Normalizes the input configuration for a forecasting model.
        Args:
            input (Input): The Input object to normalize.
        Returns:
            Pair[str, str]: (unnormalized_path, normalized_path)
        """
        raise NotImplementedError("Normalization logic is not yet implemented.")
    
    def denormalize(self, data_path: str) -> str:
        """
        Denormalizes the data back to its original configuration.
        Args:
            data_path (str): The path to the normalized data file.
        Returns:
            str: The path to the denormalized data file.
        """
        raise NotImplementedError("Denormalization logic is not yet implemented.")
    
