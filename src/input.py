class Input:
    """
    Class representing input configuration for a forecasting model.
    Attributes:
        input_path (str): Path to the input data file.
        horizon_type (str): Type of forecasting horizon ('hour', 'day', 'week').
        horizon_length (int): Length of the forecasting horizon in time units.
    """
    def __init__(
            self,
            input_path: str,
            horizon_type: str,
            horizon_length: int
    ):
        self.input_path = input_path
        self.horizon_type = horizon_type
        self.horizon_length = horizon_length

    def __repr__(self):
        return (
            f"Input(input_path={self.input_path}, "
            f"horizon_type={self.horizon_type}, "
            f"horizon_length={self.horizon_length})"
        )
