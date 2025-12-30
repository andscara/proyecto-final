from enum import Enum
from dataclasses import dataclass

class HorizonType(Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"

@dataclass
class Horizon:
    """
    Class representing forecasting horizon configuration.
    Attributes:
        type (HorizonType): Type of forecasting horizon ('hour', 'day', 'week').
        length (int): Length of the forecasting horizon in time units.
    """
    type : HorizonType
    length : int

    def __str__(self):
        return f"{self.type.name}_{self.length}"