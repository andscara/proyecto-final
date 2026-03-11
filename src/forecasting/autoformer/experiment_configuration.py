import horizon as h
from dataclasses import dataclass

@dataclass
class ExperimentConfiguration:
    windows_size: int
    horizon: h.Horizon
    label_len: int
    stride: int
    target_col_name: str
    scale: bool
    exog_cols: list[str] | None