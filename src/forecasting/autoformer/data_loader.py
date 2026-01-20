#type: ignore
import random
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from typing import Any
import numpy as np

from runner import train

class TrainDataset(data.Dataset):
    def __init__(
        self,
        data_windows: list[NDArray[Any]],
        time_features_windows: list[NDArray[Any]],
        seq_len: int,
        label_len: int,
        pred_len: int
    ):
        super().__init__()
        assert len(data_windows) == len(time_features_windows), "Data windows and time features windows must have the same length"
        self._data_windows = data_windows
        self._time_features_windows = time_features_windows
        self._seq_len = seq_len
        self._label_len = label_len
        self._pred_len = pred_len


    def __len__(self) -> int:
        return len(self._data_windows)
    
    def __getitem__(self, index: int):
        data_window = self._data_windows[index]
        time_window = self._time_features_windows[index]

        s_begin = 0
        s_end = self._seq_len
        r_begin = s_end - self._label_len
        r_end = r_begin + self._label_len + self._pred_len

        seq_x = data_window[s_begin:s_end]
        seq_x_mark = time_window[s_begin:s_end]

        seq_y = data_window[r_begin:r_begin + self._label_len]
        seq_y_mark = time_window[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

class TestDataset(data.Dataset):
    def __init__(
        self,
        data_windows: list[NDArray[np.float32]],
        time_features_windows: list[NDArray[np.float32]],
        seq_len: int,
        label_len: int,
        pred_len: int
    ):
        super().__init__()
        self.data_windows = data_windows
        self.time_windows = time_features_windows
        self._seq_len = seq_len
        self._label_len = label_len
        self._pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data_windows)

    def __getitem__(self, index: int):
        data = self.data_windows[index]
        time = self.time_windows[index]

        s_begin = 0
        s_end = self._seq_len

        r_begin = s_end - self._label_len
        r_end = r_begin + self._label_len + self._pred_len

        seq_x = data[s_begin:s_end]                     # (seq_len, 1)
        seq_y = data[r_begin:r_begin + self._label_len] # (label_len, 1)

        seq_x_mark = time[s_begin:s_end]                # (seq_len, T)
        seq_y_mark = time[r_begin:r_end]                # (label_len+pred_len, T)

        return seq_x, seq_y, seq_x_mark, seq_y_mark



def data_splitter(
    df: pd.DataFrame,
    train_val_ratio: float,
    test_ratio: float,
    windows_size: int,
    horizon: int,
    stride: int,
    target_col_name: str,
    scale: bool = True
) -> tuple[TrainDataset, TestDataset]:
    assert 0.0 < train_val_ratio < 1.0, "Train+Val ratio must be between 0 and 1"
    assert 0.0 < test_ratio < 1.0, "Test ratio must be between 0 and 1"
    assert train_val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"


    # Scale the dataset if needed
    data: NDArray[Any]
    only_data_df = df[[target_col_name]]
    if scale:
        scaler = StandardScaler()
        scaler.fit(only_data_df.values)
        data = scaler.transform(only_data_df.values)
    else:
        data = only_data_df.values

    # Add the time features
    # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1) -> already exists in the dataframe
    df["timestamp"] = pd.to_datetime(df["dia"]) + pd.to_timedelta(df["hora"] - 1, unit="h")
    df_stamp = df[['timestamp']]
    df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
    df_stamp['minute'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
    df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
    data_stamp = df_stamp.drop(columns=['timestamp'], axis=1).values

    # Calculate all  windows based on window_size, horizon and stride
    data_windows: list[NDArray[Any]] = []
    time_features_windows: list[NDArray[Any]] = []
    for start in range(0, len(df) - windows_size - horizon + 1, stride):
        end = start + windows_size + horizon
        data_windows.append(data[start:end])
        time_features_windows.append(data_stamp[start:end])

    # Split the windows randomly into train, val and test sets
    all_windows_count = len(data_windows)
    all_indexes = np.arange(all_windows_count)

    train_val_indexes = np.random.choice(
        all_indexes,
        size=int(all_windows_count * train_val_ratio),
        replace=False
    )
    train_val_data_windows = []
    train_val_time_features_windows = []
    for idx in train_val_indexes:
        train_val_data_windows.append(data_windows[idx])
        train_val_time_features_windows.append(time_features_windows[idx])

    test_indexes = np.setdiff1d(
        all_indexes,
        train_val_indexes,
        assume_unique=True
    )        
    test_data_windows = []
    test_time_features_windows = []
    for idx in test_indexes:
        test_data_windows.append(data_windows[idx])
        test_time_features_windows.append(time_features_windows[idx])

    #Create Datasets
    train_dataset = TrainDataset(
        data_windows=train_val_data_windows,
        time_features_windows=train_val_time_features_windows,
        seq_len=windows_size,
        label_len=windows_size//2,
        pred_len=horizon
    )

    test_dataset = TestDataset(
        data_windows=test_data_windows,
        time_features_windows=test_time_features_windows,
        seq_len=windows_size,
        label_len=windows_size//2,
        pred_len=horizon
    )
    return train_dataset, test_dataset

    
