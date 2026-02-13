#type: ignore
import random
from matplotlib.pyplot import sca
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from typing import Any
import numpy as np

from runner import train

class WindowsDataset(data.Dataset):
    def __init__(
        self,
        data_windows: list[NDArray[Any]],
        time_features_windows: list[NDArray[Any]],
        seq_len: int,
        label_len: int,
        pred_len: int,
        scaler: StandardScaler | None
    ):
        super().__init__()
        assert len(data_windows) == len(time_features_windows), "Data windows and time features windows must have the same length"
        self._data_windows = data_windows
        self._time_features_windows = time_features_windows
        self._seq_len = seq_len
        self._label_len = label_len
        self._pred_len = pred_len
        self.scaler = scaler


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

        seq_y = data_window[r_begin:r_end]
        seq_y_mark = time_window[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

def create_windows(
    df: pd.DataFrame,
    windows_size: int,
    horizon: int,
    stride: int,
    target_col_name: str,
    scale: bool,
    exog_cols: list[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[list[NDArray[Any]], list[NDArray[Any]], StandardScaler | None]:
    # Scale the dataset if needed
    data: NDArray[Any]

    # Determine which columns to include
    if exog_cols is None:
        data_cols = [target_col_name]
    else:
        data_cols = [target_col_name] + exog_cols

    only_data_df = df[data_cols]
    if scale:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(only_data_df.values)
        data = scaler.transform(only_data_df.values)
    else:
        data = only_data_df.values
    
    data = data.astype(np.float32)
    # Add the time features
    # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1) -> already exists in the dataframe
    df.loc[:, "timestamp"] = pd.to_datetime(df["dia"]) + pd.to_timedelta(df["hora"] - 1, unit="h")
    df_stamp = df[['timestamp']].copy()
    df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month).astype('float32')
    df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day).astype('float32')
    df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday()).astype('float32')
    df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour).astype('float32')
    # df_stamp['minute'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
    # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
    data_stamp = df_stamp.drop(columns=['timestamp'], axis=1).values
    # Calculate all  windows based on window_size, horizon and stride
    data_windows: list[NDArray[Any]] = []
    time_features_windows: list[NDArray[Any]] = []
    for start in range(0, len(df) - windows_size - horizon + 1, stride):
        end = start + windows_size + horizon
        data_windows.append(data[start:end])
        time_features_windows.append(data_stamp[start:end])
    return data_windows, time_features_windows, scaler

def data_splitter(
    df: pd.DataFrame,
    windows_size: int,
    horizon: int,
    stride: int,
    target_col_name: str,
    scale: bool = True,
    windows_to_test: int = 20,
    exog_cols: list[str] | None = None
) -> tuple[WindowsDataset, WindowsDataset, WindowsDataset, WindowsDataset]:
    
    seq_len = windows_size
    label_len = windows_size // 2
    pred_len = horizon

    # Divide the dataframe [start_train, end_train], [end_df_minus_1_year, end_df]
    train_df = df[df['dia'] <  pd.Timestamp('2024-09-01')]
    val_test_df = df[df['dia'] >= pd.Timestamp('2024-09-01')]

    train_data_windows, train_time_features_windows, train_scaler = create_windows(
        df=train_df,
        windows_size=windows_size,
        horizon=horizon,
        stride=stride,
        target_col_name=target_col_name,
        scale=scale,
        exog_cols=exog_cols
    )

    val_test_data_windows, val_test_time_features_windows, val_test_scaler = create_windows(
        df=val_test_df,
        windows_size=windows_size,
        horizon=horizon,
        stride=stride,
        target_col_name=target_col_name,
        scale=scale,
        exog_cols=exog_cols,
        scaler=train_scaler,
    )

    all_data_windows, all_time_features_windows, all_scaler = create_windows(
        df=df,
        windows_size=windows_size,
        horizon=horizon,
        stride=stride,
        target_col_name=target_col_name,
        scale=scale,
        scaler=train_scaler,
    )

    all_dataset = WindowsDataset(
        data_windows=all_data_windows,
        time_features_windows=all_time_features_windows,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        scaler=all_scaler
    )

    #Create Datasets
    train_dataset = WindowsDataset(
        data_windows=train_data_windows,
        time_features_windows=train_time_features_windows,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        scaler=train_scaler
    )
    
    # Get the windows only for testing
    all_val_test_windows_count = len(val_test_data_windows)
    
    assert windows_to_test < all_val_test_windows_count, "The number of windows to test must be less than the total number of windows available for validation and testing."

    rng = np.random.default_rng(seed=42)
    all_indexes = np.arange(all_val_test_windows_count)

    val_indexes = rng.choice(
        all_indexes,
        size=int(all_val_test_windows_count - windows_to_test),
        replace=False
    )
    val_data_windows = []
    val_data_time_features_windows = []
    for idx in val_indexes:
        val_data_windows.append(val_test_data_windows[idx])
        val_data_time_features_windows.append(val_test_time_features_windows[idx])

    val_dataset = WindowsDataset(
        data_windows=val_data_windows,
        time_features_windows=val_data_time_features_windows,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        scaler=val_test_scaler
    )

    test_indexes = np.setdiff1d(
        all_indexes,
        val_indexes,
        assume_unique=True
    )        
    test_data_windows = []
    test_time_features_windows = []
    for idx in test_indexes:
        test_data_windows.append(val_test_data_windows[idx])
        test_time_features_windows.append(val_test_time_features_windows[idx])


    test_dataset = WindowsDataset(
        data_windows=test_data_windows,
        time_features_windows=test_time_features_windows,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        scaler=val_test_scaler
    )
    return all_dataset, train_dataset, val_dataset, test_dataset

    
