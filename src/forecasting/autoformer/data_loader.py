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


def get_holidays() -> list[pd.Timestamp]:
    holidays: list[pd.Timestamp] = []
    #Fixed holidays
    for year in range(2020, 2025):
        holidays.append(pd.Timestamp(f"{year}-01-01"))
        holidays.append(pd.Timestamp(f"{year}-05-01"))
        holidays.append(pd.Timestamp(f"{year}-07-18"))
        holidays.append(pd.Timestamp(f"{year}-08-25"))
        holidays.append(pd.Timestamp(f"{year}-12-25"))
    #Special holidays per year
    """--- 2022 ---
    2022 2 28 lunes-carnaval   3
    2022 3 1 martes-carnaval   3
    2022 4 11 lunes-turismo   3
    2022 4 12 martes-turismo   3
    2022 4 13 miercoles-turismo   3
    2022 4 14 jueves-turismo   3
    2022 4 15 viernes-turismo   3
    """
    holidays.append(pd.Timestamp("2022-02-28"))
    holidays.append(pd.Timestamp("2022-03-01"))
    holidays.append(pd.Timestamp("2022-04-11"))
    holidays.append(pd.Timestamp("2022-04-12"))
    holidays.append(pd.Timestamp("2022-04-13"))
    holidays.append(pd.Timestamp("2022-04-14"))
    holidays.append(pd.Timestamp("2022-04-15"))
    """--- 2023 ---
    2023 2 20 lunes-carnaval   3
    2023 2 21 martes-carnaval   3
    2023 4 3 lunes-turismo   3
    2023 4 4 martes-turismo   3
    2023 4 5 miercoles-turismo   3
    2023 4 6 jueves-turismo   3
    2023 4 7 viernes-turismo   3
    """
    holidays.append(pd.Timestamp("2023-02-20"))
    holidays.append(pd.Timestamp("2023-02-21"))
    holidays.append(pd.Timestamp("2023-04-03"))
    holidays.append(pd.Timestamp("2023-04-04"))
    holidays.append(pd.Timestamp("2023-04-05"))
    holidays.append(pd.Timestamp("2023-04-06"))
    holidays.append(pd.Timestamp("2023-04-07"))
    """--- 2024 ---
    2024 2 12 lunes-carnaval   3
    2024 2 13 martes-carnaval   3
    2024 3 25 lunes-turismo   3
    2024 3 26 martes-turismo   3
    2024 3 27 miercoles-turismo   3
    2024 3 28 jueves-turismo   3
    2024 3 29 viernes-turismo   3
    """
    holidays.append(pd.Timestamp("2024-02-12"))
    holidays.append(pd.Timestamp("2024-02-13"))
    holidays.append(pd.Timestamp("2024-03-25"))
    holidays.append(pd.Timestamp("2024-03-26"))
    holidays.append(pd.Timestamp("2024-03-27"))
    holidays.append(pd.Timestamp("2024-03-28"))
    holidays.append(pd.Timestamp("2024-03-29"))
    """--- 2025 ---
    2025 3 3 lunes-carnaval   3
    2025 3 4 martes-carnaval   3
    2025 4 14 lunes-turismo   3
    2025 4 15 martes-turismo   3
    2025 4 16 miercoles-turismo   3
    2025 4 17 jueves-turismo   3
    2025 4 18 viernes-turismo   3
    """
    holidays.append(pd.Timestamp("2025-03-03"))
    holidays.append(pd.Timestamp("2025-03-04"))
    holidays.append(pd.Timestamp("2025-04-14"))
    holidays.append(pd.Timestamp("2025-04-15"))
    holidays.append(pd.Timestamp("2025-04-16"))
    holidays.append(pd.Timestamp("2025-04-17"))
    holidays.append(pd.Timestamp("2025-04-18"))

    return holidays

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

    # Only the target column goes into the data stream
    only_data_df = df[[target_col_name]]
    if scale:
        target_data = only_data_df.values
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(target_data)
        data = scaler.transform(target_data)
    else:
        data = only_data_df.values

    data = data.astype(np.float32)
    # Add the time features
    df.loc[:, "timestamp"] = pd.to_datetime(df["dia"]) + pd.to_timedelta(df["hora"] - 1, unit="h")
    df_stamp = df[['timestamp']].copy()
    df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month).astype('float32')
    df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day).astype('float32')
    df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday()).astype('float32')
    df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour).astype('float32')
    # Add holidays as a binary feature
    holidays = get_holidays()
    df_stamp['is_holiday'] = df_stamp['timestamp'].isin(holidays).astype(np.float32)
    data_stamp = df_stamp.drop(columns=['timestamp'], axis=1).values


    # Append exogenous columns as additional time features
    if exog_cols is not None:
        exog_data = df[exog_cols].values.astype(np.float32)
        data_stamp = np.concatenate([data_stamp, exog_data], axis=1)

    # Calculate all windows based on window_size, horizon and stride
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
    label_len: int,
    stride: int,
    target_col_name: str,
    scale: bool = True,
    windows_to_test: int = 20,
    exog_cols: list[str] | None = None
) -> tuple[WindowsDataset, WindowsDataset, WindowsDataset, WindowsDataset]:
    
    seq_len = windows_size
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
        exog_cols=exog_cols,
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

    
