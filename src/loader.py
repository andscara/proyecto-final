import pandas as pd
from pathlib import Path


# start pd.Timestamp('2019-02-28 01:00:00'), end pd.Timestamp('2020-03-01 00:00:00')
def load_customer_data(folder_path: str | Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all CSV files from a folder into a single pandas DataFrame.

    Each CSV file should have columns: timestamp, kWh, imputed
    The filename (without extension) will be used as the customer_id.

    Args:
        folder_path: Path to the folder containing CSV files

    Returns:
        pd.DataFrame: Combined DataFrame with columns: customer_id, timestamp, kWh, imputed

    Raises:
        FileNotFoundError: If the folder doesn't exist
        ValueError: If no CSV files are found in the folder
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Get all CSV files in the folder
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in: {folder_path}")

    # Load all CSV files efficiently
    dataframes = []
    metadatas = []
    for csv_file in csv_files:
        # Extract customer_id from filename (without extension)
        customer_id = csv_file.stem

        # Read CSV with proper data types
        df = pd.read_csv(
            csv_file,
            parse_dates=['timestamp'],
            dtype={
                'kWh': float,
                'imputed': bool
            }
        )
        metadata = analyze_data_holes(df)
        metadata["customer_id"] = customer_id

        # Add customer_id column
        df['customer_id'] = customer_id
        df['start_date'] = metadata['start_date']
        df['end_date'] = metadata['end_date']

        if metadata["start_date"] <= start_date and metadata["end_date"] == end_date:
            metadatas.append(metadata)
            dataframes.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Reorder columns to put customer_id first
    columns_order = ['customer_id', 'timestamp', 'kWh', 'imputed']
    combined_df = combined_df[columns_order]

    return combined_df, pd.DataFrame(metadatas)


def analyze_data_holes(df: pd.DataFrame) -> dict[str, any]:
    """
    Analyze missing timestamps (holes) in hourly time series data.

    Data is expected to be hourly from 01:00:00 to 00:00:00 (24 hours per day).
    A hole is defined as a missing timestamp in the expected hourly sequence.

    Args:
        df: DataFrame with a 'timestamp' column containing datetime values

    Returns:
        dict: Dictionary containing:
            - total_holes: Total number of missing timestamps
            - max_continuous_hole: Maximum number of consecutive missing timestamps
            - start_date: First timestamp in the data
            - end_date: Last timestamp in the data

    Raises:
        ValueError: If DataFrame is empty or doesn't have a 'timestamp' column
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")

    # Sort by timestamp to ensure correct order
    df_sorted = df.sort_values('timestamp').copy()

    # Get start and end dates
    start_date = df_sorted['timestamp'].iloc[0]
    end_date = df_sorted['timestamp'].iloc[-1]

    # Create complete hourly range from start to end
    expected_timestamps = pd.date_range(
        start=start_date,
        end=end_date,
        freq='h'
    )

    # Find missing timestamps
    actual_timestamps = set(df_sorted['timestamp'])
    expected_timestamps_set = set(expected_timestamps)
    missing_timestamps = sorted(expected_timestamps_set - actual_timestamps)

    total_holes = len(missing_timestamps)

    # Calculate maximum continuous hole
    max_continuous_hole = 0
    if missing_timestamps:
        current_continuous = 1
        max_continuous_hole = 1

        for i in range(1, len(missing_timestamps)):
            # Check if timestamps are consecutive (1 hour apart)
            time_diff = missing_timestamps[i] - missing_timestamps[i-1]
            if time_diff == pd.Timedelta(hours=1):
                current_continuous += 1
                max_continuous_hole = max(max_continuous_hole, current_continuous)
            else:
                current_continuous = 1

    return {
        'total_holes': total_holes,
        'max_continuous_hole': max_continuous_hole,
        'start_date': start_date,
        'end_date': end_date
    }


