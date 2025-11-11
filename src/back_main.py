import pandas as pd
from sklearn.cluster import KMeans
from darts.models import AutoARIMA
from darts.timeseries import TimeSeries

def read_input(file_path: str) -> pd.DataFrame:
    raw_df = pd.read_parquet(file_path)
    start_date = raw_df['start_date'].max()
    end_date = raw_df['end_date'].min()
    if start_date.hour != 0:
        tmp = start_date + pd.Timedelta(days=1)
        start_date = pd.Timestamp(
            year = tmp.year,
            month = tmp.month,
            day = tmp.day,
            hour = 0,
        )

    if end_date.hour != 23:
        tmp = end_date - pd.Timedelta(days=1)
        end_date = pd.Timestamp(
            year = tmp.year,
            month = tmp.month,
            day = tmp.day,
            hour = 23,
        )
    print(f"{start_date=}")
    print(f"{end_date=}")
    hourly_df = raw_df[(raw_df["timestamp"] >= start_date) & (raw_df["timestamp"] <= end_date)]
    hourly_df.drop(columns=["start_date", "end_date", "imputed"], inplace=True)
    hourly_df.sort_values(by=['customer_id', 'timestamp'], ascending=True, inplace=True)
    return hourly_df

def calculate_clusters(df: pd.DataFrame) -> dict[str, str]:
    # Calculate the average hourly week for each customer and implement K-Means clustering
    df['day_name'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    hourly_average_df = df.groupby(['customer_id', 'day_name', 'hour']).mean()
    pivot_hourly_df = hourly_average_df.reset_index().pivot_table(
        index='customer_id',
        columns=['day_name', 'hour'],
        values=['kWh']
    )
    del hourly_average_df

    n_clusters = 10

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pivot_hourly_df)
    pivot_hourly_df['cluster'] = kmeans.labels_
    return pivot_hourly_df['cluster'].to_dict()


def calculate_cluster_averaging(df: pd.DataFrame, clusters: dict[str, str]) -> pd.DataFrame:
    df['cluster'] = df['customer_id'].map(lambda cid: clusters[cid])
    df['kWh'] = df.groupby(['cluster', 'timestamp'])['kWh'].transform('mean')
    df.drop(columns=['customer_id'], inplace=True)
    return df

def train(df: pd.DataFrame) -> dict[str, float]:
    errors = {}
    for cluster in df['cluster'].unique():
        model = AutoARIMA(
            season_length=24*7,        # daily seasonality (24 hours per day)
            max_p=4, 
            max_q=4,
            max_P=2, 
            max_Q=2,
            d=None,
            D=None
        )
        series = TimeSeries.from_dataframe(df[df['cluster'] == cluster], 'timestamp', 'kWh')
        train, val = series.split_after(0.90)
        model.fit(train)
        model.predict(len(val))
        error = (val - model.predict(len(val))).mae().values()[0]
        print(f"Cluster {cluster} - MAE: {error}")
        errors[cluster] = error
    return errors

def main():
    raw_df = read_input('~/Downloads/goiener_post_df.parquet')
    print('Data loaded.')
    clusters = calculate_clusters(raw_df)
    print('Clusters calculated.')
    cluster_averages = calculate_cluster_averaging(raw_df, clusters)
    print('Cluster averages calculated.')
    prediction_errors = train(cluster_averages)
    print("Clusters:", clusters)
    print("Prediction Errors:", prediction_errors)


if __name__ == "__main__":
    main()