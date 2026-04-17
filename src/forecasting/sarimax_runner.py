#type: ignore
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import List

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA

from forecasting.prediction_window import PredictionWindow
from forecasting.trainer import Trainer


class SARIMAXRunner:
    """
    Rolling-window SARIMAX evaluator that produces the same List[PredictionWindow]
    output as Trainer.predict_windows(), making metrics directly comparable.

    For each test window:
      - fits MSTL(AutoARIMA) on all data up to the window start
      - predicts the next pred_len hours supplying future exogenous temperatures
      - wraps the result in a PredictionWindow

    Temperature columns are treated as exogenous regressors (oracle: real future
    values are used, which is the standard evaluation approach).
    """

    TRAIN_CUTOFF = pd.Timestamp("2024-09-01")
    # Season lengths: 1 day (24h) and 1 week (168h)
    SEASON_LENGTHS = [24, 24 * 7]

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        stride: int,
        target_col: str = "agg_valor",
        exog_cols: list[str] | None = None,
    ):
        """
        Args:
            df:         Full time series DataFrame with columns (dia, hora, <target>, [exog...])
            seq_len:    History window length shown in PredictionWindow (hours)
            pred_len:   Forecast horizon length (hours)
            stride:     How many hours to advance between windows
            target_col: Name of the target column
            exog_cols:  Exogenous regressor column names (e.g. ['temp_media'])
        """
        self._df = df.copy().reset_index(drop=True)
        self._seq_len = seq_len
        self._pred_len = pred_len
        self._stride = stride
        self._target_col = target_col
        self._exog_cols = exog_cols or []

        # Build a flat datetime index + values array for easy slicing
        self._df["timestamp"] = pd.to_datetime(self._df["dia"]) + pd.to_timedelta(
            self._df["hora"] - 1, unit="h"
        )
        self._df = self._df.sort_values("timestamp").reset_index(drop=True)

        # Fit scaler on training portion only (mirrors data_splitter behaviour)
        train_mask = self._df["timestamp"] < self.TRAIN_CUTOFF
        self._scaler = StandardScaler()
        self._scaler.fit(self._df.loc[train_mask, [target_col]])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_windows(self) -> List[PredictionWindow]:
        """
        Iterate over test windows and return one PredictionWindow per window.
        Windows start at TRAIN_CUTOFF and advance by stride hours.
        """
        test_start_idx = self._df.index[self._df["timestamp"] >= self.TRAIN_CUTOFF][0]
        windows: List[PredictionWindow] = []

        idx = test_start_idx
        while idx + self._pred_len <= len(self._df):
            train_df = self._df.iloc[:idx]
            future_df = self._df.iloc[idx : idx + self._pred_len]

            if len(future_df) < self._pred_len:
                break

            t0 = time.perf_counter()
            pred_values = self._fit_and_predict(train_df, future_df)
            elapsed = time.perf_counter() - t0
            window_ts = future_df["timestamp"].iloc[0]
            print(f"[SARIMAX] window {len(windows) + 1} @ {window_ts} — fit+predict took {elapsed:.1f}s")
            real_values = future_df[self._target_col].values.astype(np.float32)

            # History slice (seq_len hours before window start)
            hist_start = max(0, idx - self._seq_len)
            hist_values = self._df.iloc[hist_start:idx][self._target_col].values.astype(np.float32)
            hist_timestamps = self._df.iloc[hist_start:idx]["timestamp"].tolist()

            pred_timestamps = future_df["timestamp"].tolist()

            windows.append(
                PredictionWindow(
                    history=(hist_timestamps, hist_values),
                    predictions=(pred_timestamps, pred_values),
                    real_values=(pred_timestamps, real_values),
                )
            )

            idx += self._stride

        return windows

    def predict_series(
        self,
        pdf: PdfPages,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Concatenate all rolling-window predictions into a flat series and plot,
        mirroring Trainer.predict_series() output format.
        """
        windows = self.predict_windows()
        y_preds_flat = np.concatenate([pw.predictions[1] for pw in windows])
        y_reals_flat = np.concatenate([pw.real_values[1] for pw in windows])

        Trainer.plot_and_print_ys(
            pdf=pdf,
            y_preds_flat=y_preds_flat,
            y_reals_flat=y_reals_flat,
            rolling_step=self._stride,
        )
        return y_preds_flat, y_reals_flat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_and_predict(
        self,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> np.ndarray:
        """Fit MSTL(AutoARIMA) on train_df and predict pred_len steps."""
        # In statsforecast v2, exogenous columns are included directly in the
        # training DataFrame passed to fit(); future exogenous go in X_df on predict().
        sf_train = pd.DataFrame(
            {
                "unique_id": "series",
                "ds": train_df["timestamp"],
                "y": train_df[self._target_col].values.astype(np.float64),
            }
        )

        exog_future: pd.DataFrame | None = None
        if self._exog_cols:
            for col in self._exog_cols:
                sf_train[col] = train_df[col].values.astype(np.float64)
            exog_future = pd.DataFrame(
                {"unique_id": "series", "ds": future_df["timestamp"]}
                | {col: future_df[col].values.astype(np.float64) for col in self._exog_cols}
            )

        model = StatsForecast(
            models=[MSTL(season_length=self.SEASON_LENGTHS, trend_forecaster=AutoARIMA())],
            freq="h",
            n_jobs=4,
        )
        model.fit(sf_train)
        forecast = model.predict(h=self._pred_len, X_df=exog_future)

        return forecast["MSTL"].values.astype(np.float32)
