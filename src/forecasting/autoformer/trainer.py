import logging
import time
from typing import List
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from forecasting.autoformer.data_loader import WindowsDataset
from forecasting.autoformer.tools import EarlyStopping, StandardScaler, adjust_learning_rate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy.typing as npt
from datetime import datetime, timedelta
import torch.nn.functional as F
from pathlib import Path
from prediction_window import PredictionWindow

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        window_stride_in_days: int,
        all_dataset: WindowsDataset,
        train_loader: data.DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        val_loader: data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        test_loader: data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
        label_len: int,
        pred_len: int,
        output_attention: bool,
        device_name: str = 'cpu',
        seed: int = 42
    ):
        self.model = model
        self.window_stride_in_days = window_stride_in_days
        self.all_dataset = all_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.rng = np.random.default_rng(seed=seed)

    def _predict(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
        rolling: bool = False,
        rolling_step: int = 0,
    ):
        if rolling and rolling_step > 0:
            return self._predict_rolling(batch_x, batch_y, batch_x_mark, batch_y_mark, rolling_step)

        # decoder input
        # Zero out future predictions (data stream only contains the target variable)
        dec_inp_future = torch.zeros_like(batch_y[:, -self.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp_future], dim=1).to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.output_attention:
                outputs = outputs[0]
            return outputs

        outputs = _run_model()

        outputs = outputs[:, -self.pred_len:, :]
        batch_y = batch_y[:, -self.pred_len:, :].to(self.device)

        return outputs, batch_y

    def _predict_rolling(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
        rolling_step: int,
    ):
        """
        Rolling multi-step prediction. Generates `rolling_step` time steps per roll,
        then shifts the encoder window forward by `rolling_step` and repeats until
        the full `pred_len` is covered.

        The model always runs with its original pred_len — we just keep only the
        first `rolling_step` values from each prediction and discard the rest.
        """
        total_pred_len = self.pred_len
        all_outputs = []
        steps_generated = 0

        # Current encoder input (will shift forward at each roll)
        cur_x = batch_x  # (B, seq_len, 1)
        cur_x_mark = batch_x_mark  # (B, seq_len, d_mark)

        while steps_generated < total_pred_len:
            this_step = min(rolling_step, total_pred_len - steps_generated)

            # Build decoder input: last label_len from cur_x + zeros for full pred_len
            dec_label = cur_x[:, -self.label_len:, :]  # (B, label_len, 1)
            dec_future = torch.zeros(cur_x.shape[0], self.pred_len, cur_x.shape[2], device=self.device)
            dec_inp = torch.cat([dec_label, dec_future], dim=1).to(self.device)  # (B, label_len + pred_len, 1)

            # Build decoder time marks: last label_len + full pred_len of marks
            dec_mark_label = cur_x_mark[:, -self.label_len:, :]
            # Take pred_len future marks starting from the current offset
            mark_start = self.label_len + steps_generated
            mark_end = mark_start + self.pred_len
            # Clamp to available marks, pad with last mark if needed
            available_end = min(mark_end, batch_y_mark.shape[1])
            dec_mark_future = batch_y_mark[:, mark_start:available_end, :]
            if dec_mark_future.shape[1] < self.pred_len:
                pad_len = self.pred_len - dec_mark_future.shape[1]
                pad = dec_mark_future[:, -1:, :].repeat(1, pad_len, 1)
                dec_mark_future = torch.cat([dec_mark_future, pad], dim=1)
            dec_y_mark = torch.cat([dec_mark_label, dec_mark_future], dim=1).to(self.device)

            # Run model with its original pred_len
            outputs = self.model(cur_x, cur_x_mark, dec_inp, dec_y_mark)
            if self.output_attention:
                outputs = outputs[0]

            # Keep only the first `this_step` values, discard the rest
            step_pred = outputs[:, :this_step, :]  # (B, this_step, 1)
            all_outputs.append(step_pred)
            steps_generated += this_step

            if steps_generated < total_pred_len:
                # Shift encoder window: drop first `this_step` timesteps, append predictions
                cur_x = torch.cat([cur_x[:, this_step:, :], step_pred], dim=1)
                cur_x_mark = torch.cat([
                    cur_x_mark[:, this_step:, :],
                    batch_y_mark[:, self.label_len + steps_generated - this_step:self.label_len + steps_generated, :]
                ], dim=1)

        outputs = torch.cat(all_outputs, dim=1)  # (B, pred_len, 1)
        batch_y = batch_y[:, -self.pred_len:, :].to(self.device)

        return outputs, batch_y
    
    def vali(
        self,
        data_loader: data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        rolling: bool = False,
        rolling_step: int = 0,
    ):
        total_loss = []
        total_mape = []
        self.model.eval()
        with torch.no_grad():
            scaler = data_loader.dataset.scaler
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                outputs, batch_y = self._predict(
                    batch_x, batch_y, batch_x_mark, batch_y_mark,
                    rolling=rolling, rolling_step=rolling_step
                )

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                if scaler is not None:
                    outputs_orig = outputs * scaler.scale_[0] + scaler.mean_[0]
                    batch_y_orig = batch_y * scaler.scale_[0] + scaler.mean_[0]
                else:
                    outputs_orig = outputs
                    batch_y_orig = batch_y

                # MAPE per batch: avoid division by zero
                abs_y = torch.abs(batch_y_orig)
                nonzero_mask = abs_y > 1e-10
                if nonzero_mask.any():
                    mape = (torch.abs(batch_y_orig[nonzero_mask] - outputs_orig[nonzero_mask]) / abs_y[nonzero_mask]).mean().item() * 100
                    total_mape.append(mape)


        total_loss = np.average(total_loss)
        total_mape = np.average(total_mape) if total_mape else float('nan')
        self.model.train()
        return total_loss, total_mape

    def train(
        self,
        checkpoint_path: Path,
        patience: int = 7,
        verbose: bool = True,
        learning_rate: float = 0.001,
        train_epochs: int = 10,
        rolling_step: int = 0
    ):
        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=patience, verbose=verbose)

        model_optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        criterion = nn.MSELoss().to(self.device)

        use_rolling = rolling_step > 0

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device) # [Batch_Size, Window, 1]
                batch_y = batch_y.to(self.device) # [Batch_Size, Horizon, 1]
                batch_x_mark = batch_x_mark.to(self.device) # [Batch_Size, Window, num_time_features]
                batch_y_mark = batch_y_mark.to(self.device) # [Batch_Size, Horizon, num_time_features]
                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if torch.isfinite(total_norm):
                    model_optim.step()
                else:
                    model_optim.zero_grad()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mape = self.vali(self.val_loader, criterion)

            if use_rolling:
                vali_loss_rolling, vali_mape_rolling = self.vali(
                    self.val_loader, criterion,
                    rolling=True, rolling_step=rolling_step
                )
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                    "Vali Loss (single-shot): {3:.7f} Vali MAPE: {4:.2f}% "
                    "Vali Loss (rolling-{5}): {6:.7f} Vali MAPE (rolling): {7:.2f}%".format(
                        epoch + 1, train_steps, train_loss, vali_loss, vali_mape,
                        rolling_step, vali_loss_rolling, vali_mape_rolling
                    )
                )
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali MAPE: {4:.2f}%".format(
                    epoch + 1, train_steps, train_loss, vali_loss, vali_mape))

            early_stopping(vali_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1)

        best_model_path = checkpoint_path / 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    def _inverse_scale(
        self,
        data: npt.NDArray[np.float32],          # (B, T, 1)
        scaler: StandardScaler | None
    ) -> npt.NDArray[np.float32]:
        if scaler is None:
            return data
        B, T, C = data.shape
        data_2d = data.reshape(-1, C)      # (B*T, 1)
        data_2d = scaler.inverse_transform(data_2d)
        return data_2d.reshape(B, T, C)
    
    def predict(
        self,
        checkpoint_path: Path,
        load: bool = False
    ):

        if load:
            best_model_path = checkpoint_path / 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        timestamps = []
        real_ys = []
        global_errors = []

        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                real_y = batch_y[:, -self.pred_len:, 0:1]
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)
                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                mse_per_item = F.mse_loss(
                    outputs,
                    batch_y,
                    reduction='none'
                ).mean(dim=(1, 2))

                for error in mse_per_item:
                    global_errors.append(error.item())

                x_hist = batch_x[:,:,0:1].detach().cpu().numpy()     # (B, seq_len, 1)
                x_hist = self._inverse_scale(x_hist, self.train_loader.dataset.scaler)
                y_pred = outputs.detach().cpu().numpy()     # (B, pred_len, 1)
                y_pred = self._inverse_scale(y_pred, self.test_loader.dataset.scaler)
            
                real_y = real_y.detach().cpu().numpy()       # (B, pred_len, 1)
                real_y = self._inverse_scale(real_y, self.test_loader.dataset.scaler)
                real_ys.append(real_y)


                full_series = np.concatenate([x_hist, y_pred], axis=1)
                preds.append(full_series)
                window_timestamps = batch_x_mark.detach().cpu().numpy()
                pred_timestamps = batch_y_mark.detach().cpu().numpy()
                full_timestamps = np.concatenate([window_timestamps, pred_timestamps[:, -self.pred_len:, :]], axis=1)
                timestamps.append(full_timestamps)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        real_ys = np.array(real_ys)
        real_ys = real_ys.reshape(-1, real_ys.shape[-2], real_ys.shape[-1])

        timestamps = np.array(timestamps)
        timestamps = timestamps.reshape(-1, timestamps.shape[-2], timestamps.shape[-1])

        print(f"Shape of predictions: {preds.shape}")
        print(f"Shape of timestamps: {timestamps.shape}")
        print(f"First timestamp example: {timestamps[0]}")

        plots_paths = Path('results') / checkpoint_path  / 'graficas.pdf'
        plots_paths.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(plots_paths) as pdf:
            y_preds = []
            y_reals = []

            for start_index in range(0, len(self.all_dataset), 1):
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.all_dataset[start_index]
                real_y = batch_y[-self.pred_len:, 0:1]
                #Add batch dimension
                batch_x = torch.tensor(batch_x)
                batch_x = batch_x.unsqueeze(0)
                batch_y = torch.tensor(batch_y)
                batch_y = batch_y.unsqueeze(0)
                batch_x_mark = torch.tensor(batch_x_mark)
                batch_x_mark = batch_x_mark.unsqueeze(0)
                batch_y_mark = torch.tensor(batch_y_mark)
                batch_y_mark = batch_y_mark.unsqueeze(0)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)
                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                #Remove batch dimension because we are predicting one by one
                #outputs = outputs.squeeze(0)
                #batch_y = batch_y.squeeze(0)

                y_pred = outputs.detach().cpu().numpy()     # (pred_len, 1)
                y_pred = self._inverse_scale(y_pred, self.all_dataset.scaler)
                y_preds.append(y_pred[0,0:24])
                # real_y = real_y.detach().cpu().numpy()       # (pred_len, 1)
                real_y = self._inverse_scale(real_y[np.newaxis, :, :], self.all_dataset.scaler)
                y_reals.append(real_y[0,0:24])

                # plt.plot(real_y[0], label="Real", color="blue")
                # plt.plot(y_pred[0], label="Predicción", color="green", linestyle='dashed', alpha=0.5)
                # pdf.savefig()
                # plt.close()

            y_preds = np.array(y_preds).flatten()
            y_reals = np.array(y_reals).flatten()
            
            plt.plot(y_reals, label="Real", color="blue")
            plt.plot(y_preds, label="Predicción", color="green", linestyle='dashed', alpha=0.5)
            pdf.savefig()
            plt.close()


            

            # Generate plots for each prediction in Test set
            for i in range(len(preds)):
                start_datetime = datetime(2000, int(timestamps[i][0][0]), int(timestamps[i][0][1]), int(timestamps[i][0][3]))  # Dummy year
                dates = [start_datetime + timedelta(hours=j) for j in range(len(preds[i]))]
                plt.plot(dates[:self.seq_len], preds[i][:self.seq_len], label="Historia", color="blue")
                plt.plot(dates[self.seq_len:], preds[i][self.seq_len:], label="Predicción", color="orange")
                plt.plot(dates[self.seq_len:], real_ys[i], label="Real", color="green", linestyle='dashed')
                plt.xticks(dates[::112])
                plt.text(
                    0.99, 0.01,
                    f'Window Error (MSE): {global_errors[i]:.4f}',
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    ha='right',
                    va='bottom'
                )
                pdf.savefig()
                plt.close()

        mean_global_error = np.mean(global_errors)
        print(f"Global Error (MSE): {mean_global_error}")

    def predict_series(
        self,
        pdf: PdfPages,
        checkpoint_path: Path,
        rolling_step: int = 0,
        load: bool = True,
    ):
        """
        Predict over the whole series using all_dataset windows.
        Each window uses real (ground truth) values for the encoder input.

        - rolling_step=0: single-shot, takes all pred_len steps from each prediction,
          advances by pred_len (stride = pred_len // 24 windows).
        - rolling_step>0: takes only the first rolling_step hours from each prediction,
          advances by rolling_step (stride = rolling_step // 24 windows).
          Uses single-shot prediction (not _predict_rolling), since the encoder
          always has real data.
        """
        if load:
            best_model_path = checkpoint_path / 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        take_steps = self.pred_len if rolling_step == 0 else rolling_step
        # Window stride in the dataset is 24h (1 day), so to advance by take_steps hours
        # we skip take_steps // 24 windows
        window_advance = max(1, take_steps // 24)

        y_preds = []
        y_reals = []

        self.model.eval()
        with torch.no_grad():
            for idx in range(0, len(self.all_dataset), window_advance):
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.all_dataset[idx]
                real_y = batch_y[-self.pred_len:, 0:1]  # (pred_len, 1)

                # Add batch dimension
                batch_x = torch.tensor(batch_x).unsqueeze(0).to(self.device)
                batch_y = torch.tensor(batch_y).unsqueeze(0).to(self.device)
                batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).to(self.device)
                batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0).to(self.device)

                # Always single-shot: encoder has real data, no need for rolling
                outputs, _ = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                y_pred = outputs.detach().cpu().numpy()  # (1, pred_len, 1)
                y_pred = self._inverse_scale(y_pred, self.all_dataset.scaler)
                y_preds.append(y_pred[0, :take_steps])  # (take_steps, 1)

                real_y = self._inverse_scale(real_y[np.newaxis, :, :], self.all_dataset.scaler)
                y_reals.append(real_y[0, :take_steps])  # (take_steps, 1)

        y_preds_flat = np.array(y_preds).flatten()
        y_reals_flat = np.array(y_reals).flatten()

        Trainer.plot_and_print_ys(pdf, y_preds_flat, y_reals_flat, rolling_step)

        return y_preds_flat, y_reals_flat
    
    @staticmethod
    def plot_and_print_ys(
        pdf: PdfPages,
        y_preds_flat: npt.NDArray[np.float32],
        y_reals_flat: npt.NDArray[np.float32],
        rolling_step: int
    ):
        mse = np.mean((y_preds_flat - y_reals_flat) ** 2)
        mae = np.mean(np.abs(y_preds_flat - y_reals_flat))
        # MAPE: avoid division by zero by masking near-zero real values
        nonzero_mask = np.abs(y_reals_flat) > 1e-8
        mape = np.mean(np.abs((y_reals_flat[nonzero_mask] - y_preds_flat[nonzero_mask]) / y_reals_flat[nonzero_mask])) * 100

        mode_label = f"rolling-{rolling_step}" if rolling_step > 0 else "single-shot"
        print(f"predict_series ({mode_label}): MSE={mse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")

        plt.figure(figsize=(20, 6))
        plt.plot(y_reals_flat, label="Real", color="blue")
        plt.plot(y_preds_flat, label=f"Predicción ({mode_label})", color="green", linestyle='dashed', alpha=0.7)
        plt.legend()
        plt.title(f"Series prediction ({mode_label}) — MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        pdf.savefig()
        plt.close()

    def get_window_indexes(
        self,
        num_windows: int | None = None
    ) -> npt.NDArray[np.int_]:
        test_len = len(self.test_loader.dataset)
        if num_windows is None:
            return np.arange(test_len)
        else:
            window_indexes = self.rng.choice(
                test_len,
                size=num_windows,
                replace=False
            )
        return window_indexes

    def predict_windows(
        self,
        pdf: PdfPages,
        checkpoint_path: Path,
        rolling_step: int = 0,
        load: bool = True,
        windows_to_predict: int = 20
    )->List[PredictionWindow]:
        """
        Predict each test window individually and plot history + prediction + real.
        Same parameters as predict_series.

        - rolling_step=0: single-shot, predicts the full pred_len in one pass.
        - rolling_step>0: multi-step rolling prediction, predicts rolling_step
          at a time and feeds predictions back into the encoder to cover
          the full pred_len.
        """
        if load:
            best_model_path = checkpoint_path / 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        use_rolling = rolling_step > 0
        mode_label = f"rolling-{rolling_step}" if use_rolling else "single-shot"

        global_errors = []
        global_mapes = []

        self.model.eval()

        window_indexes = self.get_window_indexes(windows_to_predict)
        prediction_windows: List[PredictionWindow] = []
        with torch.no_grad():
            for window_idx in window_indexes:
                # Find the batch that contains this window index
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.test_loader.dataset[window_idx]
                
                batch_x = torch.tensor(batch_x, device=self.device)
                batch_y = torch.tensor(batch_y, device=self.device)
                batch_x_mark = torch.tensor(batch_x_mark, device=self.device)
                batch_y_mark = torch.tensor(batch_y_mark, device=self.device)

                batch_x = batch_x.unsqueeze(0)  # (1, seq_len, 1)
                batch_y = batch_y.unsqueeze(0)  # (1, seq_len + pred_len, 1)
                batch_x_mark = batch_x_mark.unsqueeze(0)  # (1, seq_len, d_mark)
                batch_y_mark = batch_y_mark.unsqueeze(0)   # (1, seq_len + pred_len, d_mark)

                real_y = batch_y[:, -self.pred_len:, 0:1]
                outputs, _ = self._predict(
                    batch_x, batch_y, batch_x_mark, batch_y_mark,
                    rolling=use_rolling, rolling_step=rolling_step
                )

                # Compute per-item MSE over the full pred_len
                mse_per_item = F.mse_loss(
                    outputs,
                    real_y,
                    reduction='none'
                ).mean(dim=(1, 2))

                # Inverse scale for plotting
                x_hist = batch_x[:, :, 0:1].detach().cpu().numpy()
                x_hist = self._inverse_scale(x_hist, self.train_loader.dataset.scaler)

                y_pred = outputs.detach().cpu().numpy()
                y_pred = self._inverse_scale(y_pred, self.test_loader.dataset.scaler)

                real_y_np = real_y.detach().cpu().numpy()
                real_y_np = self._inverse_scale(real_y_np, self.test_loader.dataset.scaler)

                timestamps = batch_x_mark.detach().cpu().numpy()

                # Plot each item in the batch
                for b in range(batch_x.shape[0]):
                    error = mse_per_item[b].item()
                    global_errors.append(error)

                    hist = x_hist[b, :, 0]  # (seq_len,)
                    pred = y_pred[b, :, 0]  # (pred_len,)
                    real = real_y_np[b, :, 0]  # (pred_len,)

                    # Per-window MAPE
                    nonzero = np.abs(real) > 1e-8
                    window_mape = np.mean(np.abs((real[nonzero] - pred[nonzero]) / real[nonzero])) * 100

                    # Build datetime axis
                    ts = timestamps[b]  # (seq_len, d_mark)
                    start_datetime = datetime(
                        2000, int(ts[0][0]), int(ts[0][1]), int(ts[0][3])
                    )
                    total_len = len(hist) + self.pred_len
                    dates = [start_datetime + timedelta(hours=j) for j in range(total_len)]

                    plt.figure(figsize=(16, 5))
                    plt.plot(dates[:self.seq_len], hist, label="Historia", color="blue")
                    plt.plot(dates[self.seq_len:], pred, label=f"Predicción ({mode_label})", color="orange")
                    plt.plot(dates[self.seq_len:], real, label="Real", color="green", linestyle='dashed')
                    plt.xticks(dates[::max(1, total_len // 6)], rotation=30)
                    plt.legend()
                    plt.text(
                        0.99, 0.01,
                        f'MSE: {error:.4f} | MAPE: {window_mape:.2f}%',
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        ha='right',
                        va='bottom'
                    )
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

                    global_mapes.append(window_mape)

        mean_error = np.mean(global_errors)
        mean_mape = np.mean(global_mapes)
        print(f"predict_windows ({mode_label}): Mean MSE={mean_error:.4f}, Mean MAPE={mean_mape:.2f}% over {len(global_errors)} windows")
        return prediction_windows

    def plot_prediction_windows(
        self,
        pdf: PdfPages,
        prediction_windows: List[PredictionWindow],
        mode_label: str
    ):
        for pw in prediction_windows:
            start_datetime = pw.history[0][0]
            total_len = len(pw.history[1]) + len(pw.predictions[1])
            dates = [start_datetime + timedelta(hours=j) for j in range(total_len)]
            plt.figure(figsize=(16, 5))
            plt.plot(pw.history[0], pw.history[1], label="Historia", color="blue")
            plt.plot(pw.predictions[0], pw.predictions[1], label=f"Predicción ({mode_label})", color="orange")
            plt.plot(pw.real_values[0], pw.real_values[1], label="Real", color="green", linestyle='dashed')
            plt.xticks(dates[::max(1, total_len // 6)], rotation=30)
            plt.legend()
            plt.text(
                        0.99, 0.01,
                        f'MSE: {error:.4f} | MAPE: {window_mape:.2f}%',
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        ha='right',
                        va='bottom'
                    )
            plt.tight_layout()
            pdf.savefig()
            plt.close()
