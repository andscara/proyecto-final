import logging
import time
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from forecasting.autoformer.tools import EarlyStopping, StandardScaler, adjust_learning_rate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy.typing as npt
from datetime import datetime, timedelta

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: data.DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        val_loader: data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        test_loader: data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
        label_len: int,
        pred_len: int,
        output_attention: bool,
        device_name: str = 'cpu',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.device = torch.device(device_name)
        self.model.to(self.device)

    def _predict(
        self,
        batch_x: torch.Tensor, 
        batch_y: torch.Tensor, 
        batch_x_mark: torch.Tensor, 
        batch_y_mark: torch.Tensor
    ):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.output_attention:
                outputs = outputs[0]
            return outputs

        # if self.args.use_amp:
        #     with torch.cuda.amp.autocast():
        #         outputs = _run_model()
        # else:
        outputs = _run_model()

        # f_dim = -1 if self.args.features == 'MS' else 0
        f_dim = 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
    
    def vali(
        self, 
        data_loader: data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module
    ):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(
        self,
        checkpoint_path: str,
        patience: int = 7,
        verbose: bool = True,
        learning_rate: float = 0.001,
        train_epochs: int = 10,
    ):
        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=patience, verbose=verbose)

        model_optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        criterion = nn.MSELoss()

        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()


        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device) # [Batch_Size, Window, 1]
                batch_y = batch_y.float().to(self.device) # [Batch_Size, Horizon, 1]
                batch_x_mark = batch_x_mark.float().to(self.device) # [Batch_Size, Window, num_time_features]
                batch_y_mark = batch_y_mark.float().to(self.device) # [Batch_Size, Horizon, num_time_features]
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

                # if self.args.use_amp:
                #     scaler.scale(loss).backward()
                #     scaler.step(model_optim)
                #     scaler.update()
                # else:
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.val_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1)

        best_model_path = checkpoint_path + '/' + 'checkpoint.pth'
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
        checkpoint_path: str,
        load: bool = False
    ):

        if load:
            best_model_path = checkpoint_path + '/' + 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        timestamps = []

        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                x_hist = batch_x.detach().cpu().numpy()     # (B, seq_len, 1)
                x_hist = self._inverse_scale(x_hist, self.train_loader.dataset.scaler)
                y_pred = outputs.detach().cpu().numpy()     # (B, pred_len, 1)
                y_pred = self._inverse_scale(y_pred, self.test_loader.dataset.scaler)

                # print(f"Shape of x_hist: {x_hist.shape}")
                # print(f"Shape of y_pred: {y_pred.shape}")

                full_series = np.concatenate([x_hist, y_pred], axis=1)
                preds.append(full_series)
                window_timestamps = batch_x_mark.detach().cpu().numpy()
                pred_timestamps = batch_y_mark.detach().cpu().numpy()
                # print(f"Window timestamps shape: {window_timestamps.shape}")
                # print(f"Prediction timestamps shape: {pred_timestamps.shape}")
                # Concatenate timestamps of x and the last part (pred_len) of y
                full_timestamps = np.concatenate([window_timestamps, pred_timestamps[:, -self.pred_len:, :]], axis=1)
                # full_timestamps = np.concatenate([window_timestamps, pred_timestamps], axis=1)
                timestamps.append(full_timestamps)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        timestamps = np.array(timestamps)
        timestamps = timestamps.reshape(-1, timestamps.shape[-2], timestamps.shape[-1])

        print(f"Shape of predictions: {preds.shape}")
        print(f"Shape of timestamps: {timestamps.shape}")
        print(f"First timestamp example: {timestamps[0]}")


        # # result save
        # folder_path = './results/' + checkpoint_path + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)

        plots_paths = './results/' + checkpoint_path + '/' + 'graficas.pdf'

        with PdfPages(plots_paths) as pdf:
            for i in range(len(preds)):
                start_datetime = datetime(2000, int(timestamps[i][0][0]), int(timestamps[i][0][1]), int(timestamps[i][0][3]))  # Dummy year
                dates = [start_datetime + timedelta(hours=j) for j in range(len(preds[i]))]
                plt.plot(dates[:self.seq_len], preds[i][:self.seq_len], label="Historia", color="blue")
                plt.plot(dates[self.seq_len:], preds[i][self.seq_len:], label="Predicción", color="orange")
                plt.xticks(dates[::112])
                pdf.savefig()
                plt.close()

