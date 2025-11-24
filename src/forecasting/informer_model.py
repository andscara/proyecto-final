import io
import logging
import time

import pandas as pd
import torch
from forecasting.forecast_model import ForecastModel, ForecastModelID
from dataclasses import dataclass

from forecasting.forecast_result import ForecastPredictionResult
from forecasting.informer.data_loader import CustomDataset
from forecasting.informer.model import Informer
from forecasting.informer.tools import EarlyStopping, adjust_learning_rate
from input import Input
from metrics import ForecastMetricType
from storage.storage import BaseStorage
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np

@dataclass
class InformerForecastModelID(ForecastModelID):
    id: str
    enc_in: int
    dec_in: int 
    c_out: int
    seq_len: int
    label_len: int
    out_len: int
    factor: int
    d_model: int
    n_heads: int
    e_layers: int
    d_layers: int
    d_ff: int
    padding_type: int
    features: str
    train_epochs: int
    learning_rate: float
    batch_size: int
    patience: int
    cross_validation_count: int
    device: str
    frequency: str


class InformerForecastModel(ForecastModel):
    
    def __init__(
            self,
            logger: logging.Logger,
            enc_in: int,
            dec_in: int, 
            c_out: int,
            seq_len: int,
            label_len: int,
            out_len: int,
            factor: int = 5, 
            d_model: int =512,
            n_heads: int = 8,
            e_layers: int = 3,
            d_layers: int = 2,
            d_ff: int = 51,
            padding_type: int = 0,
            features: str = 'S',
            train_epochs: int = 10,
            learning_rate: float = 0.001,
            batch_size: int = 32,
            patience: int = 3,
            cross_validation_count: int = 5,
            device: str = "cpu",
            frequency: str = 'h'
        ):
        model_id = InformerForecastModelID(
            id= "informer_model",
            enc_in= enc_in,
            dec_in= dec_in,
            c_out= c_out,
            seq_len= seq_len,
            label_len= label_len,
            out_len= out_len,
            factor= factor,
            d_model= d_model,
            n_heads= n_heads,
            e_layers= e_layers,
            d_layers= d_layers,
            d_ff= d_ff,
            padding_type = padding_type,
            features= features,
            train_epochs= train_epochs,
            learning_rate= learning_rate,
            batch_size= batch_size,
            patience= patience,
            cross_validation_count= cross_validation_count,
            device= device,
            frequency= frequency
        )
        super().__init__(logger, model_id)
        assert enc_in > 0, "enc_in must be a positive integer."
        self._enc_in = enc_in
        assert dec_in > 0, "dec_in must be a positive integer."
        self._dec_in = dec_in
        assert c_out > 0, "c_out must be a positive integer."
        self._c_out = c_out
        assert seq_len > 0, "seq_len must be a positive integer."
        self._seq_len = seq_len
        assert label_len > 0, "label_len must be a positive integer."
        self._label_len = label_len
        assert out_len > 0, "out_len must be a positive integer."
        self._out_len = out_len
        assert factor > 0, "factor must be a positive integer."
        self._factor = factor
        assert d_model > 0, "d_model must be a positive integer."
        self._d_model = d_model
        assert n_heads > 0, "n_heads must be a positive integer."
        self._n_heads = n_heads
        assert e_layers > 0, "e_layers must be a positive integer."
        self._e_layers = e_layers
        assert d_layers > 0, "d_layers must be a positive integer."
        self._d_layers = d_layers
        assert d_ff > 0, "d_ff must be a positive integer."
        self._d_ff = d_ff
        assert padding_type in [0, 1], "padding_type must be either 0 (zero padding) or 1 (one padding)."
        self._padding_type = padding_type
        assert features in ['S', 'M', 'MS'], "features must be one of 'S', 'M', or 'MS'."
        self._features = features
        assert train_epochs > 0, "train_epochs must be a positive integer."
        self._train_epochs = train_epochs
        assert learning_rate > 0, "learning_rate must be a positive number."
        self._learning_rate = learning_rate
        assert cross_validation_count > 0, "cross_validation_count must be a positive integer."
        self._cross_validation_count = cross_validation_count
        assert batch_size > 0, "batch_size must be a positive integer."
        self._batch_size = batch_size
        assert patience > 0, "patience must be a positive integer."
        self._patience = patience
        assert device in ["cpu", "cuda", "mps"], "Device must be either 'cpu', 'cuda', or 'mps'"
        self._device = torch.device(device)
        assert frequency in ['h', 'd', 'w'], "frequency must be one of 'h', 'd', or 'w'."
        self._frequency = frequency
        self._trained_models: dict[ForecastMetricType, Informer] = {}

    def _get_optimizer(
            self,
            model: Informer
        )-> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=self._learning_rate)

    def _get_criterion(self, metric: ForecastMetricType) -> nn.Module:
        if metric == ForecastMetricType.MSE:
            return nn.MSELoss()
        elif metric == ForecastMetricType.MAE:
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported metric type: {metric}")
        
    def _process_one_batch(
            self, 
            model: Informer,
            batch_x: torch.Tensor, 
            batch_y: torch.Tensor, 
            batch_x_mark: torch.Tensor, 
            batch_y_mark: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_x = batch_x.float().to(self._device)
        batch_y = batch_y.float().to(self._device)
        batch_x_mark = batch_x_mark.float().to(self._device)
        batch_y_mark = batch_y_mark.float().to(self._device)
        # decoder input
        dec_inp: torch.Tensor | None = None
        if self._padding_type==0:
            dec_inp = torch.zeros([batch_y.shape[0], self._out_len, batch_y.shape[-1]]).float()
        elif self._padding_type==1:
            dec_inp = torch.ones([batch_y.shape[0], self._out_len, batch_y.shape[-1]]).float()
        if dec_inp is None:
            raise ValueError("dec_inp is not initialized.")
        dec_inp = torch.cat([batch_y[:,:self._label_len,:], dec_inp], dim=1).float().to(self._device)
        # encoder - decoder
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self._features=='MS' else 0
        batch_y = batch_y[:,-self._out_len:,f_dim:].to(self._device)
        return outputs, batch_y

    def _validate(
            self, 
            model: Informer,
            vali_loader: DataLoader[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], 
            criterion: nn.Module
        ):
        model.eval()
        total_loss: list[float] = []
        for _, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(model, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss.item())
        model.train()
        return np.average(total_loss)

    def _fit(
        self, 
        model: Informer,
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame,
        metric: ForecastMetricType
    ) -> float:
        train_dataset = CustomDataset(
            df_raw=train_df,
            size=[self._seq_len, self._label_len, self._out_len],
            features=self._features,
            target='kmw',
            scale=True,
            inverse=False,
            timeenc=0,
            freq=self._frequency,
            cols=None
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            num_workers = 4 # TODO Arbitrary number, make configurable later 
        )
        val_dataset = CustomDataset(
            df_raw=val_df,
            size=[self._seq_len, self._label_len, self._out_len],
            features=self._features,
            target='kmw',
            scale=True,
            inverse=False,
            timeenc=0,
            freq=self._frequency,
            cols=None
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            num_workers = 4 # TODO Arbitrary number, make configurable later 
        )

        train_steps = len(train_dataloader)
        early_stopping = EarlyStopping(
            patience=self._patience,
            verbose=True
        )

        model_optim = self._get_optimizer(model)
        criterion = self._get_criterion(metric)

        time_now = time.time()
        average_train_loss = 0.0
        for epoch in range(self._train_epochs):
            iter_count = 0
            train_loss: list[float] = []
            model.train()
            epoch_time: float = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_dataloader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(model, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                loss_item = loss.item()
                train_loss.append(loss_item)
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_item))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self._train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                average_train_loss = float(np.average(train_loss)) if train_loss else 0.0
                vali_loss = self._validate(model, val_dataloader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, average_train_loss, vali_loss))
                early_stopping(vali_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(
                    model_optim,
                    epoch+1,
                    self._learning_rate,
                    'type1'
                )
        return average_train_loss

    def fit(
        self, 
        data: pd.DataFrame, 
        input: Input
    ) -> dict[ForecastMetricType, float]:
        cross_validation_results: list[dict[ForecastMetricType, float]] = []
        self._best_model_per_metric = {}
        for iteration, _ in enumerate(range(self._cross_validation_count)):
            self.logger.info(f"Starting cross-validation fold {iteration + 1} of {self._cross_validation_count}")
            train_df, val_df = self.get_train_val_dfs(data, input)
            metrics: dict[ForecastMetricType, float] = {}
            for metric in ForecastMetricType:
                self.logger.info(f"Starting cross-validation fold with metric: {metric.name}")
                model =  Informer(
                    enc_in=self._enc_in,
                    dec_in=self._dec_in,
                    c_out=self._c_out,
                    seq_len=self._seq_len,
                    label_len=self._label_len,
                    out_len=self._out_len,
                    factor=self._factor,
                    d_model=self._d_model,
                    n_heads=self._n_heads,
                    e_layers=self._e_layers,
                    d_layers=self._d_layers,
                    d_ff=self._d_ff
                )
                metric_score = self._fit(model, train_df, val_df, metric)
                metrics[metric] = metric_score
                self.logger.info(f"Completed cross-validation fold with metric: {metric.name}, score: {metric_score}")
            cross_validation_results.append(metrics)
        # Return the average metrics across cross-validation folds
        averaged_metrics: dict[ForecastMetricType, float] = {}
        for metric in ForecastMetricType:
            averaged_metrics[metric] = sum(
                metric_result.get(metric, 0.0) for metric_result in cross_validation_results
            ) / self._cross_validation_count


        # We train the final model using all the data available with the exception of the last possible validation window for validation
        train_df, val_df = self.get_train_val_dfs(data, input, validation_last_possible_window=True)
        for metric in ForecastMetricType:
            self.logger.info(f"Training final model for metric: {metric.name} on full training data")
            model =  Informer(
                enc_in=self._enc_in,
                dec_in=self._dec_in,
                c_out=self._c_out,
                seq_len=self._seq_len,
                label_len=self._label_len,
                out_len=self._out_len,
                factor=self._factor,
                d_model=self._d_model,
                n_heads=self._n_heads,
                e_layers=self._e_layers,
                d_layers=self._d_layers,
                d_ff=self._d_ff
            )
            self._fit(model, train_df, val_df, metric)
            self._trained_models[metric] = model
            self.logger.info(f"Completed training final model for metric: {metric.name}")
        return averaged_metrics
    
    def predict(
        self,
        data: pd.DataFrame, 
        metric: ForecastMetricType,
        input: Input
    ) -> ForecastPredictionResult:
        model = self._trained_models.get(metric)
        if model is None:
            raise ValueError(f"No trained model found for metric: {metric.name}")
        
        model.eval()
        dataset = CustomDataset(
            df_raw=data,
            size=[self._seq_len, self._label_len, self._out_len],
            features=self._features,
            target='kmw',
            scale=True,
            inverse=False,
            timeenc=0,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=False,
            num_workers = 4 # TODO Arbitrary number, make configurable later
        )
        raw_preds: list[np.ndarray] = []

        for _, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(dataloader):
            pred, _ = self._process_one_batch(model, batch_x, batch_y, batch_x_mark, batch_y_mark)
            raw_preds.append(pred.detach().cpu().numpy())

        np_preds = np.array(raw_preds)
        np_preds = np_preds.reshape(-1, np_preds.shape[-2], np_preds.shape[-1])


        predictions: list[float] = []
        # We are only interested in the kmw predictions
        for i in range(np_preds.shape[0]):
            for j in range(np_preds.shape[1]):
                predictions.append(float(np_preds[i][j][0]))  # Assuming kmw is the first feature
                
        return ForecastPredictionResult(predictions=predictions)

        
    
    def save_model(self, storage: BaseStorage) -> None:
        for metric, model in self._trained_models.items():
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer) #type: ignore
            buffer.seek(0)
            storage.save(f"informer_model_{metric.name}.pth", buffer.read())
    
    def load_model(self, storage: BaseStorage) -> None:
        for metric in ForecastMetricType:
            buffer = io.BytesIO()
            model_data = storage.load(f"informer_model_{metric.name}.pth")
            buffer.write(model_data)
            buffer.seek(0)
            model =  Informer(
                enc_in=self._enc_in,
                dec_in=self._dec_in,
                c_out=self._c_out,
                seq_len=self._seq_len,
                label_len=self._label_len,
                out_len=self._out_len,
                factor=self._factor,
                d_model=self._d_model,
                n_heads=self._n_heads,
                e_layers=self._e_layers,
                d_layers=self._d_layers,
                d_ff=self._d_ff
            )
            model.load_state_dict(torch.load(buffer)) #type: ignore
            model.to(self._device)
            self._trained_models[metric] = model
    
    
