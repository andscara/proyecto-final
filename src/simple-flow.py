#type: ignore

from tabnanny import check
import time
import duckdb as ddb
from matplotlib import pyplot as plt
from numpy import c_
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages

from forecasting.autoformer.autoformer import Autoformer
from forecasting.autoformer.prediction_window import PredictionWindow
from forecasting.autoformer.trainer import Trainer
from torch.utils import data
from forecasting.autoformer.data_loader import data_splitter
from pathlib import Path
from dotenv import load_dotenv
import os
import horizon as h
from enum import Enum
import numpy as np
from typing import List

from runner import predict


load_dotenv()
PATH = os.getenv("DATA_PATH")
WINDOW_SIZE = 24*7*2 # 2 weeks
HORIZON = h.Horizon(type = h.HorizonType.HOUR, length=24*7) # 1 week
BATCH_SIZE = 64
LABEL_LEN = WINDOW_SIZE

EXOG_COLS = ['temp_max', 'temp_min', 'temp_media']
# EXOG_COLS = ['temperature']

class Region(Enum):
    NORTH = ("NORTH", ["ARTIGAS", "SALTO", "RIVERA", "TACUAREMBO", "CERRO LARGO"])
    # SOUTH = ("SOUTH", ["SAN JOSE", "COLONIA", "CANELONES", "FLORES", "FLORIDA", "SORIANO"])
    # EAST = ("EAST", ["MALDONADO", "ROCHA", "TREINTA Y TRES", "LAVALLEJA"])
    # WEST = ("WEST", ["PAYSANDU","RIO NEGRO", "DURAZNO"])
    # MONTEVIDEO = ("MONTEVIDEO", ["MONTEVIDEO"])

    def __init__(self, code: str, departamentos: list[str]):
        self.code = code
        self.departamentos = departamentos


def print_test_metrics(
    predictions: List[PredictionWindow],
    prefix: str,
    pdf: PdfPages,
):
    # Calculate the error metrics for the aggregated predictions and print them in the pdf as text, using the global_prediction_windows to calculate the error metrics for each window and then averaging them to get the global error metrics
    errors = PredictionWindow.calculate_error_metrics(predictions)
    plt.figure(figsize=(10, 6))
    metrics_pattern = f"""
    {prefix} MSE: {errors['mse']:.4f} 
    {prefix} MIN MSE: {errors['min_mse']:.4f} 
    {prefix} MAX MSE: {errors['max_mse']:.4f}\n
    {prefix} MAE: {errors['mae']:.4f} 
    {prefix} MIN MAE: {errors['min_mae']:.4f} 
    {prefix} MAX MAE: {errors['max_mae']:.4f}\n
    {prefix} MAPE: {errors['mape']:.2f} 
    {prefix} MIN MAPE: {errors['min_mape']:.2f} 
    {prefix} MAX MAPE: {errors['max_mape']:.2f}\n
    """

    plt.text(0.1, 0.5, metrics_pattern, fontsize=12)
    plt.title(f"{prefix} Error Metrics", fontsize=16)
    plt.axis('off')
    pdf.savefig()
    plt.close()

def main(
    train: bool = True,
    clustering_type: str = "regional"
):
    # query = f"""
    # select departamento, dia, hora, SUM(valor) as agg_valor
    # from read_parquet('{PATH}')
    # group by departamento, dia, hora
    # order by departamento, dia, hora;
    # """

    # query = f"""
    # select e.departamento, e.dia, e.hora, agg_valor, (temp_max + 15) / 65 as temp_max, (temp_min + 15) / 65 as temp_min, (temp_media + 15) / 65 as temp_media
    # from (
    #     select departamento, dia, hora, SUM(valor) as agg_valor
    #     from read_parquet('{PATH}')
    #     group by departamento, dia, hora
    # ) e inner join temperatura_departamento t on e.dia=t.dia and e.departamento=t.departamento
    # order by e.departamento, e.dia, e.hora
    # """
    y_preds_flat_results = []
    y_reals_flat_results = []
    global_prediction_windows: List[PredictionWindow] = []
    for region in Region:
        query = f"""
        select e.dia, e.hora, SUM(agg_valor) as agg_valor, AVG((temp_max + 15) / 65) as temp_max, AVG((temp_min + 15) / 65) as temp_min, AVG((temp_media + 15) / 65) as temp_media
        from (
            select departamento, dia, hora, SUM(valor) as agg_valor
            from read_parquet('{PATH}')
            where departamento in {tuple(region.departamentos)}
            group by departamento, dia, hora
        ) e inner join temperatura_departamento t on e.dia=t.dia and e.departamento=t.departamento
        group by e.dia, e.hora
        order by e.dia, e.hora
        """

        con = ddb.connect(database=os.getenv("DB_PATH"))
        ts_agg_region = con.execute(query).fetchdf()
        print(f"Cantidad de registros totales en todos los departamentos agregados: {len(ts_agg_region)}")
        con.close()
        print ("Creating datasets...")
        region_data = ts_agg_region
        # Train & Test DataLoader
        
        all_dataset, train_dataset, val_dataset, test_dataset = data_splitter(
            df=region_data,
            windows_size=WINDOW_SIZE,
            horizon=HORIZON,
            label_len=LABEL_LEN,
            stride=24, # every day
            target_col_name="agg_valor",
            scale=True,
            exog_cols=EXOG_COLS
        )

        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False
        )

        val_dataloader = data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )

        test_dataloader = data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )
        print("Datasets created.")
        
        print("Creating model and trainer...")
        seq_len = WINDOW_SIZE
        pred_len = HORIZON.length
        model = Autoformer(
            seq_len=seq_len,
            label_len=LABEL_LEN,
            pred_len=pred_len,
            c_out=1,
            enc_in=1,
            dec_in=1,
            d_model=256,
            n_heads=4,
            d_ff=1024,
            e_layers=3,
            d_layers=2,
            dropout=0,
            factor=2,
            d_mark=8  # 4 time features (month, day, weekday, hour) + 3 temperature col + 1 holiday col
        )
        trainer = Trainer(
            model=model,
            window_stride_in_days=1,
            all_dataset=all_dataset,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            seq_len=seq_len,
            label_len=LABEL_LEN,
            pred_len=pred_len,
            output_attention=False,
            device_name=os.getenv("DEVICE_NAME") #mps for mac and cuda for gpu
        )

        checkpoint_path = Path("checkpoints") / region.code
        patience = 50
        lr = 0.00001 
        train_epochs = 1
        setting = 'patience_{}_lr_{}_epochs_{}'.format(
            patience,
            lr,
            train_epochs
        )
        path = checkpoint_path / setting
        if not os.path.exists(path):
            os.makedirs(path)
        if train:
            print(f"Starting training for region {region.code}...")
            trainer.train(
                patience=patience,
                verbose=True,
                learning_rate=lr,
                train_epochs=train_epochs,
                checkpoint_path=path
            )
            print("Training finished.")
        print("Starting testing...")
        plots_path = Path('results') / path / f'graficas.pdf'
        plots_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(plots_path) as pdf:
            trainer.predict_series(
                pdf=pdf,
                checkpoint_path=path,
                rolling_step=0,
                load= not train
            )
            y_preds_flat, y_reals_flat = trainer.predict_series(
                pdf=pdf,
                checkpoint_path=path,
                rolling_step=24 * 1,
                load= not train
            )
            y_preds_flat_results.append(y_preds_flat)
            y_reals_flat_results.append(y_reals_flat)
            region_prediction_windows: List[PredictionWindow] = trainer.predict_windows(
                checkpoint_path=path,
                rolling_step=0,
                load= not train
            )
            # We need to agregate each item in the existing global_prediction_windows with the corresponding item in the region_prediction_windows
            if len(global_prediction_windows) == 0:
                global_prediction_windows = region_prediction_windows
            else:
                global_prediction_windows = [
                    pw_global.aggregate(pw_region)
                    for pw_global, pw_region in zip(global_prediction_windows, region_prediction_windows)
                ]
            Trainer.plot_prediction_windows(
                pdf=pdf,
                prediction_windows=region_prediction_windows,
                rolling_step=0
            )
            print_test_metrics(
                predictions=region_prediction_windows,
                prefix=f"{clustering_type} - {region.code}",
                pdf=pdf
            )
            
    plots_path = Path('results') / clustering_type / f'graficas.pdf'
    plots_path.parent.mkdir(parents=True, exist_ok=True)
    # need to sum all the y_preds_flat_results and y_reals_flat_results element-wise before flattening, since we want to compare the sum of the predictions of all regions with the sum of the real values of all regions
    y_preds_flat = np.sum(np.array(y_preds_flat_results), axis=0).flatten()
    y_reals_flat = np.sum(np.array(y_reals_flat_results), axis=0).flatten()
    with PdfPages(plots_path) as pdf:
        print_test_metrics(
            predictions=global_prediction_windows,
            prefix=f"{clustering_type} - Global",
            pdf=pdf
        )
        # Plot the entire aggregated predictions vs real values
        Trainer.plot_and_print_ys(
            pdf=pdf, 
            y_preds_flat=y_preds_flat, 
            y_reals_flat=y_reals_flat,
            rolling_step=24 * 1
        )
        # Plot the aggregated prediction windows
        Trainer.plot_prediction_windows(
            pdf=pdf,
            prediction_windows=global_prediction_windows,
            rolling_step=0
        )
        


if __name__ == "__main__":
    main(train=True)