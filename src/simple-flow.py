#type: ignore

from tabnanny import check
from matplotlib import pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import random
import torch

from forecasting.autoformer.autoformer import Autoformer
from forecasting.autoformer.baseline import LinearBaseline
from forecasting.autoformer.baseline2 import LinearBaseline2
from forecasting.autoformer.prediction_window import PredictionWindow
from forecasting.autoformer.trainer import Trainer
from torch.utils import data
from pathlib import Path
from dotenv import load_dotenv
import os
import horizon as h
from enum import Enum
import numpy as np
from typing import List
from forecasting.autoformer.experiment_handler import experiment_factory, BaseExperimentHandler, ExperimentType
from forecasting.autoformer.experiment_configuration import ExperimentConfiguration


load_dotenv()
PATH = os.getenv("DATA_PATH")
WINDOW_SIZE = 24*7 # 2 weeks
HORIZON = h.Horizon(type = h.HorizonType.HOUR, length=24) # 7 day
BATCH_SIZE = 128
LABEL_LEN = WINDOW_SIZE // 2

# EXOG_COLS = ['temp_max', 'temp_min', 'temp_media']
EXOG_COLS = ['temp_media']
# EXOG_COLS = ['temperature']

class Region(Enum):
    NORTH = ("NORTH", "LA MAGNOLIA", ["ARTIGAS", "SALTO", "RIVERA", "TACUAREMBO", "CERRO LARGO"])
    SOUTH = ("SOUTH", "LAS BRUJAS", ["SAN JOSE", "COLONIA", "CANELONES", "FLORES", "FLORIDA", "SORIANO"])
    EAST = ("EAST", "PASO DE LA LAGUNA", ["MALDONADO", "ROCHA", "TREINTA Y TRES", "LAVALLEJA"])
    WEST = ("WEST", "GLENCOE", ["PAYSANDU","RIO NEGRO", "DURAZNO"])
    MONTEVIDEO = ("MONTEVIDEO", "LAS BRUJAS", ["MONTEVIDEO"])

    def __init__(
            self, 
            code: str, 
            estacion: str,
            departamentos: list[str]
        ):
        self.code = code
        self.estacion = estacion
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
    train: bool,
    expiment_type: ExperimentType,
    training_runs: int | None
):
    if training_runs is None:
        print("Setting seed for reproducibility...")
        # Fijar semillas para reproducibilidad
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Generator para DataLoaders
        generator = torch.Generator()
        generator.manual_seed(SEED)
        training_runs = 1
    else:
        generator = None

    global_prediction_windows: List[PredictionWindow] = []

    experiment_handler: BaseExperimentHandler = experiment_factory(
            experiment_type=expiment_type,
            db_path=os.getenv("DB_PATH"),
            data_path=os.getenv("DATA_PATH"),
            exp_config=ExperimentConfiguration(
                windows_size=WINDOW_SIZE,
                horizon=HORIZON,
                label_len=LABEL_LEN,
                stride=24, # every day
                target_col_name="agg_valor",
                scale=True,
                exog_cols=EXOG_COLS
            )
        )

    y_preds_flat_results = []
    y_reals_flat_results = []

    while experiment_handler.has_next():
        experiment_group = experiment_handler.next_experiment_group()
        train_dataloader = data.DataLoader(
            experiment_group.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            generator=generator
        )

        val_dataloader = data.DataLoader(
            experiment_group.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            generator=generator
        )

        test_dataloader = data.DataLoader(
            experiment_group.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            generator=generator
        )
        
        print("Creating model and trainer...")
        seq_len = WINDOW_SIZE
        pred_len = HORIZON.length
        def create_model():
            # return LinearBaseline(
            #     seq_len=seq_len,
            #     pred_len=pred_len,
            #     exog_size= False, #len(EXOG_COLS) if experiment_handler.use_exogenous() else 0,
            #     include_holiday=False
            # )
            return LinearBaseline2(
                seq_len=seq_len,
                pred_len=pred_len
            )
            # return Autoformer(
            #     seq_len=seq_len,
            #     label_len=LABEL_LEN,
            #     pred_len=pred_len,
            #     c_out=1,
            #     enc_in=1,
            #     dec_in=1,
            #     d_model=128,
            #     n_heads=2,
            #     d_ff=256,
            #     e_layers=2,
            #     d_layers=1,
            #     dropout=0,
            #     factor=5,
            #     d_mark=5, # 4 time features (month, day, weekday, hour) + 1 holiday col
            #     exog_c_in=1, # 1 temperature column (temp_media)
            #     use_exog_vars = experiment_handler.use_exogenous()
            # )
        trainer = Trainer(
            model_factory=create_model,
            window_stride_in_days=1,
            all_dataset=experiment_group.full_dataset,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            seq_len=seq_len,
            label_len=LABEL_LEN,
            pred_len=pred_len,
            output_attention=False,
            device_name=os.getenv("DEVICE_NAME") #mps for mac and cuda for gpu
        )

        checkpoint_path = Path("checkpoints") / experiment_group.name
        patience = 50
        lr = 0.00003
        train_epochs = 300
        setting = 'patience_{}_lr_{}_epochs_{}'.format(
            patience,
            lr,
            train_epochs
        )
        path = checkpoint_path / setting
        if not os.path.exists(path):
            os.makedirs(path)
        if train:
            print(f"Starting training for experiment group {experiment_group.name}...")
            trainer.train(
                patience=patience,
                verbose=False,
                learning_rate=lr,
                train_epochs=train_epochs,
                checkpoint_path=path,
                training_runs=training_runs
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
                prefix=f"{expiment_type.value} - {experiment_group.name}",
                pdf=pdf
            )
    plots_path = Path('results') / expiment_type.value / f'graficas.pdf'
    plots_path.parent.mkdir(parents=True, exist_ok=True)
    # need to sum all the y_preds_flat_results and y_reals_flat_results element-wise before flattening, since we want to compare the sum of the predictions of all regions with the sum of the real values of all regions
    y_preds_flat = np.sum(np.array(y_preds_flat_results), axis=0).flatten()
    y_reals_flat = np.sum(np.array(y_reals_flat_results), axis=0).flatten()
    with PdfPages(plots_path) as pdf:
        print_test_metrics(
            predictions=global_prediction_windows,
            prefix=f"{expiment_type.value} - Global",
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
    main(
        train=True, 
        expiment_type=ExperimentType.COUNTRY,
        training_runs=None
    )