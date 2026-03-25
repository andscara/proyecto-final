#type: ignore

import argparse
from matplotlib import pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import random
import torch

from forecasting.autoformer.prediction_window import PredictionWindow
from forecasting.autoformer.trainer import Trainer
from forecasting.autoformer.sarimax_runner import SARIMAXRunner
from forecasting.autoformer.flow_config import FlowConfig
from torch.utils import data
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

load_dotenv()
from typing import List
from forecasting.autoformer.experiment_handler import experiment_factory, BaseExperimentHandler, ExperimentType
from forecasting.autoformer.experiment_configuration import ExperimentConfiguration

EXOG_COLS = ["temp_media"]


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
    {prefix} MAX MAPE: {errors['max_mape']:.2f} 
    {prefix} STD MAPE: {errors['std_mape']:.2f}
    """


    plt.text(0.1, 0.5, metrics_pattern, fontsize=12)
    plt.title(f"{prefix} Error Metrics", fontsize=16)
    plt.axis('off')
    pdf.savefig()
    plt.close()


    #Plot all mapes
    all_mapes = errors['all_mapes']
    plt.figure(figsize=(10, 6))
    # print the bin values in the histogram
    counts, bins, patches = plt.hist(all_mapes, bins=20, edgecolor='black')
    #print in the x-axis the bin values with 2 decimal places
    plt.xticks(bins, [f"{b:.2f}" for b in bins], rotation=45)
    plt.title(f"{prefix} MAPE Distribution", fontsize=16)
    plt.xlabel("MAPE", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    pdf.savefig()
    plt.close()

def main(cfg: FlowConfig, config_name: str, experiment_type: ExperimentType):
    train          = cfg.train
    expiment_type  = experiment_type
    training_runs  = cfg.training_runs
    WINDOW_SIZE    = cfg.window_size
    HORIZON        = cfg.horizon
    BATCH_SIZE     = cfg.batch_size
    LABEL_LEN      = cfg.label_len

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

    clustering_results_path = Path('results') / expiment_type.value / config_name / "clustering_assignments.json"

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
            ),
            clustering_path=os.getenv("CLUSTERING_PATH"),
            clustering_results_path=clustering_results_path,
            run_clustering=cfg.run_clustering,
            n_clusters_per_region=cfg.n_clusters_per_region,
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
        
        seq_len = WINDOW_SIZE
        pred_len = HORIZON.length
        plots_path = Path('results') / expiment_type.value / config_name / f'{experiment_group.name}.pdf'
        plots_path.parent.mkdir(parents=True, exist_ok=True)

        if cfg.is_sarima:
            print(f"Running SARIMAX for {experiment_group.name}...")
            runner = SARIMAXRunner(
                df=experiment_group.raw_df,
                seq_len=seq_len,
                pred_len=pred_len,
                stride=24,
                target_col="agg_valor",
                exog_cols=EXOG_COLS if experiment_handler.use_exogenous() else None,
            )
            with PdfPages(plots_path) as pdf:
                y_preds_flat, y_reals_flat = runner.predict_series(pdf=pdf)
                y_preds_flat_results.append(y_preds_flat)
                y_reals_flat_results.append(y_reals_flat)
                region_prediction_windows: List[PredictionWindow] = runner.predict_windows()
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
                    rolling_step=0,
                )
                print_test_metrics(
                    predictions=region_prediction_windows,
                    prefix=f"{expiment_type.value} - {experiment_group.name}",
                    pdf=pdf,
                )
        else:
            print("Creating model and trainer...")
            create_model = cfg.make_model_factory(
                seq_len=seq_len,
                pred_len=pred_len,
                use_exog=experiment_handler.use_exogenous(),
            )
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
            patience = cfg.patience
            lr = cfg.learning_rate
            train_epochs = cfg.train_epochs
            setting = 'patience_{}_lr_{}_epochs_{}'.format(patience, lr, train_epochs)
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
            with PdfPages(plots_path) as pdf:
                trainer.predict_series(
                    pdf=pdf,
                    checkpoint_path=path,
                    rolling_step=0,
                    load=not train
                )
                y_preds_flat, y_reals_flat = trainer.predict_series(
                    pdf=pdf,
                    checkpoint_path=path,
                    rolling_step=24 * 1,
                    load=not train
                )
                y_preds_flat_results.append(y_preds_flat)
                y_reals_flat_results.append(y_reals_flat)
                region_prediction_windows: List[PredictionWindow] = trainer.predict_windows(
                    checkpoint_path=path,
                    rolling_step=0,
                    load=not train
                )
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
                trainer.plot_temp_encoder_thresholds(pdf=pdf)
    plots_path = Path('results') / f'{expiment_type.value}_{config_name}.pdf'
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to TOML config file (e.g. configs/baseline_country.toml)")
    parser.add_argument("experiment_type", choices=[e.value for e in ExperimentType], help="Experiment type (e.g. country, regions, region_clustering)")
    args = parser.parse_args()
    config_name = Path(args.config).stem
    main(FlowConfig.from_toml(args.config), config_name, ExperimentType(args.experiment_type))