import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch.nn as nn
import horizon as h

from forecasting.autoformer.experiment_handler import ExperimentType
from forecasting.autoformer.autoformer import Autoformer
from forecasting.autoformer.baseline import LinearBaseline
from forecasting.autoformer.baseline2 import LinearBaseline2


@dataclass
class FlowConfig:
    train: bool
    training_runs: int | None  # None = single run with fixed seed

    # data
    window_size: int
    horizon: h.Horizon
    label_len: int
    batch_size: int

    # experiment
    experiment_type: ExperimentType

    # training
    patience: int
    learning_rate: float
    train_epochs: int

    # model
    model_type: str
    model_params: dict

    @classmethod
    def from_toml(cls, path: str | Path) -> "FlowConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        horizon_type_map = {
            "hour": h.HorizonType.HOUR,
            "week": h.HorizonType.WEEK,
            "day":  h.HorizonType.DAY,
        }
        experiment_type_map = {
            "country": ExperimentType.COUNTRY,
            "regions": ExperimentType.REGIONS,
        }

        data     = raw["data"]
        training = raw["training"]
        model    = raw["model"]

        training_runs_raw = raw.get("training_runs", 0)
        training_runs = None if training_runs_raw == 0 else int(training_runs_raw)

        return cls(
            train=bool(raw["train"]),
            training_runs=training_runs,
            window_size=int(data["window_size"]),
            horizon=h.Horizon(
                type=horizon_type_map[data["horizon_type"]],
                length=int(data["horizon_length"]),
            ),
            label_len=int(data["label_len"]),
            batch_size=int(data["batch_size"]),
            experiment_type=experiment_type_map[raw["experiment"]["type"]],
            patience=int(training["patience"]),
            learning_rate=float(training["learning_rate"]),
            train_epochs=int(training["train_epochs"]),
            model_type=str(model["type"]),
            model_params={k: v for k, v in model.items() if k != "type"},
        )

    def make_model_factory(
        self,
        seq_len: int,
        pred_len: int,
        use_exog: bool,
    ) -> Callable[[], nn.Module]:
        p = self.model_params
        label_len = self.label_len

        if self.model_type == "linear_baseline":
            exog_size = len(p.get("exog_cols", ["temp_media"])) if use_exog else 0

            def factory() -> nn.Module:
                return LinearBaseline(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    exog_size=exog_size,
                    include_holiday=bool(p.get("include_holiday", True)),
                )

        elif self.model_type == "linear_baseline2":
            def factory() -> nn.Module:
                return LinearBaseline2(seq_len=seq_len, pred_len=pred_len)

        elif self.model_type == "autoformer":
            def factory() -> nn.Module:
                return Autoformer(
                    seq_len=seq_len,
                    label_len=label_len,
                    pred_len=pred_len,
                    c_out=1,
                    enc_in=1,
                    dec_in=1,
                    d_model=int(p.get("d_model", 128)),
                    n_heads=int(p.get("n_heads", 2)),
                    d_ff=int(p.get("d_ff", 256)),
                    e_layers=int(p.get("e_layers", 2)),
                    d_layers=int(p.get("d_layers", 1)),
                    dropout=float(p.get("dropout", 0.0)),
                    factor=int(p.get("factor", 5)),
                    d_mark=int(p.get("d_mark", 5)),
                    exog_c_in=int(p.get("exog_c_in", 1)) if use_exog else 0,
                    use_exog_vars=use_exog,
                )

        else:
            raise ValueError(f"Unknown model type: {self.model_type!r}")

        return factory
