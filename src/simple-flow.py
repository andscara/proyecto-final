#type: ignore

import time
import duckdb as ddb
from numpy import c_
import pandas as pd

from forecasting.autoformer.autoformer import Autoformer
from forecasting.autoformer.data_loader import TestDataset, TrainDataset
from forecasting.autoformer.trainer import Trainer
from torch.utils import data
from forecasting.autoformer.data_loader import data_splitter



PATH = "C:\\Users\\andres\\Documents\\ute\\cleanup\\res-outliers"
WINDOW_SIZE = 24*7*2 # 2 weeks
HORIZON = 24*7 # 1 week
BATCH_SIZE = 64


def main():
    query = f"""
    select departamento, dia, hora, SUM(valor) as agg_valor
    from read_parquet('{PATH}')
    group by departamento, dia, hora
    order by departamento, dia, hora;
    """   
    con = ddb.connect()
    ts_agg_departamento = con.execute(query).fetchdf()
    print(f"Cantidad de registros totales en todos los departamentos agregados: {len(ts_agg_departamento)}")
    con.close()
    print ("Creating datasets...")
    montevideo_data = ts_agg_departamento[ts_agg_departamento["departamento"] == "MONTEVIDEO"]
    # Train & Test DataLoader
    
    train_dataset, test_dataset = data_splitter(
        df=montevideo_data,
        train_val_ratio=0.8,
        test_ratio=0.2,
        windows_size=WINDOW_SIZE,
        horizon=HORIZON,
        stride=24*3, # every 3 days
        target_col_name="agg_valor",
        scale=True
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
    label_len = WINDOW_SIZE // 2
    pred_len = HORIZON
    model = Autoformer(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        c_out=1,
        enc_in=1,
        dec_in=1
    )
    trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=test_dataloader,
        label_len=label_len,
        pred_len=pred_len,
        output_attention=False,
        device_name='cuda' #mps for mac and cuda for gpu
    )
    print("Starting training...")
    trainer.train(
        patience=5,
        verbose=True,
        learning_rate=0.001,
        train_epochs=10
    )

    ...

if __name__ == "__main__":
    main()