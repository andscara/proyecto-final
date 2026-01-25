#type: ignore

from tabnanny import check
import time
import duckdb as ddb
from numpy import c_
import pandas as pd
import os

from forecasting.autoformer.autoformer import Autoformer
from forecasting.autoformer.data_loader import ValTestDataset, TrainDataset
from forecasting.autoformer.trainer import Trainer
from torch.utils import data
from forecasting.autoformer.data_loader import data_splitter



PATH = "/Users/martin_martinez/ORT/Tesis/ute/final_datasets/residenciales/residenciales_complete_ids_only/"
WINDOW_SIZE = 24*7*2 # 2 weeks
HORIZON = 24*7 # 1 week
BATCH_SIZE = 64


def main(
    train: bool = True
):
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
    
    train_dataset, val_dataset, test_dataset = data_splitter(
        df=montevideo_data,
        windows_size=WINDOW_SIZE,
        horizon=HORIZON,
        stride=24, # every day
        target_col_name="agg_valor",
        scale=True
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
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        output_attention=False,
        device_name='cpu' #mps for mac and cuda for gpu
    )

    checkpoint_path = "./checkpoints" 
    patience = 5
    lr = 0.0001
    train_epochs = 20
    setting = 'patience_{}_lr_{}_epochs_{}'.format(
        patience,
        lr,
        train_epochs
    )
    path = os.path.join(checkpoint_path, setting)
    if not os.path.exists(path):
        os.makedirs(path)
    if train:
        print("Starting training...")
        trainer.train(
            patience=patience,
            verbose=True,
            learning_rate=lr,
            train_epochs=train_epochs,
            checkpoint_path=path
        )
        print("Training finished.")
    print("Starting testing...")
    trainer.predict(
        checkpoint_path=path,
        load= not train
    )



if __name__ == "__main__":
    main(train=False)