#type: ignore

from tabnanny import check
import time
import duckdb as ddb
from numpy import c_
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages

from forecasting.autoformer.autoformer import Autoformer
from forecasting.autoformer.trainer import Trainer
from torch.utils import data
from forecasting.autoformer.data_loader import data_splitter
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
PATH = os.getenv("DATA_PATH")
WINDOW_SIZE = 24*7*2 # 2 weeks
HORIZON = 24*7 # 1 week
BATCH_SIZE = 32
LABEL_LEN = WINDOW_SIZE * 4 // 4

#EXOG_COLS = ['temp_max', 'temp_min', 'temp_media']
EXOG_COLS = ['temperature']


def main(
    train: bool = True
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

    query = f"""
    select e.departamento, e.dia, e.hora, agg_valor, (temperature + 15) / 65 as temperature
    from (
        select departamento, dia, hora, SUM(valor) as agg_valor
        from read_parquet('{PATH}')
        where departamento='MONTEVIDEO'
        group by departamento, dia, hora
    ) e inner join temperatura_montevideo t on e.dia=t.dia and e.hora=t.hora
    order by e.departamento, e.dia, e.hora
    """   
    con = ddb.connect(database=os.getenv("DB_PATH"))
    ts_agg_departamento = con.execute(query).fetchdf()
    print(f"Cantidad de registros totales en todos los departamentos agregados: {len(ts_agg_departamento)}")
    con.close()
    print ("Creating datasets...")
    montevideo_data = ts_agg_departamento[ts_agg_departamento["departamento"] == "MONTEVIDEO"]
    # Train & Test DataLoader
    
    all_dataset, train_dataset, val_dataset, test_dataset = data_splitter(
        df=montevideo_data,
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
    pred_len = HORIZON
    model = Autoformer(
        seq_len=seq_len,
        label_len=LABEL_LEN,
        pred_len=pred_len,
        c_out=1,
        enc_in=1,
        dec_in=1,
        e_layers=3,
        d_layers=2,
        moving_avg=167,
        factor=2,
        d_mark=6  # 4 time features (month, day, weekday, hour) + 1 temperature col + 1 holiday col
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

    checkpoint_path = Path("checkpoints")
    patience = 50
    lr = 0.0001
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
    plots_path = Path('results') / path / f'graficas.pdf'
    plots_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(plots_path) as pdf:
        trainer.predict_series(
            pdf=pdf,
            checkpoint_path=path,
            rolling_step=0,
            load= not train
        )
        trainer.predict_series(
            pdf=pdf,
            checkpoint_path=path,
            rolling_step=24 * 1,
            load= not train
        )
        trainer.predict_windows(
            pdf=pdf,
            checkpoint_path=path,
            rolling_step=0,
            load= not train
        )
        trainer.predict_windows(
            pdf=pdf,
            checkpoint_path=path,
            rolling_step=24 * 1,
            load= not train
        )


if __name__ == "__main__":
    main(train=True)