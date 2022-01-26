from base64 import decode
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss

df = pd.read_csv("kelios_forecast_dataset.csv")
df.set_index("datetime", inplace=True)
df.index = pd.to_datetime(df.index)
data = df.copy()

max_prediction_length = 12
max_encoder_length = 24
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="nb_perturbations",
    group_ids=["month", "weekday"],
    min_encoder_length=0,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=["month", "weekday"],
    # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["temperature", "wind", "humidity", "nb_events"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "nb_perturbations",
        "nb_logs",
        "nb_covid_cases",
    ],
    target_normalizer=GroupNormalizer(
        groups=["weekday", "month"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)

best_tft = TemporalFusionTransformer.load_from_checkpoint(
    "checkpoints/checkpoint_e94.ckpt"
)


def predict_for_day(day, ax):
    pred_idx = df.loc[day]["time_idx"]
    #print(pred_idx)
    encoder_data = df[
        lambda x: (x.time_idx > pred_idx - max_encoder_length)
        & (x.time_idx <= pred_idx)
    ]
    encoder_data["date"] = encoder_data.index

    # select last known data point and create decoder data from it by repeating it and incrementing the month
    # in a real world dataset, we should not just forward fill the covariates but specify them to account
    # for changes in special days and prices (which you absolutely should do but we are too lazy here)
    last_data = df[lambda x: x.time_idx == pred_idx]
    decoder_data = pd.concat(
        [
            last_data.assign(date=lambda x: x.index + pd.Timedelta(i, unit="h"))
            for i in range(1, max_prediction_length + 1)
        ],
        ignore_index=True,
    )

    # add time index consistent with "data"
    decoder_data["time_idx"] = decoder_data["date"].dt.hour
    decoder_data["time_idx"] += pred_idx + 1 - decoder_data["time_idx"].min()
    # adjust additional time feature(s)
    decoder_data["month"] = decoder_data.date.dt.strftime("%b")

    # combine encoder and decoder data
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    new_raw_predictions, new_x = best_tft.predict(
        new_prediction_data, mode="raw", return_x=True
    )
    fig = best_tft.plot_prediction(
        new_x,
        new_raw_predictions,
        idx=0,
        show_future_observed=True,
        add_loss_to_title=False,
        plot_attention=False,
        ax=ax,
    )
    #print(decoder_data["time_idx"])
    return fig, new_raw_predictions
