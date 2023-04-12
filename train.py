import json

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import LatentSpaceDataModule
from models import TeacherStudentModel

def train(config):
    with open(config) as f:
        config = json.load(f)

    pl.seed_everything(config["seed"], workers=True)

    data_cfg = config["data"]
    model_cfg = config["model"]
    trainer_cfg = config["trainer"]

    datamodule = LatentSpaceDataModule(**data_cfg)
    model = TeacherStudentModel(model_cfg)

    callbacks = [
        ModelCheckpoint(**trainer_cfg["checkpoint"]),
        EarlyStopping(**trainer_cfg["early_stopping"])
    ]

    trainer = pl.Trainer(
        **trainer_cfg["args"],
        logger=TensorBoardLogger(**trainer_cfg["logger"]),
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)

if __name__ == '__main__':
    import fire

    fire.Fire(train)