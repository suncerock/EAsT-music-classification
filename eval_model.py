import json

import pytorch_lightning as pl

from data import LatentSpaceDataModule
from models import TeacherStudentModel


def eval_model(config, ckpt_path, manifest_path):
    with open(config) as f:
        config = json.load(f)

    data_cfg = config["data"]
    data_cfg["test_manifest_path"] = manifest_path
    datamodule = LatentSpaceDataModule(**data_cfg)

    model = TeacherStudentModel.load_from_checkpoint(ckpt_path)

    evaluator = pl.Trainer(accelerator='gpu')
    evaluator.test(model, datamodule=datamodule)

if __name__ == '__main__':
    import fire

    fire.Fire(eval_model)