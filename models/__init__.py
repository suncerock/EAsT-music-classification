from typing import Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .basemodels import get_base_model_and_pred_from
from .dist_loss_fn import get_dist_loss_fn
from .metrics import MultiLabelBinaryEval

class TeacherStudentModel(pl.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.backbone_cfg = configs["backbone"]
        self.optim_cfg = configs["optim"]

        self.student, self.student_pred_from = get_base_model_and_pred_from(self.backbone_cfg["student"])
        self.feature = self.backbone_cfg["feature"]

        if self.backbone_cfg["teacher"] is not None:
            self.teacher, self.teacher_pred_from = get_base_model_and_pred_from(self.backbone_cfg["teacher"]) 
            if not self.backbone_cfg["teacher_learning"]:
                for param in self.teacher.parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError
            self.dist_loss_weight = self.backbone_cfg["dist_loss_weight"]
            self.dist_loss_fn = get_dist_loss_fn(self.backbone_cfg["distillation"])
        else:
            self.teacher = None

        self.train_metrics = MultiLabelBinaryEval(num_classes=self.backbone_cfg["student"]["args"]["num_classes"])
        self.val_metrics = MultiLabelBinaryEval(num_classes=self.backbone_cfg["student"]["args"]["num_classes"])
        self.test_metrics = MultiLabelBinaryEval(num_classes=self.backbone_cfg["student"]["args"]["num_classes"])

    def training_step(self, batch, batch_idx) -> Any:
        loss_dict, logits = self.common_step(batch)

        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"])
        self.log_dict_prefix(loss_dict, "train")

        self.train_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

        return loss_dict["loss/total"]

    def validation_step(self, batch, batch_idx) -> Any:
        loss_dict, logits = self.common_step(batch)

        self.log_dict_prefix(loss_dict, "val")
        
        self.val_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

        return loss_dict["loss/total"]

    def test_step(self, batch, batch_idx):
        loss_dict, logits = self.common_step(batch)

        self.log_dict_prefix(loss_dict, "test")

        self.test_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

    def common_step(self, batch):
        y = batch['y']
        y_mask = batch['y_mask']
        
        loss_dict = dict()
        output_dict_s = self.submodel_forward(batch, self.student, self.student_pred_from)

        loss_pred = F.binary_cross_entropy(output_dict_s['pred_logits'], y, reduction='none')
        loss_pred = loss_pred[y_mask].mean()

        loss_dict['loss/pred'] = loss_pred

        if self.teacher is not None:
            output_dict_t = self.submodel_forward(batch, self.teacher, self.teacher_pred_from)
            loss_dist = self.dist_loss_fn(output_dict_s, output_dict_t)
            # loss_dist = loss_dist[y_mask].mean()
            loss_dict['loss/dist'] = loss_dist
            loss = loss_pred * (1 - self.dist_loss_weight) + loss_dist * self.dist_loss_weight
        else:
            loss = loss_pred
        loss_dict['loss/total'] = loss
        return loss_dict, output_dict_s['logits']

    def submodel_forward(self, batch, model, pred_from):
        x = batch["x"]
        feature = batch[self.feature]

        if pred_from == "x":
            return model(x)
        elif pred_from == "feature":
            return model(feature)
        else:
            raise NotImplementedError

    def on_train_epoch_start(self) -> None:
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        metric_dict = self.train_metrics.compute()
        if not isinstance(metric_dict, dict):
            metric_dict = dict(acc=metric_dict)
        self.log_dict_prefix(metric_dict, 'train')

    def on_validation_epoch_end(self) -> None:
        metric_dict = self.val_metrics.compute()
        if not isinstance(metric_dict, dict):
            metric_dict = dict(acc=metric_dict)
        self.log_dict_prefix(metric_dict, 'val')

    def on_test_epoch_end(self) -> None:
        metric_dict = self.test_metrics.compute()
        if not isinstance(metric_dict, dict):
            metric_dict = dict(acc=metric_dict)
        self.log_dict_prefix(metric_dict, 'test')

    def log_dict_prefix(self, d, prefix):
        for k, v in d.items():
            self.log("{}/{}".format(prefix, k), v)

    def configure_optimizers(self) -> Any:
        optimizer_cfg = self.optim_cfg["optimizer"]
        scheduler_cfg = self.optim_cfg["scheduler"]

        optimizer = torch.optim.__dict__.get(optimizer_cfg["name"])(self.parameters(), **optimizer_cfg["args"])
        scheduler = torch.optim.lr_scheduler.__dict__.get(scheduler_cfg["name"])(optimizer, **scheduler_cfg["args"])
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                monitor=scheduler_cfg["monitor"],
            ))