from typing import Literal, Dict

import torch.nn as nn
import torch.nn.functional as F

from .feature_space_reg import FeatureSpaceRegularizationLoss
from .kd import KDLoss

class CombineDistRegLoss(nn.Module):
    def __init__(
        self,
        feature_mode,
        feature_stages_args: Dict[str, Dict],
        label_mode : Literal['soft', 'hard'],
        label_tau : int = 2.0,
        kd_weight : float = 0.3
    ) -> None:
        super().__init__()

        self.feature_loss_fn = FeatureSpaceRegularizationLoss(feature_mode, feature_stages_args)
        self.kd_loss_fn = KDLoss(label_mode, label_tau)
        self.kd_weight = kd_weight

    def forward(self, output_dict_s, output_dict_t, mask):
        feature_loss = self.feature_loss_fn(output_dict_s, output_dict_t, mask)
        kd_loss = self.kd_loss_fn(output_dict_s, output_dict_t, mask)
        return kd_loss * self.kd_weight + feature_loss * (1 - self.kd_weight)
