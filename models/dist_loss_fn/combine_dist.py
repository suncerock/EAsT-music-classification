from typing import Literal, Dict

import torch.nn as nn
import torch.nn.functional as F

from .feature_space_reg import FeatureSpaceRegularizationLoss
from .label_distillation import LabelDistillationLoss

class CombineDistillationLoss(nn.Module):
    def __init__(
        self,
        feature_mode,
        feature_stages_args: Dict[str, Dict],
        label_mode : Literal['soft', 'hard'],
        label_tau : int = 2.0
    ) -> None:
        super().__init__()

        self.feature_loss = FeatureSpaceRegularizationLoss(feature_mode, feature_stages_args)
        self.label_loss = LabelDistillationLoss(label_mode, label_tau)

    def forward(self, output_dict_s, output_dict_t):
        feature_loss = self.feature_loss(output_dict_s, output_dict_t)
        label_loss = self.label_loss(output_dict_s, output_dict_t)
        return feature_loss * 0.7 + label_loss * 0.3



