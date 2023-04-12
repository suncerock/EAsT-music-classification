from typing import Literal

import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-9

class KDLoss(nn.Module):
    def __init__(
        self,
        mode : Literal['soft', 'hard'],
        tau : int = 2.0
    ) -> None:
        super().__init__()

        self.mode = mode
        self.tau = tau

    def forward(self, output_dict_s, output_dict_t, mask):
        logits_s = output_dict_s['logits']
        logits_t = output_dict_t['logits']

        if self.mode == 'soft':
            scores_s = output_dict_s['scores'] / self.tau
            scores_t = output_dict_t['scores'] / self.tau
            logits_s = scores_s.sigmoid()
            logits_t = scores_t.sigmoid()
            # KD for binary classification
            loss = logits_t * (scores_t - scores_s) + ((1 - logits_t + EPSILON) / (1 - logits_s + EPSILON)).log()
            loss = loss[mask].mean() * self.tau ** 2
            
        else:
            loss = F.binary_cross_entropy(logits_s, logits_t.round(), reduce="none")
            loss = loss[mask].mean()
        return loss



