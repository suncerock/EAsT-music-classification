from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelDistillationLoss(nn.Module):
    def __init__(
        self,
        mode : Literal['soft', 'hard'],
        tau : int = 2.0
    ) -> None:
        super().__init__()

        self.mode = mode
        self.tau = tau

    def forward(self, output_dict_s, output_dict_t):
        logits_s = output_dict_s['dist_logits']
        logits_t = output_dict_t['logits']

        if self.mode == 'soft':
            scores_s = output_dict_s['dist_scores'] / self.tau
            scores_t = output_dict_t['scores'] / self.tau
            logits_s = scores_s.sigmoid()
            logits_t = scores_t.sigmoid()
            loss = logits_t * (scores_t - scores_s) + ((1 - logits_t + 1e-9) / (1 - logits_s + 1e-9)).log()
            loss = loss.mean() * self.tau ** 2
            # loss = loss * self.tau ** 2
            
        else:
            loss = F.binary_cross_entropy(logits_s, logits_t.round())
        return loss



