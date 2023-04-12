from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureSpaceRegularizationLoss(nn.Module):
    """
    Feature space regularization loss

    Inputs
    ----------
    output_dict_s: Dict
        student output dict, might use "output_1", "output_2", "output_3", (N, T, C)
    output_dict_t: Dict
        teacher output dict, might use "feature", (N, T, C)
    """
    def __init__(
        self,
        mode,
        stages_args: Dict[str, Dict],
    ) -> None:
        super().__init__()

        self.mode = mode
        self.stage_args = stages_args
        weights = {stage: args.get("weight", 0.0) for stage, args in stages_args.items()}
        self.weights = {stage: weight / sum(weights.values()) for stage, weight in weights.items()}
        self.student_expand = {stage: args.get("student_expand", -1) for stage, args in stages_args.items()}
        self.teacher_expand = {stage: args.get("teacher_expand", -1) for stage, args in stages_args.items()}

    def forward(self, output_dict_s, output_dict_t, mask):
        loss = None
        for stage in self.stage_args:
            if not self.weights[stage] > 0:
                continue
            output = output_dict_s["output_{}".format(stage)]
            target = output_dict_t["feature"]

            output = self.expand_feature_time(output, self.student_expand[stage])
            target = self.expand_feature_time(target, self.teacher_expand[stage])
            # print(output.shape, target.shape)
            if len(target.shape) == 3:
                length = min(target.shape[1], output.shape[1])
                output = output[:, :length]
                target = target[:, :length]
            assert output.shape[:-1] == target.shape[:-1]

            if loss is None:
                loss = self.weights[stage] * self.compute_reg_loss(output, target)
            else:
                loss += self.weights[stage] * self.compute_reg_loss(output, target)
        return loss

    def expand_feature_time(self, feature, expand):
        if expand == -1:
            return torch.mean(feature, dim=1)
        else:
            return torch.repeat_interleave(feature, expand, dim=1)

    def compute_distance_correlation(self, x, y):
        # x = F.normalize(x, dim=-1)  # N, T, C or N, C
        # y = F.normalize(y, dim=-1)  # N, T, C or N, C

        if len(x.shape) == 3:
            x = x.transpose(0, 1)  # T, N, C
            y = y.transpose(0, 1)  # T, N, C
 
        x = torch.sqrt(torch.sum(torch.square(x.unsqueeze(-3) - x.unsqueeze(-2)), dim = -1) + 1e-12)  # T, N, N or N, N
        y = torch.sqrt(torch.sum(torch.square(y.unsqueeze(-3) - y.unsqueeze(-2)), dim = -1) + 1e-12)  # T, N, N or N, N

        x = x - torch.mean(x, dim=-2, keepdims=True) - torch.mean(x, dim=-1, keepdims=True) + torch.mean(x, dim=(-2, -1), keepdims=True)
        y = y - torch.mean(y, dim=-2, keepdims=True) - torch.mean(y, dim=-1, keepdims=True) + torch.mean(y, dim=(-2, -1), keepdims=True)

        xy = torch.mean(x * y, dim=(-2, -1))
        xx = torch.mean(x * x, dim=(-2, -1))
        yy = torch.mean(y * y, dim=(-2, -1))

        correlation_r = xy / torch.sqrt(xx * yy + 1e-9)
        return (1 - correlation_r).mean()

    def compute_cosine_distance_difference(self, x, y):
        x = F.normalize(x, dim=-1)  # N, T, C or N, C
        y = F.normalize(y, dim=-1)  # N, T, C or N, C

        if len(x.shape) == 3:
            x = x.transpose(0, 1)  # T, N, C
            y = y.transpose(0, 1)  # T, N, C

        x = torch.matmul(x, x.transpose(-1, -2))  # T, N, N or N, N
        y = torch.matmul(y, y.transpose(-1, -2))  # T, N, N or N, N

        dist = torch.abs(x - y)
        return dist.mean()

    def compute_reg_loss(self, x, y):
        if self.mode == "distance_correlation":
            return self.compute_distance_correlation(x, y)
        elif self.mode == "cosine_distance_difference":
            return self.compute_cosine_distance_difference(x, y)
        else:
            raise NotImplementedError