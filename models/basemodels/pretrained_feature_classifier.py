import torch
import torch.nn as nn


class PretrainedFeatureClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes
    ) -> None:
        super().__init__()
        
        self.proj = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        output = self.proj(x.mean(dim=1))
        return dict(
            scores=output,
            logits=torch.sigmoid(output),
            feature=x
            )
