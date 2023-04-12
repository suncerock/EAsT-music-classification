import torch
import torch.nn as nn


class PretrainedFeatureClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=None,
        dropout=0.0,
        classifier='linear',
    ) -> None:
        super().__init__()
        
        if classifier == 'linear':
            self.proj = nn.Linear(input_dim, num_classes)
        elif classifier == 'mlp':
            self.proj == nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            raise NotImplementedError


    def forward(self, x):
        output = self.proj(x.mean(dim=1))
        return dict(
            logits=torch.sigmoid(output),
            pred_logits=torch.sigmoid(output),
            scores=output,
            pred_scores=output,
            feature=x
            )
