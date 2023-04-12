import torch.nn as nn

class WaveformFeaturizer(nn.Identity):
    def __init__(self, training=True) -> None:
        super().__init__()