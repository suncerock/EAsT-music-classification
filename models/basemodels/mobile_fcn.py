import torch
import torch.nn as nn

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class Conv_2d_DW(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, shape, stride=stride, padding=shape//2, groups=input_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.conv2 = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.mp(self.relu(self.bn2(self.conv2(out))))
        return out

    
class MobileFCN(nn.Module):
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self, n_mels=128, num_classes=50):
        super(MobileFCN, self).__init__()

        # FCN
        self.layer1 = Conv_2d(1, 64, pooling=(2,4))
        self.layer2 = Conv_2d(64, 128, pooling=(2,3))
        self.layer3 = Conv_2d_DW(128, 128, pooling=(2,2))
        if n_mels == 128:
            self.layer4 = Conv_2d_DW(128, 128, pooling=(4,2))
        else:
            self.layer4 = Conv_2d_DW(128, 128, pooling=(3,2))

        self.layer5 = Conv_2d_DW(128, 128, pooling=(4,2))
        self.layer6 = Conv_2d_DW(128, 256, pooling=1)
        self.layer7 = Conv_2d_DW(256, 256, pooling=1)
        self.layer8 = Conv_2d_DW(256, 256, pooling=1)

        # Dense
        self.dense = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(dim=1)

        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output_1 = x

        x = self.dropout(x)
        x = self.layer5(x)
        x = self.layer6(x)
        output_2 = x

        x = self.dropout(x)
        x = self.layer7(x)
        x = self.layer8(x)
        output_3 = x

        # Dense
        x = self.dropout(x)
        output = x.mean(dim=-1).mean(dim=-1)
        output = self.dense(output)
        # print(output_1.shape, output_2.shape, output_3.shape)
        return dict(
            logits=torch.sigmoid(output),
            pred_logits=torch.sigmoid(output),
            dist_logits=torch.sigmoid(output),
            scores=output,
            pred_scores=output,
            dist_scores=output,
            output_1=output_1.mean(dim=-2).transpose(-1, -2),
            output_2=output_2.mean(dim=-2).transpose(-1, -2),
            output_3=output_3.mean(dim=-2).transpose(-1, -2)
            )

if __name__ == "__main__":
    model = MobileFCN(num_classes=50)
    import numpy as np
    print(np.sum([np.prod(param.shape) for param in model.parameters()]))

    x = torch.randn(2, 128, 1457)
    model(x)