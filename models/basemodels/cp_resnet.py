# coding: utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


layer_index_total = 0


def initialize_weights_fixup(module):
    # source: https://github.com/ajbrock/BoilerPlate/blob/master/Models/fixup.py
    if isinstance(module, BasicBlock):
        # He init, rescaled by Fixup multiplier
        b = module
        n = b.conv1.kernel_size[0] * b.conv1.kernel_size[1] * b.conv1.out_channels
        #print(b.layer_index, math.sqrt(2. / n), layer_index_total ** (-0.5))
        b.conv1.weight.data.normal_(0, (layer_index_total ** (-0.5)) * math.sqrt(2. / n))
        b.conv2.weight.data.zero_()
        if b.shortcut._modules.get('conv') is not None:
            convShortcut = b.shortcut._modules.get('conv')
            n = convShortcut.kernel_size[0] * convShortcut.kernel_size[1] * convShortcut.out_channels
            convShortcut.weight.data.normal_(0, math.sqrt(2. / n))
    if isinstance(module, nn.Conv2d):
        pass
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


def calc_padding(kernal):
    try:
        return kernal // 3
    except TypeError:
        return [k // 3 for k in kernal]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k1=3, k2=3):
        super(BasicBlock, self).__init__()
        global layer_index_total
        self.layer_index = layer_index_total
        layer_index_total = layer_index_total + 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=k1,
            stride=stride,  # downsample with first conv
            padding=calc_padding(k1),
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=k2,
            stride=1,
            padding=calc_padding(k2),
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y



class CPResNet(nn.Module):
    def __init__(
        self,
        rho,
        in_channel,
        base_channels=128,
        num_classes=20,
    ):

        super(CPResNet, self).__init__()


        self.in_c = nn.Sequential(
            nn.Conv2d(in_channel, base_channels, 5, 2, 2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )

        extra_kernal_rf = rho - 7

        self.stage1 = self._make_stage(
            base_channels, base_channels, 4,
                maxpool={1:2, 2:2, 4:2},
                k1s=(
                    3,
                    3 - (-extra_kernal_rf > 6) * 2,
                    3 - (-extra_kernal_rf > 4) * 2,
                    3 - (-extra_kernal_rf > 2) * 2),
                k2s=(
                    1,
                    3 - (-extra_kernal_rf > 5) * 2,
                    3 - (-extra_kernal_rf > 3) * 2,
                    3 - (-extra_kernal_rf > 1) * 2))

        self.stage2 = self._make_stage(
            base_channels, base_channels * 2, 4,
                k1s=(
                    3 - (-extra_kernal_rf > 0) * 2,
                    1 + (extra_kernal_rf > 1) * 2,
                    1 + (extra_kernal_rf > 3) * 2,
                    1 + (extra_kernal_rf > 5) * 2),
                k2s=(1,
                    3 - (-extra_kernal_rf > 5) * 2,
                    3 - (-extra_kernal_rf > 3) * 2,
                    3 - (-extra_kernal_rf > 1) * 2))

        self.stage3 = self._make_stage(
            base_channels * 2, base_channels * 4, 4,
                k1s=(
                    1 + (extra_kernal_rf > 7) * 2,
                    1 + (extra_kernal_rf > 9) * 2,
                    1 + (extra_kernal_rf > 11) * 2,
                    1 + (extra_kernal_rf > 13) * 2),
                k2s=(
                    1 + (extra_kernal_rf > 8) * 2,
                    1 + (extra_kernal_rf > 10) * 2,
                    1 + (extra_kernal_rf > 12) * 2,
                    1 + (extra_kernal_rf > 14) * 2))

        self.feed_forward = nn.Linear(base_channels * 4, num_classes)

        # initialize weights
        self.apply(initialize_weights)
        self.apply(initialize_weights_fixup)

    def _make_stage(self, in_channels, out_channels, n_blocks, maxpool=set(), k1s=[3, 3, 3, 3, 3, 3],
                    k2s=[3, 3, 3, 3, 3, 3]):
        stage = nn.Sequential()

        for index in range(n_blocks):
            stage.add_module('block{}'.format(index + 1),
                BasicBlock(in_channels, out_channels, stride=1, k1=k1s[index], k2=k2s[index]))

            in_channels = out_channels
            if index + 1 in maxpool:
                stage.add_module("maxpool{}".format(index + 1), nn.MaxPool2d(maxpool[index + 1]))
        return stage

    def forward_conv(self, x):
        x = self.in_c(x)
        output_1 = self.stage1(x)
        output_2 = self.stage2(output_1)
        output_3 = self.stage3(output_2)
        return output_1, output_2, output_3

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        output_1, output_2, output_3 = self.forward_conv(x)
        output = output_3.mean(dim=-1).mean(dim=-1)
        output = self.feed_forward(output)
        return dict(
            logits=torch.sigmoid(output),
            scores=output,
            output_1=output_1.mean(dim=-2).transpose(-1, -2),
            output_2=output_2.mean(dim=-2).transpose(-1, -2),
            output_3=output_3.mean(dim=-2).transpose(-1, -2)
            )
