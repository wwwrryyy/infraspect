"""
model.py
========
Stage 3 Multi-Label Defect Classifier on 64x64 ROI Crops:
1. SpikingMobileNetV3Classifier: Depthwise separable convolutions with LIF neurons and rate-coding readout over T=4 steps.
2. CNNMobileNetV3Classifier: MobileNetV3-style depthwise separable continuous CNN (T=1).
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from data.datasets import DEFECT_CLASSES


class SpikingDepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 Conv + Pointwise 1x1 Conv with LIF neurons."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, beta: float = 0.85):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Depthwise
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        # Pointwise
        self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def init_mem(self):
        return self.dw_lif.init_leaky(), self.pw_lif.init_leaky()

    def forward(self, x: torch.Tensor, mem_dw: torch.Tensor, mem_pw: torch.Tensor):
        spk_dw, mem_dw = self.dw_lif(self.dw_bn(self.dw_conv(x)), mem_dw)
        spk_pw, mem_pw = self.pw_lif(self.pw_bn(self.pw_conv(spk_dw)), mem_pw)
        return spk_pw, mem_dw, mem_pw


class SpikingMobileNetV3Classifier(nn.Module):
    """Spiking MobileNetV3 Classifier on 64x64 crops with rate-coding readout."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = len(DEFECT_CLASSES),
        num_steps: int = 4,
        beta: float = 0.85,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.beta = beta

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Initial stem
        self.stem_conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        # Depthwise separable stages
        self.block1 = SpikingDepthwiseSeparableBlock(16, 32, stride=2, beta=beta)  # 32 -> 16
        self.block2 = SpikingDepthwiseSeparableBlock(32, 64, stride=2, beta=beta)  # 16 -> 8
        self.block3 = SpikingDepthwiseSeparableBlock(64, 128, stride=2, beta=beta) # 8 -> 4

        # Readout head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem_stem = self.stem_lif.init_leaky()
        mem_b1_dw, mem_b1_pw = self.block1.init_mem()
        mem_b2_dw, mem_b2_pw = self.block2.init_mem()
        mem_b3_dw, mem_b3_pw = self.block3.init_mem()

        spk_acc = 0.0

        for t in range(self.num_steps):
            stem_spk, mem_stem = self.stem_lif(self.stem_bn(self.stem_conv(x)), mem_stem)
            b1_spk, mem_b1_dw, mem_b1_pw = self.block1(stem_spk, mem_b1_dw, mem_b1_pw)
            b2_spk, mem_b2_dw, mem_b2_pw = self.block2(b1_spk, mem_b2_dw, mem_b2_pw)
            b3_spk, mem_b3_dw, mem_b3_pw = self.block3(b2_spk, mem_b3_dw, mem_b3_pw)

            pooled = self.pool(b3_spk).flatten(1)
            spk_acc = spk_acc + pooled

        rate_feature = spk_acc / float(self.num_steps)
        logits = self.classifier(rate_feature)
        return logits


class CNNMobileNetV3Classifier(nn.Module):
    """Continuous MobileNetV3-style Depthwise Separable Classifier (T=1)."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = len(DEFECT_CLASSES),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.net = nn.Sequential(
            # Stem (64 -> 32)
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),

            # Block 1 (32 -> 16)
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),

            # Block 2 (16 -> 8)
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),

            # Block 3 (8 -> 4)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
