"""
autoencoder.py
==============
Stage 2 Negative-Only Anomaly Detector Architectures:
1. SNNAutoencoder: Spiking Convolutional Autoencoder with LIF dynamics unrolled over T=4 steps.
2. CNNAutoencoder: Continuous Single-Step (T=1) Convolutional Autoencoder with SiLU.
3. ONNXExportableSNN: Export wrapper for deterministic ONNX tracing.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNAutoencoder(nn.Module):
    """Spiking Neural Network (SNN) Autoencoder for negative-only baseline reconstruction.

    Trained exclusively on healthy concrete surface baselines. When a defective region
    (crack, spalling, corrosion) is presented, the bottleneck fails to reconstruct it,
    yielding high pixel-wise reconstruction error in the anomaly heatmap.
    """

    def __init__(self, in_channels: int = 3, beta: float = 0.85, num_steps: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.beta = beta
        self.num_steps = num_steps

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)

        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)

        # Bottleneck Latent Representation
        self.bottleneck_conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn_bot = nn.BatchNorm2d(64)
        self.lif_bottleneck = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)

        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.lif4 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)

        self.out_conv = nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward_snn(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        assert x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0, "Spatial dimensions must be divisible by 4"

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_bot = self.lif_bottleneck.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        rec_acc = 0.0
        spikes_history = []

        for t in range(self.num_steps):
            c1 = self.bn1(self.enc_conv1(x))
            spk1, mem1 = self.lif1(c1, mem1)

            c2 = self.bn2(self.enc_conv2(spk1))
            spk2, mem2 = self.lif2(c2, mem2)

            c_bot = self.bn_bot(self.bottleneck_conv(spk2))
            spk_bot, mem_bot = self.lif_bottleneck(c_bot, mem_bot)

            d1 = self.bn3(self.dec_conv1(spk_bot))
            spk3, mem3 = self.lif3(d1, mem3)

            d2 = self.bn4(self.dec_conv2(spk3))
            spk4, mem4 = self.lif4(d2, mem4)

            out = self.sigmoid(self.out_conv(spk4))
            rec_acc = rec_acc + out
            spikes_history.append(spk_bot)

        reconstruction = rec_acc / float(self.num_steps)
        return reconstruction, spikes_history

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, _ = self.forward_snn(x)
        return reconstruction


class CNNAutoencoder(nn.Module):
    """Continuous Convolutional Autoencoder for high-FPS edge anomaly detection (T=1)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0, "Spatial dimensions must be divisible by 4"
        latent = self.encoder(x)
        return self.decoder(latent)


class ONNXExportableSNN(nn.Module):
    """Wrapper for clean ONNX tracing."""

    def __init__(self, base_snn: nn.Module):
        super().__init__()
        self.base_snn = base_snn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_snn(x)
