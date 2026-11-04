"""
model.py
========
Stage 1 Restoration / Despeckler Denoiser Architectures:
1. Spiking U-Net (SpikingUNet): Encoder-decoder with LIF neurons, skip connections,
   and un-reset continuous V_mem readout (reset_mechanism="none") for static temporal unrolling.
2. Continuous CNN U-Net (CNNUNet): High-throughput single-step (T=1) continuous ResUNet
   with SiLU activations and continuous skip connections.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingConvBlock(nn.Module):
    """Conv2d → BatchNorm → LIF neuron block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, beta: float = 0.85):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def init_mem(self):
        return self.lif.init_leaky()

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        cur = self.bn(self.conv(x))
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SpikingTransConvBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → LIF neuron block."""

    def __init__(self, in_ch: int, out_ch: int, beta: float = 0.85):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.tconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def init_mem(self):
        return self.lif.init_leaky()

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        cur = self.bn(self.tconv(x))
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SpikingUNet(nn.Module):
    """Spiking U-Net (S-UNet) for optical and speckle image restoration.

    Args:
        in_channels: Number of input channels (3 for RGB, 1 for grayscale).
        base_ch: Base channel count (doubled at each encoder stage).
        num_steps: Number of spiking timesteps T (default 4).
        beta: LIF membrane decay rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        num_steps: int = 4,
        beta: float = 0.85,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_steps = num_steps

        # --- Encoder ---
        self.enc1 = SpikingConvBlock(in_channels, base_ch, stride=2, beta=beta)
        self.enc2 = SpikingConvBlock(base_ch, base_ch * 2, stride=2, beta=beta)
        self.enc3 = SpikingConvBlock(base_ch * 2, base_ch * 4, stride=1, beta=beta)  # bottleneck

        # --- Decoder ---
        self.dec1 = SpikingTransConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2, beta=beta)
        self.dec2 = SpikingTransConvBlock(base_ch * 2 + base_ch, base_ch, beta=beta)

        # Final readout conv with reset_mechanism="none" so V_mem integrates continuously
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=1)
        self.out_lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            init_hidden=False,
            output=True,
            reset_mechanism="none",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass unrolled statically over num_steps."""
        assert x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0, "Input spatial dimensions must be divisible by 4"

        mem_e1 = self.enc1.init_mem()
        mem_e2 = self.enc2.init_mem()
        mem_e3 = self.enc3.init_mem()
        mem_d1 = self.dec1.init_mem()
        mem_d2 = self.dec2.init_mem()
        mem_out = self.out_lif.init_leaky()

        for t in range(self.num_steps):
            # Encoder
            spk_e1, mem_e1 = self.enc1(x, mem_e1)
            spk_e2, mem_e2 = self.enc2(spk_e1, mem_e2)
            spk_e3, mem_e3 = self.enc3(spk_e2, mem_e3)

            # Decoder with skip connections
            d1_in = torch.cat([spk_e3, spk_e2], dim=1)
            spk_d1, mem_d1 = self.dec1(d1_in, mem_d1)

            d2_in = torch.cat([spk_d1, spk_e1], dim=1)
            spk_d2, mem_d2 = self.dec2(d2_in, mem_d2)

            # Readout: accumulate V_mem without threshold reset
            cur_out = self.out_conv(spk_d2)
            spk_out, mem_out = self.out_lif(cur_out, mem_out)

        # Continuous pixel output in [0, 1]
        return torch.sigmoid(mem_out)


# ---------------------------------------------------------------------------
# Continuous CNN Baseline U-Net (T=1 Single-Step)
# ---------------------------------------------------------------------------

class CNNConvBlock(nn.Module):
    """Conv2d → BatchNorm → SiLU block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNNUNet(nn.Module):
    """Continuous CNN U-Net for real-time edge image restoration (T=1)."""

    def __init__(self, in_channels: int = 3, base_ch: int = 32):
        super().__init__()
        self.in_channels = in_channels

        # Encoder
        self.enc1 = CNNConvBlock(in_channels, base_ch, stride=2)
        self.enc2 = CNNConvBlock(base_ch, base_ch * 2, stride=2)
        self.enc3 = CNNConvBlock(base_ch * 2, base_ch * 4, stride=1)  # bottleneck

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4 + base_ch * 2, base_ch * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2 + base_ch, base_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0, "Input spatial dimensions must be divisible by 4"
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d1 = self.dec1(torch.cat([e3, e2], dim=1))
        d2 = self.dec2(torch.cat([d1, e1], dim=1))
        return self.out_conv(d2)
