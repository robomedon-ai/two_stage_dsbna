"""
Attention U-Net arhitektura za segmentaciju prostate na MR slikama.

Referenca: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
MIDL 2018.

Attention gate mehanizam na skip konekcijama omogućuje modelu da se fokusira
na relevantne regije i potisne irelevantne značajke.
"""

import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    """Dvostruki konvolucijski blok: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention Gate modul.

    Koristi gating signal (iz dekodera) i skip konekciju (iz enkodera)
    za generiranje attention mape koja naglašava relevantne značajke.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.relu(g + s)
        attention = self.psi(attention)
        return skip * attention


class AttentionUNet(nn.Module):
    """
    Attention U-Net za binarnu segmentaciju.

    Standardni U-Net s attention gate modulima na skip konekcijama.

    Ulaz: (B, C, H, W)
    Izlaz: (B, 1, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Enkoder
        self.enc1 = ConvBlock2D(in_channels, f)
        self.enc2 = ConvBlock2D(f, f * 2)
        self.enc3 = ConvBlock2D(f * 2, f * 4)
        self.enc4 = ConvBlock2D(f * 4, f * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock2D(f * 8, f * 16)

        # Dekoder s attention gate-ovima
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(gate_channels=f * 8, skip_channels=f * 8,
                                  inter_channels=f * 4)
        self.dec4 = ConvBlock2D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(gate_channels=f * 4, skip_channels=f * 4,
                                  inter_channels=f * 2)
        self.dec3 = ConvBlock2D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(gate_channels=f * 2, skip_channels=f * 2,
                                  inter_channels=f)
        self.dec2 = ConvBlock2D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.att1 = AttentionGate(gate_channels=f, skip_channels=f,
                                  inter_channels=f // 2)
        self.dec1 = ConvBlock2D(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enkoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Dekoder s attention gate-ovima
        d4 = self.up4(b)
        e4 = self.att4(gate=d4, skip=e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3 = self.att3(gate=d3, skip=e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2 = self.att2(gate=d2, skip=e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1 = self.att1(gate=d1, skip=e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
