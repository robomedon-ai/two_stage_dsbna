"""
2D U-Net arhitektura za segmentaciju prostate na MR slikama.

Referenca: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.
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


class Encoder2D(nn.Module):
    """Enkoder (kontraktivni put) s max poolingom."""

    def __init__(self, in_channels: int, base_filters: int):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock2D(in_channels, f)
        self.enc2 = ConvBlock2D(f, f * 2)
        self.enc3 = ConvBlock2D(f * 2, f * 4)
        self.enc4 = ConvBlock2D(f * 4, f * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4


class Decoder2D(nn.Module):
    """Dekoder (ekspanzivni put) s transponiranim konvolucijama i skip konekcijama."""

    def __init__(self, base_filters: int, out_channels: int):
        super().__init__()
        f = base_filters

        self.up4 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock2D(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(f * 4, f * 2)

        self.up2 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, e1, e2, e3, e4):
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.final_conv(d2)


class Bottleneck2D(nn.Module):
    """Bottleneck (most) između enkodera i dekodera."""

    def __init__(self, base_filters: int):
        super().__init__()
        f = base_filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock2D(f * 8, f * 8)
        self.up = nn.ConvTranspose2d(f * 8, f * 8, kernel_size=2, stride=2)

    def forward(self, e4: torch.Tensor) -> torch.Tensor:
        b = self.pool(e4)
        b = self.bottleneck(b)
        b = self.up(b)
        return b


class UNet2D(nn.Module):
    """
    2D U-Net za binarnu segmentaciju.

    Ulaz: (B, C, H, W) gdje je C=1 za jednokanalni MR rez
    Izlaz: (B, 1, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        self.encoder = Encoder2D(in_channels, f)

        # Bottleneck
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock2D(f * 8, f * 16)

        # Dekoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock2D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2D(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enkoder
        e1, e2, e3, e4 = self.encoder(x)

        # Bottleneck
        b = self.pool(e4)
        b = self.bottleneck(b)

        # Dekoder sa skip konekcijama
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
