"""
Residual U-Net (ResUNet) arhitektura za segmentaciju prostate na MR slikama.

Referenca: Zhang et al., "Road Extraction by Deep Residual U-Net", IEEE GRSL 2018.
           Diakogiannis et al., "ResUNet-a: A deep learning framework for semantic
           segmentation of remotely sensed data", ISPRS 2020.

Rezidualne konekcije unutar svakog bloka omogućuju treniranje dubljih mreža
bez degradacije gradijenta.
"""

import torch
import torch.nn as nn


class ResConvBlock2D(nn.Module):
    """
    Rezidualni konvolucijski blok.

    Conv -> BN -> ReLU -> Conv -> BN + rezidualna konekcija -> ReLU
    Ako se dimenzije ulaza i izlaza razlikuju, koristi se 1x1 konvolucija
    za prilagodbu dimenzija.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_block(x) + self.shortcut(x))


class ResUNet(nn.Module):
    """
    Residual U-Net za binarnu segmentaciju.

    Standardna U-Net arhitektura s rezidualnim konekcijama u svakom
    konvolucijskom bloku.

    Ulaz: (B, C, H, W)
    Izlaz: (B, 1, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Enkoder
        self.enc1 = ResConvBlock2D(in_channels, f)
        self.enc2 = ResConvBlock2D(f, f * 2)
        self.enc3 = ResConvBlock2D(f * 2, f * 4)
        self.enc4 = ResConvBlock2D(f * 4, f * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResConvBlock2D(f * 8, f * 16)

        # Dekoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ResConvBlock2D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ResConvBlock2D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ResConvBlock2D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock2D(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enkoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

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
