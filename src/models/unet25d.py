"""
2.5D U-Net arhitektura za segmentaciju prostate na MR slikama.

Pristup 2.5D koristi N susjednih rezova kao ulazne kanale,
čime se dobiva prostorni kontekst duž Z-osi bez potrebe za
punim 3D konvolucijama. Arhitektura je identična 2D U-Netu,
osim što prima višekanalni ulaz.

Prednosti:
  - Više konteksta nego čisti 2D pristup
  - Značajno manja memorijska potrošnja od 3D pristupa
  - Ista brzina zaključivanja kao 2D U-Net
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


class UNet25D(nn.Module):
    """
    2.5D U-Net za binarnu segmentaciju.

    Ulaz: (B, N, H, W) gdje je N broj susjednih rezova
    Izlaz: (B, 1, H, W) logiti za središnji rez
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
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
