"""
3D Residual U-Net (ResUNet) arhitektura za segmentaciju prostate na MR slikama.

Referenca: Zhang et al., "Road Extraction by Deep Residual U-Net", IEEE GRSL 2018.
(3D varijanta)

3D verzija ResUNeta s rezidualnim konekcijama unutar svakog bloka za
volumetrijsku segmentaciju.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock3D(nn.Module):
    """
    3D rezidualni konvolucijski blok.

    Conv3D -> BN -> ReLU -> Conv3D -> BN + rezidualna konekcija -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_block(x) + self.shortcut(x))


class ResUNet3D(nn.Module):
    """
    3D Residual U-Net za binarnu volumetrijsku segmentaciju.

    Ulaz: (B, C, D, H, W)
    Izlaz: (B, 1, D, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Enkoder
        self.enc1 = ResConvBlock3D(in_channels, f)
        self.enc2 = ResConvBlock3D(f, f * 2)
        self.enc3 = ResConvBlock3D(f * 2, f * 4)
        self.enc4 = ResConvBlock3D(f * 4, f * 8)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResConvBlock3D(f * 8, f * 16)

        # Dekoder
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ResConvBlock3D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ResConvBlock3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ResConvBlock3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock3D(f * 2, f)

        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enkoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Dekoder sa skip konekcijama
        d4 = self._pad_and_cat(self.up4(b), e4)
        d4 = self.dec4(d4)

        d3 = self._pad_and_cat(self.up3(d4), e3)
        d3 = self.dec3(d3)

        d2 = self._pad_and_cat(self.up2(d3), e2)
        d2 = self.dec2(d2)

        d1 = self._pad_and_cat(self.up1(d2), e1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    @staticmethod
    def _pad_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff_d = skip.size(2) - upsampled.size(2)
        diff_h = skip.size(3) - upsampled.size(3)
        diff_w = skip.size(4) - upsampled.size(4)
        upsampled = F.pad(upsampled, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ])
        return torch.cat([upsampled, skip], dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
