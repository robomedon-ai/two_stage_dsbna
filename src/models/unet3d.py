"""
3D U-Net arhitektura za segmentaciju prostate na MR slikama.

Referenca: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation
from Sparse Annotation", MICCAI 2016.

Koristi 3D konvolucije za obradu čitavih volumena,
čime se u potpunosti iskorištava prostorni kontekst u svim osima.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Dvostruki 3D konvolucijski blok: Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net za binarnu segmentaciju volumena.

    Ulaz: (B, 1, D, H, W) volumetrijski MR podatak
    Izlaz: (B, 1, D, H, W) logiti
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Enkoder
        self.enc1 = ConvBlock3D(in_channels, f)
        self.enc2 = ConvBlock3D(f, f * 2)
        self.enc3 = ConvBlock3D(f * 2, f * 4)
        self.enc4 = ConvBlock3D(f * 4, f * 8)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(f * 8, f * 16)

        # Dekoder
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

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
        d4 = self.up4(b)
        d4 = self._pad_and_cat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    @staticmethod
    def _pad_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Usklađuje dimenzije i spaja (concat) upsampled i skip tensor.
        Potrebno jer 3D volumeni mogu imati neparne dimenzije.
        """
        diff_d = skip.size(2) - upsampled.size(2)
        diff_h = skip.size(3) - upsampled.size(3)
        diff_w = skip.size(4) - upsampled.size(4)

        upsampled = nn.functional.pad(
            upsampled,
            [diff_w // 2, diff_w - diff_w // 2,
             diff_h // 2, diff_h - diff_h // 2,
             diff_d // 2, diff_d - diff_d // 2]
        )
        return torch.cat([upsampled, skip], dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
