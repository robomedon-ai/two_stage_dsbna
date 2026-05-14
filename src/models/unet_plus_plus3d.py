"""
3D U-Net++ (Nested U-Net) arhitektura za segmentaciju prostate na MR slikama.

Referenca: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image
Segmentation", DLMIA 2018. (3D varijanta)

3D verzija U-Net++ s gustim ugniježđenim skip pathovima za volumetrijsku
segmentaciju.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class UNetPlusPlus3D(nn.Module):
    """
    3D U-Net++ (Nested U-Net) za binarnu volumetrijsku segmentaciju.

    Ulaz: (B, C, D, H, W)
    Izlaz: (B, 1, D, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters
        filters = [f, f * 2, f * 4, f * 8, f * 16]

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Enkoder čvorovi (X^{i,0})
        self.conv0_0 = ConvBlock3D(in_channels, filters[0])
        self.conv1_0 = ConvBlock3D(filters[0], filters[1])
        self.conv2_0 = ConvBlock3D(filters[1], filters[2])
        self.conv3_0 = ConvBlock3D(filters[2], filters[3])
        self.conv4_0 = ConvBlock3D(filters[3], filters[4])

        # Ugniježđeni čvorovi
        # Razina 0
        self.up0_1 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv0_1 = ConvBlock3D(filters[0] * 2, filters[0])

        self.up0_2 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv0_2 = ConvBlock3D(filters[0] * 3, filters[0])

        self.up0_3 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv0_3 = ConvBlock3D(filters[0] * 4, filters[0])

        self.up0_4 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv0_4 = ConvBlock3D(filters[0] * 5, filters[0])

        # Razina 1
        self.up1_1 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=2, stride=2)
        self.conv1_1 = ConvBlock3D(filters[1] * 2, filters[1])

        self.up1_2 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=2, stride=2)
        self.conv1_2 = ConvBlock3D(filters[1] * 3, filters[1])

        self.up1_3 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=2, stride=2)
        self.conv1_3 = ConvBlock3D(filters[1] * 4, filters[1])

        # Razina 2
        self.up2_1 = nn.ConvTranspose3d(filters[3], filters[2], kernel_size=2, stride=2)
        self.conv2_1 = ConvBlock3D(filters[2] * 2, filters[2])

        self.up2_2 = nn.ConvTranspose3d(filters[3], filters[2], kernel_size=2, stride=2)
        self.conv2_2 = ConvBlock3D(filters[2] * 3, filters[2])

        # Razina 3
        self.up3_1 = nn.ConvTranspose3d(filters[4], filters[3], kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock3D(filters[3] * 2, filters[3])

        self.final_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enkoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Stupac 1
        x0_1 = self.conv0_1(self._up_and_cat(self.up0_1, x1_0, x0_0))
        x1_1 = self.conv1_1(self._up_and_cat(self.up1_1, x2_0, x1_0))
        x2_1 = self.conv2_1(self._up_and_cat(self.up2_1, x3_0, x2_0))
        x3_1 = self.conv3_1(self._up_and_cat(self.up3_1, x4_0, x3_0))

        # Stupac 2
        x0_2 = self.conv0_2(self._up_and_cat_multi(self.up0_2, x1_1, [x0_0, x0_1]))
        x1_2 = self.conv1_2(self._up_and_cat_multi(self.up1_2, x2_1, [x1_0, x1_1]))
        x2_2 = self.conv2_2(self._up_and_cat_multi(self.up2_2, x3_1, [x2_0, x2_1]))

        # Stupac 3
        x0_3 = self.conv0_3(self._up_and_cat_multi(self.up0_3, x1_2,
                                                     [x0_0, x0_1, x0_2]))
        x1_3 = self.conv1_3(self._up_and_cat_multi(self.up1_3, x2_2,
                                                     [x1_0, x1_1, x1_2]))

        # Stupac 4
        x0_4 = self.conv0_4(self._up_and_cat_multi(self.up0_4, x1_3,
                                                     [x0_0, x0_1, x0_2, x0_3]))

        return self.final_conv(x0_4)

    @staticmethod
    def _pad_to_match(upsampled: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_d = target.size(2) - upsampled.size(2)
        diff_h = target.size(3) - upsampled.size(3)
        diff_w = target.size(4) - upsampled.size(4)
        return F.pad(upsampled, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ])

    def _up_and_cat(self, up_layer, x_below, x_skip):
        up = up_layer(x_below)
        up = self._pad_to_match(up, x_skip)
        return torch.cat([x_skip, up], dim=1)

    def _up_and_cat_multi(self, up_layer, x_below, x_skips):
        up = up_layer(x_below)
        up = self._pad_to_match(up, x_skips[0])
        return torch.cat([*x_skips, up], dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
