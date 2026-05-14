"""
U-Net++ (Nested U-Net) arhitektura za segmentaciju prostate na MR slikama.

Referenca: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image
Segmentation", DLMIA 2018.

Gusti ugniježđeni skip pathovi smanjuju semantički jaz između enkodera i dekodera
što poboljšava propagaciju značajki.
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


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) za binarnu segmentaciju.

    Koristi gusto ugniježđene skip konekcije između enkodera i dekodera.
    Intermediate čvorovi (X^{i,j}) primaju ulaze od svih prethodnih čvorova
    u istom redu i upsampliranog izlaza iz reda ispod.

    Ulaz: (B, C, H, W)
    Izlaz: (B, 1, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters
        filters = [f, f * 2, f * 4, f * 8, f * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Enkoder čvorovi (X^{i,0})
        self.conv0_0 = ConvBlock2D(in_channels, filters[0])
        self.conv1_0 = ConvBlock2D(filters[0], filters[1])
        self.conv2_0 = ConvBlock2D(filters[1], filters[2])
        self.conv3_0 = ConvBlock2D(filters[2], filters[3])
        self.conv4_0 = ConvBlock2D(filters[3], filters[4])

        # Ugniježđeni čvorovi (X^{i,j})
        # Razina 0 (najplića)
        self.up0_1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv0_1 = ConvBlock2D(filters[0] * 2, filters[0])

        self.up0_2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv0_2 = ConvBlock2D(filters[0] * 3, filters[0])

        self.up0_3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv0_3 = ConvBlock2D(filters[0] * 4, filters[0])

        self.up0_4 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv0_4 = ConvBlock2D(filters[0] * 5, filters[0])

        # Razina 1
        self.up1_1 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.conv1_1 = ConvBlock2D(filters[1] * 2, filters[1])

        self.up1_2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.conv1_2 = ConvBlock2D(filters[1] * 3, filters[1])

        self.up1_3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.conv1_3 = ConvBlock2D(filters[1] * 4, filters[1])

        # Razina 2
        self.up2_1 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.conv2_1 = ConvBlock2D(filters[2] * 2, filters[2])

        self.up2_2 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.conv2_2 = ConvBlock2D(filters[2] * 3, filters[2])

        # Razina 3
        self.up3_1 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock2D(filters[3] * 2, filters[3])

        # Završni sloj
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enkoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Ugniježđeni dekoder - stupac 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up0_1(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1_1(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], dim=1))

        # Ugniježđeni dekoder - stupac 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up0_2(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1_2(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2_2(x3_1)], dim=1))

        # Ugniježđeni dekoder - stupac 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up0_3(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up1_3(x2_2)], dim=1))

        # Ugniježđeni dekoder - stupac 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3,
                                        self.up0_4(x1_3)], dim=1))

        return self.final_conv(x0_4)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
