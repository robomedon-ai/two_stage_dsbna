"""
3D Attention U-Net arhitektura za segmentaciju prostate na MR slikama.

Referenca: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
MIDL 2018. (3D varijanta)

3D verzija Attention U-Neta s attention gate mehanizmom na skip konekcijama
za volumetrijsku segmentaciju.
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


class AttentionGate3D(nn.Module):
    """
    3D Attention Gate modul.

    Koristi gating signal (iz dekodera) i skip konekciju (iz enkodera)
    za generiranje 3D attention mape.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.relu(g + s)
        attention = self.psi(attention)
        return skip * attention


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net za binarnu volumetrijsku segmentaciju.

    Ulaz: (B, C, D, H, W)
    Izlaz: (B, 1, D, H, W) logiti (prije sigmoide)
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

        # Dekoder s attention gate-ovima
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate3D(gate_channels=f * 8, skip_channels=f * 8,
                                    inter_channels=f * 4)
        self.dec4 = ConvBlock3D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(gate_channels=f * 4, skip_channels=f * 4,
                                    inter_channels=f * 2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(gate_channels=f * 2, skip_channels=f * 2,
                                    inter_channels=f)
        self.dec2 = ConvBlock3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(gate_channels=f, skip_channels=f,
                                    inter_channels=f // 2)
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

        # Dekoder s attention gate-ovima
        d4 = self.up4(b)
        d4 = self._pad_to_match(d4, e4)
        e4_att = self.att4(gate=d4, skip=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_to_match(d3, e3)
        e3_att = self.att3(gate=d3, skip=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_to_match(d2, e2)
        e2_att = self.att2(gate=d2, skip=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_to_match(d1, e1)
        e1_att = self.att1(gate=d1, skip=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    @staticmethod
    def _pad_to_match(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff_d = skip.size(2) - upsampled.size(2)
        diff_h = skip.size(3) - upsampled.size(3)
        diff_w = skip.size(4) - upsampled.size(4)
        upsampled = F.pad(upsampled, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ])
        return upsampled

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
