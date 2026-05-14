"""
MSDA-Net (Multi-Scale Dual Attention Network) - 3D varijanta.

3D verzija MSDA-Neta za volumetrijsku segmentaciju prostate na MR slikama.
Koristi 3D konvolucije i smanjene ASPP stope dilatacije za GPU efikasnost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEResConvBlock3D(nn.Module):
    """3D SE-Residual konvolucijski blok."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        r = max(1, out_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(out_channels, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, out_channels),
            nn.Sigmoid(),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_block(x)
        se_weights = self.se(out).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = out * se_weights
        return self.relu(out + residual)


class ASPP3D(nn.Module):
    """3D ASPP sa smanjenim stopama dilatacije za volumetrijske podatke."""

    def __init__(self, in_channels: int, out_channels: int,
                 rates: tuple = (3, 6, 9)):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=rate,
                          dilation=rate, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels),  # GroupNorm umjesto BN za spatial 1x1x1
            nn.ReLU(inplace=True),
        )
        num_branches = 2 + len(rates)
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        features = [self.conv1x1(x)]
        for conv in self.atrous_convs:
            features.append(conv(x))
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="trilinear", align_corners=False)
        features.append(gp)
        return self.project(torch.cat(features, dim=1))


class DualAttentionGate3D(nn.Module):
    """3D Dual Attention Gate (spatial + channel attention)."""

    def __init__(self, gate_channels: int, skip_channels: int,
                 inter_channels: int):
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
        r = max(1, skip_channels // 16)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(skip_channels, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, skip_channels),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        spatial_att = self.psi(self.relu(g + s))
        channel_att = self.channel_att(skip).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return skip * spatial_att * channel_att


class BoundaryRefinementModule3D(nn.Module):
    """3D modul za detekciju rubova segmentacije."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary_conv(x)


class MSDANet3D(nn.Module):
    """
    3D MSDA-Net za volumetrijsku binarnu segmentaciju.

    Ulaz: (B, C, D, H, W)
    Izlaz:
      - Treniranje: dict s ključevima "main", "aux4", "aux3", "aux2", "boundary"
      - Evaluacija: (B, 1, D, H, W) logiti
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 16):
        super().__init__()
        f = base_filters

        # Enkoder
        self.enc1 = SEResConvBlock3D(in_channels, f)
        self.enc2 = SEResConvBlock3D(f, f * 2)
        self.enc3 = SEResConvBlock3D(f * 2, f * 4)
        self.enc4 = SEResConvBlock3D(f * 4, f * 8)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck + ASPP
        self.bottleneck = SEResConvBlock3D(f * 8, f * 16)
        self.aspp = ASPP3D(f * 16, f * 16, rates=(3, 6, 9))

        # Dekoder
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.att4 = DualAttentionGate3D(f * 8, f * 8, f * 4)
        self.dec4 = SEResConvBlock3D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att3 = DualAttentionGate3D(f * 4, f * 4, f * 2)
        self.dec3 = SEResConvBlock3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att2 = DualAttentionGate3D(f * 2, f * 2, f)
        self.dec2 = SEResConvBlock3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.att1 = DualAttentionGate3D(f, f, max(1, f // 2))
        self.dec1 = SEResConvBlock3D(f * 2, f)

        # Izlazi
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)
        self.aux_head4 = nn.Conv3d(f * 8, out_channels, kernel_size=1)
        self.aux_head3 = nn.Conv3d(f * 4, out_channels, kernel_size=1)
        self.aux_head2 = nn.Conv3d(f * 2, out_channels, kernel_size=1)
        self.boundary_module = BoundaryRefinementModule3D(f)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        # Enkoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck + ASPP
        b = self.bottleneck(self.pool(e4))
        b = self.aspp(b)

        # Dekoder
        d4 = self._pad_to_match(self.up4(b), e4)
        e4_att = self.att4(gate=d4, skip=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self._pad_to_match(self.up3(d4), e3)
        e3_att = self.att3(gate=d3, skip=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self._pad_to_match(self.up2(d3), e2)
        e2_att = self.att2(gate=d2, skip=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self._pad_to_match(self.up1(d2), e1)
        e1_att = self.att1(gate=d1, skip=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        main_out = self.final_conv(d1)

        if self.training:
            aux4 = F.interpolate(self.aux_head4(d4), size=input_size,
                                 mode="trilinear", align_corners=False)
            aux3 = F.interpolate(self.aux_head3(d3), size=input_size,
                                 mode="trilinear", align_corners=False)
            aux2 = F.interpolate(self.aux_head2(d2), size=input_size,
                                 mode="trilinear", align_corners=False)
            boundary = self.boundary_module(d1)
            return {
                "main": main_out,
                "aux4": aux4,
                "aux3": aux3,
                "aux2": aux2,
                "boundary": boundary,
            }

        return main_out

    @staticmethod
    def _pad_to_match(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff_d = skip.size(2) - upsampled.size(2)
        diff_h = skip.size(3) - upsampled.size(3)
        diff_w = skip.size(4) - upsampled.size(4)
        return F.pad(upsampled, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
