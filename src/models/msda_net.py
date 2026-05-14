"""
MSDA-Net (Multi-Scale Dual Attention Network) - 2D varijanta.

Nova arhitektura za segmentaciju prostate na MR slikama koja kombinira:
  1. SE-Residual blokove (Squeeze-and-Excitation + rezidualne konekcije)
  2. ASPP modul (Atrous Spatial Pyramid Pooling) u bottlenecku
  3. Dual Attention Gate na skip konekcijama (channel + spatial attention)
  4. Deep Supervision (pomoćni izlazi na svakoj razini dekodera)
  5. Boundary Refinement Module (detekcija rubova za precizniju segmentaciju)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Modul A: SE-Residual Block (Squeeze-and-Excitation + Residual)
# =============================================================================

class SEResConvBlock2D(nn.Module):
    """
    Rezidualni konvolucijski blok s Squeeze-and-Excitation attention mehanizmom.

    Conv -> BN -> ReLU -> Conv -> BN -> SE -> + shortcut -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Squeeze-and-Excitation
        r = max(1, out_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, out_channels),
            nn.Sigmoid(),
        )

        # Shortcut konekcija
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_block(x)

        # SE: channel attention
        se_weights = self.se(out).unsqueeze(-1).unsqueeze(-1)
        out = out * se_weights

        return self.relu(out + residual)


# =============================================================================
# Modul B: ASPP (Atrous Spatial Pyramid Pooling)
# =============================================================================

class ASPP2D(nn.Module):
    """
    Atrous Spatial Pyramid Pooling za multi-scale kontekst u bottlenecku.

    Paralelne konvolucije s različitim stopama dilatacije + globalni pooling,
    zatim konkatenacija i projekcija.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 rates: tuple = (6, 12, 18)):
        super().__init__()

        # 1x1 konvolucija
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Dilatirane konvolucije
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate,
                          dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

        # Globalni Average Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

        # Projekcija (5 grana: 1x1, 3 dilatirane, GAP)
        num_branches = 2 + len(rates)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]

        features = [self.conv1x1(x)]
        for atrous_conv in self.atrous_convs:
            features.append(atrous_conv(x))

        # Globalni pooling + upsample natrag
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)
        features.append(gp)

        return self.project(torch.cat(features, dim=1))


# =============================================================================
# Modul C: Dual Attention Gate (Channel + Spatial Attention)
# =============================================================================

class DualAttentionGate2D(nn.Module):
    """
    Dual Attention Gate koji kombinira spatial i channel attention na skip
    konekcijama.

    Spatial attention: gating signal iz dekodera kontrolira koje prostorne
    lokacije iz skip konekcije su relevantne.

    Channel attention: SE-style mehanizam koji rekalibrira kanale skip
    konekcije.
    """

    def __init__(self, gate_channels: int, skip_channels: int,
                 inter_channels: int):
        super().__init__()

        # Spatial attention
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

        # Channel attention (SE-style)
        r = max(1, skip_channels // 16)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(skip_channels, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, skip_channels),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Spatial attention
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        spatial_att = self.psi(self.relu(g + s))

        # Channel attention
        channel_att = self.channel_att(skip).unsqueeze(-1).unsqueeze(-1)

        return skip * spatial_att * channel_att


# =============================================================================
# Modul E: Boundary Refinement Module
# =============================================================================

class BoundaryRefinementModule2D(nn.Module):
    """
    Modul za poboljšanje detekcije rubova segmentacije.

    Uzima značajke iz zadnjeg sloja dekodera i generira boundary mapu.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary_conv(x)


# =============================================================================
# Kompletna MSDA-Net 2D arhitektura
# =============================================================================

class MSDANet2D(nn.Module):
    """
    MSDA-Net (Multi-Scale Dual Attention Network) za 2D binarnu segmentaciju.

    Kombinira SE-Residual blokove, ASPP bottleneck, Dual Attention Gates,
    Deep Supervision i Boundary Refinement za state-of-the-art segmentaciju.

    Ulaz: (B, C, H, W)
    Izlaz:
      - Treniranje: dict s ključevima "main", "aux4", "aux3", "aux2", "boundary"
      - Evaluacija: (B, 1, H, W) logiti
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Enkoder (SE-Residual blokovi)
        self.enc1 = SEResConvBlock2D(in_channels, f)
        self.enc2 = SEResConvBlock2D(f, f * 2)
        self.enc3 = SEResConvBlock2D(f * 2, f * 4)
        self.enc4 = SEResConvBlock2D(f * 4, f * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck s ASPP
        self.bottleneck = SEResConvBlock2D(f * 8, f * 16)
        self.aspp = ASPP2D(f * 16, f * 16)

        # Dekoder s Dual Attention Gates
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.att4 = DualAttentionGate2D(f * 8, f * 8, f * 4)
        self.dec4 = SEResConvBlock2D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att3 = DualAttentionGate2D(f * 4, f * 4, f * 2)
        self.dec3 = SEResConvBlock2D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att2 = DualAttentionGate2D(f * 2, f * 2, f)
        self.dec2 = SEResConvBlock2D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.att1 = DualAttentionGate2D(f, f, f // 2)
        self.dec1 = SEResConvBlock2D(f * 2, f)

        # Glavni izlaz
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # Deep Supervision pomoćni izlazi
        self.aux_head4 = nn.Conv2d(f * 8, out_channels, kernel_size=1)
        self.aux_head3 = nn.Conv2d(f * 4, out_channels, kernel_size=1)
        self.aux_head2 = nn.Conv2d(f * 2, out_channels, kernel_size=1)

        # Boundary Refinement Module
        self.boundary_module = BoundaryRefinementModule2D(f)

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

        # Dekoder s Dual Attention Gates
        d4 = self.up4(b)
        e4_att = self.att4(gate=d4, skip=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(gate=d3, skip=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(gate=d2, skip=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(gate=d1, skip=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        # Glavni izlaz
        main_out = self.final_conv(d1)

        if self.training:
            # Deep Supervision: upsample pomoćnih izlaza na originalnu rezoluciju
            aux4 = F.interpolate(self.aux_head4(d4), size=input_size,
                                 mode="bilinear", align_corners=False)
            aux3 = F.interpolate(self.aux_head3(d3), size=input_size,
                                 mode="bilinear", align_corners=False)
            aux2 = F.interpolate(self.aux_head2(d2), size=input_size,
                                 mode="bilinear", align_corners=False)

            # Boundary prediction
            boundary = self.boundary_module(d1)

            return {
                "main": main_out,
                "aux4": aux4,
                "aux3": aux3,
                "aux2": aux2,
                "boundary": boundary,
            }

        return main_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
