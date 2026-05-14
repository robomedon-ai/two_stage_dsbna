"""
DSBANet (Deep Supervision Boundary-Aware Attention Network) - 3D varijanta.

3D verzija DSBANeta za volumetrijsku segmentaciju prostate na MR slikama.
Uključuje:
  1. Inflated ResNet50 enkoder (pretrained 2D težine proširene na 3D)
  2. ASPP3D u bottlenecku
  3. MSAF3D (Multi-Scale Attention Fusion) na skip konekcijama
  4. FFM3D (Feature Fusion Module)
  5. Deep Supervision
  6. Boundary Refinement Module

Svaka komponenta se može uključiti/isključiti za ablacijsku studiju.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import torchvision.models as models


# =============================================================================
# Bazni blokovi
# =============================================================================

class ConvBlock3D(nn.Module):
    """Standardni dvostruki 3D konvolucijski blok."""

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


# =============================================================================
# Inflated ResNet50 Encoder (2D pretrained → 3D)
# =============================================================================

def _inflate_conv(conv2d: nn.Conv2d, time_dim: int = 3) -> nn.Conv3d:
    """Pretvara 2D conv u 3D conv inflacijom kernela duž vremenske/z osi."""
    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_size=(time_dim, *conv2d.kernel_size),
        stride=(1, *conv2d.stride),
        padding=(time_dim // 2, *conv2d.padding),
        dilation=(1, *conv2d.dilation),
        groups=conv2d.groups,
        bias=conv2d.bias is not None,
    )
    # Kopiraj težine: prosječni kernel duž z-osi
    with torch.no_grad():
        weight_2d = conv2d.weight.data  # (out, in, kH, kW)
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1) / time_dim
        conv3d.weight.copy_(weight_3d)
        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias.data)
    return conv3d


def _inflate_bn(bn2d: nn.BatchNorm2d) -> nn.BatchNorm3d:
    """Pretvara 2D BatchNorm u 3D BatchNorm."""
    bn3d = nn.BatchNorm3d(bn2d.num_features)
    bn3d.weight.data.copy_(bn2d.weight.data)
    bn3d.bias.data.copy_(bn2d.bias.data)
    bn3d.running_mean.copy_(bn2d.running_mean)
    bn3d.running_var.copy_(bn2d.running_var)
    bn3d.eps = bn2d.eps
    bn3d.momentum = bn2d.momentum
    return bn3d


def _inflate_bottleneck(block2d, stride_3d=(1, 1, 1)):
    """Inflacija jednog ResNet Bottleneck bloka iz 2D u 3D."""
    from torchvision.models.resnet import Bottleneck

    # Kopiraj conv1 (1x1 -> 1x1x1)
    conv1_3d = nn.Conv3d(
        block2d.conv1.in_channels, block2d.conv1.out_channels,
        kernel_size=1, bias=False
    )
    with torch.no_grad():
        conv1_3d.weight.copy_(block2d.conv1.weight.unsqueeze(2))
    bn1_3d = _inflate_bn(block2d.bn1)

    # conv2 (3x3 -> 3x3x3)
    conv2_3d = _inflate_conv(block2d.conv2, time_dim=3)
    # Prilagodi stride za z-os
    if block2d.conv2.stride[0] > 1:
        conv2_3d.stride = (stride_3d[0], block2d.conv2.stride[0],
                           block2d.conv2.stride[1])
    bn2_3d = _inflate_bn(block2d.bn2)

    # conv3 (1x1 -> 1x1x1)
    conv3_3d = nn.Conv3d(
        block2d.conv3.in_channels, block2d.conv3.out_channels,
        kernel_size=1, bias=False
    )
    with torch.no_grad():
        conv3_3d.weight.copy_(block2d.conv3.weight.unsqueeze(2))
    bn3_3d = _inflate_bn(block2d.bn3)

    # Downsample
    downsample_3d = None
    if block2d.downsample is not None:
        ds_conv = block2d.downsample[0]
        ds_bn = block2d.downsample[1]
        ds_conv3d = nn.Conv3d(
            ds_conv.in_channels, ds_conv.out_channels, kernel_size=1,
            stride=(stride_3d[0], ds_conv.stride[0], ds_conv.stride[1]),
            bias=False
        )
        with torch.no_grad():
            ds_conv3d.weight.copy_(ds_conv.weight.unsqueeze(2))
        ds_bn3d = _inflate_bn(ds_bn)
        downsample_3d = nn.Sequential(ds_conv3d, ds_bn3d)

    return _Bottleneck3D(conv1_3d, bn1_3d, conv2_3d, bn2_3d,
                         conv3_3d, bn3_3d, downsample_3d)


class _Bottleneck3D(nn.Module):
    """3D Bottleneck blok kompatibilan s inflated ResNet."""

    def __init__(self, conv1, bn1, conv2, bn2, conv3, bn3, downsample=None):
        super().__init__()
        self.conv1, self.bn1 = conv1, bn1
        self.conv2, self.bn2 = conv2, bn2
        self.conv3, self.bn3 = conv3, bn3
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class InflatedResNet50Encoder3D(nn.Module):
    """
    Inflated ResNet50 enkoder: 2D pretrained težine proširene na 3D.

    Izlazi na 5 razina:
      x0: 64 ch, (D/2, H/2, W/2)
      x1: 256 ch, (D/2, H/4, W/4)
      x2: 512 ch, (D/2, H/8, W/8)
      x3: 1024 ch, (D/2, H/16, W/16)
      x4: 2048 ch, (D/2, H/32, W/32)

    Z-os se ne smanjuje (stride=1 u z) kako bi se očuvala dubinska rezolucija
    koja je ionako mala u MRI volumenima.
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT
                                 if pretrained else None)

        # conv1: 7x7 -> 3x7x7, stride (1,2,2)
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        if pretrained:
            with torch.no_grad():
                w2d = resnet.conv1.weight.data  # (64, 3, 7, 7)
                w_avg = w2d.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                w3d = w_avg.unsqueeze(2).repeat(1, 1, 3, 1, 1) / 3.0
                self.conv1.weight.copy_(w3d)

        self.bn1 = _inflate_bn(resnet.bn1) if pretrained else nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        # Infliraj svaki layer
        self.layer1 = self._inflate_layer(resnet.layer1)   # 256 ch
        self.layer2 = self._inflate_layer(resnet.layer2)   # 512 ch
        self.layer3 = self._inflate_layer(resnet.layer3)   # 1024 ch
        self.layer4 = self._inflate_layer(resnet.layer4)   # 2048 ch

        self.out_channels = [64, 256, 512, 1024, 2048]

    def _inflate_layer(self, layer2d):
        blocks = []
        for block in layer2d:
            blocks.append(_inflate_bottleneck(block, stride_3d=(1, 1, 1)))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))   # 64, D, H/2, W/2
        x1 = self.layer1(self.maxpool(x0))         # 256, D, H/4, W/4
        # Gradient checkpointing za dublje layere (štedi memoriju)
        if self.training and x.requires_grad:
            x2 = grad_checkpoint(self.layer2, x1, use_reentrant=False)
            x3 = grad_checkpoint(self.layer3, x2, use_reentrant=False)
            x4 = grad_checkpoint(self.layer4, x3, use_reentrant=False)
        else:
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


# =============================================================================
# ASPP3D
# =============================================================================

class ASPP3D(nn.Module):
    """3D ASPP sa smanjenim stopama dilatacije."""

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
            nn.GroupNorm(1, out_channels),
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


# =============================================================================
# MSAF3D (Multi-Scale Attention Fusion)
# =============================================================================

class MSAF3D(nn.Module):
    """3D Multi-Scale Attention Fusion na skip konekcijama."""

    def __init__(self, gate_channels: int, skip_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels), nn.PReLU(),
        )
        self.conv3x3_d1 = nn.Sequential(
            nn.Conv3d(skip_channels, out_channels, 3, padding=1, dilation=1,
                      bias=False),
            nn.BatchNorm3d(out_channels), nn.PReLU(),
        )
        self.conv3x3_d2 = nn.Sequential(
            nn.Conv3d(skip_channels, out_channels, 3, padding=2, dilation=2,
                      bias=False),
            nn.BatchNorm3d(out_channels), nn.PReLU(),
        )
        self.conv3x3_d3 = nn.Sequential(
            nn.Conv3d(skip_channels, out_channels, 3, padding=3, dilation=3,
                      bias=False),
            nn.BatchNorm3d(out_channels), nn.PReLU(),
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(skip_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels), nn.PReLU(),
        )
        self.gate_proj = nn.Sequential(
            nn.Conv3d(gate_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.transition = nn.Sequential(
            nn.Conv3d(out_channels * 6, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Conv3d(out_channels, 1, 1, bias=False),
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

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        size = skip.shape[2:]
        f1 = self.conv1x1(skip)
        f2 = self.conv3x3_d1(skip)
        f3 = self.conv3x3_d2(skip)
        f4 = self.conv3x3_d3(skip)
        f5 = self.global_pool(skip)
        f5 = F.interpolate(f5, size=size, mode="trilinear", align_corners=False)
        g = self.gate_proj(gate)
        fused = torch.cat([f1, f2, f3, f4, f5, g], dim=1)
        spatial_att = self.attention(self.transition(fused))
        channel_att = self.channel_att(skip).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return skip * spatial_att * channel_att


# =============================================================================
# Dual Attention Gate 3D (za ablaciju)
# =============================================================================

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


# =============================================================================
# FFM3D (Feature Fusion Module)
# =============================================================================

class FFM3D(nn.Module):
    """3D Feature Fusion Module — spaja izlaze svih decoder razina."""

    def __init__(self, channel_list: list, out_channels: int = 1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv3d(ch, out_channels, kernel_size=1) for ch in channel_list
        ])

    def forward(self, features: list, target_size: tuple) -> torch.Tensor:
        fused = None
        for proj, feat in zip(self.projections, features):
            out = proj(feat)
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode="trilinear",
                                    align_corners=False)
            fused = out if fused is None else fused + out
        return fused


# =============================================================================
# Boundary Refinement Module 3D
# =============================================================================

class BoundaryRefinementModule3D(nn.Module):
    """3D modul za detekciju rubova segmentacije."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, max(1, in_channels // 2), kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm3d(max(1, in_channels // 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, in_channels // 2), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary_conv(x)


# =============================================================================
# DSBANet3D — kompletna arhitektura
# =============================================================================

class DSBANet3D(nn.Module):
    """
    3D DSBANet za volumetrijsku binarnu segmentaciju.

    Konfigurabilne komponente:
      - use_se, use_pretrained, use_aspp, use_msaf, use_dual_attention,
        use_ffm, use_deep_supervision, use_boundary
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 16,
                 use_se: bool = True,
                 use_pretrained: bool = False,
                 use_aspp: bool = True,
                 use_msaf: bool = True,
                 use_dual_attention: bool = True,
                 use_ffm: bool = True,
                 use_deep_supervision: bool = True,
                 use_boundary: bool = True):
        super().__init__()
        self.use_pretrained = use_pretrained
        self.use_aspp = use_aspp
        self.use_msaf = use_msaf
        self.use_dual_attention = use_dual_attention
        self.use_ffm = use_ffm
        self.use_deep_supervision = use_deep_supervision
        self.use_boundary = use_boundary

        if use_pretrained:
            self._build_pretrained(in_channels, out_channels, use_se,
                                   use_aspp, use_msaf, use_dual_attention,
                                   use_ffm, use_deep_supervision, use_boundary)
        else:
            self._build_custom(in_channels, out_channels, base_filters, use_se,
                               use_aspp, use_msaf, use_dual_attention,
                               use_ffm, use_deep_supervision, use_boundary)

    def _build_pretrained(self, in_channels, out_channels, use_se,
                          use_aspp, use_msaf, use_dag, use_ffm, use_ds, use_brm):
        self.encoder = InflatedResNet50Encoder3D(in_channels, pretrained=True)
        s0, s1, s2, s3, fb = self.encoder.out_channels  # 64, 256, 512, 1024, 2048

        d4_ch, d3_ch, d2_ch, d1_ch = 512, 256, 128, 64
        Block = SEResConvBlock3D if use_se else ConvBlock3D

        if use_aspp:
            self.aspp = ASPP3D(fb, fb, rates=(2, 4, 6))
        self.bottleneck_proj = nn.Sequential(
            nn.Conv3d(fb, s3, 1, bias=False), nn.BatchNorm3d(s3), nn.ReLU(True))

        self.up4 = nn.ConvTranspose3d(s3, s3, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec4 = Block(s3 + s3, d4_ch)

        self.up3 = nn.ConvTranspose3d(d4_ch, d4_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = Block(d4_ch + s2, d3_ch)

        self.up2 = nn.ConvTranspose3d(d3_ch, d3_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = Block(d3_ch + s1, d2_ch)

        self.up1 = nn.ConvTranspose3d(d2_ch, d2_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1_a = Block(d2_ch + s0, d1_ch)

        self.up0 = nn.ConvTranspose3d(d1_ch, d1_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = Block(d1_ch, d1_ch)

        if use_msaf:
            self.msaf4 = MSAF3D(s3, s3, s3 // 4)
            self.msaf3 = MSAF3D(d4_ch, s2, s2 // 4)
            self.msaf2 = MSAF3D(d3_ch, s1, s1 // 4)
            self.msaf1 = MSAF3D(d2_ch, s0, max(s0 // 4, 1))
        elif use_dag:
            self.att4 = DualAttentionGate3D(s3, s3, s3 // 4)
            self.att3 = DualAttentionGate3D(d4_ch, s2, s2 // 4)
            self.att2 = DualAttentionGate3D(d3_ch, s1, s1 // 4)
            self.att1 = DualAttentionGate3D(d2_ch, s0, max(s0 // 4, 1))

        self.final_conv = nn.Conv3d(d1_ch, out_channels, kernel_size=1)

        if use_ffm:
            self.ffm = FFM3D([d4_ch, d3_ch, d2_ch, d1_ch], out_channels)
        if use_ds:
            self.aux_head4 = nn.Conv3d(d4_ch, out_channels, kernel_size=1)
            self.aux_head3 = nn.Conv3d(d3_ch, out_channels, kernel_size=1)
            self.aux_head2 = nn.Conv3d(d2_ch, out_channels, kernel_size=1)
        if use_brm:
            self.boundary_module = BoundaryRefinementModule3D(d1_ch)

    def _build_custom(self, in_channels, out_channels, base_filters, use_se,
                      use_aspp, use_msaf, use_dag, use_ffm, use_ds, use_brm):
        f = base_filters
        Block = SEResConvBlock3D if use_se else ConvBlock3D

        self.enc1 = Block(in_channels, f)
        self.enc2 = Block(f, f * 2)
        self.enc3 = Block(f * 2, f * 4)
        self.enc4 = Block(f * 4, f * 8)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = Block(f * 8, f * 16)
        if use_aspp:
            self.aspp = ASPP3D(f * 16, f * 16, rates=(3, 6, 9))

        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)

        if use_msaf:
            self.msaf4 = MSAF3D(f * 8, f * 8, f * 4)
            self.msaf3 = MSAF3D(f * 4, f * 4, f * 2)
            self.msaf2 = MSAF3D(f * 2, f * 2, f)
            self.msaf1 = MSAF3D(f, f, max(1, f // 2))
        elif use_dag:
            self.att4 = DualAttentionGate3D(f * 8, f * 8, f * 4)
            self.att3 = DualAttentionGate3D(f * 4, f * 4, f * 2)
            self.att2 = DualAttentionGate3D(f * 2, f * 2, f)
            self.att1 = DualAttentionGate3D(f, f, max(1, f // 2))

        self.dec4 = Block(f * 16, f * 8)
        self.dec3 = Block(f * 8, f * 4)
        self.dec2 = Block(f * 4, f * 2)
        self.dec1 = Block(f * 2, f)

        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)

        if use_ffm:
            self.ffm = FFM3D([f * 8, f * 4, f * 2, f], out_channels)
        if use_ds:
            self.aux_head4 = nn.Conv3d(f * 8, out_channels, kernel_size=1)
            self.aux_head3 = nn.Conv3d(f * 4, out_channels, kernel_size=1)
            self.aux_head2 = nn.Conv3d(f * 2, out_channels, kernel_size=1)
        if use_brm:
            self.boundary_module = BoundaryRefinementModule3D(f)

    def _apply_attention(self, gate, skip, level):
        if self.use_msaf:
            msaf = getattr(self, f"msaf{level}", None)
            if msaf is not None:
                skip = msaf(gate=gate, skip=skip)
        elif self.use_dual_attention:
            att = getattr(self, f"att{level}", None)
            if att is not None:
                skip = att(gate=gate, skip=skip)
        return torch.cat([gate, skip], dim=1)

    @staticmethod
    def _match_size(x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode="trilinear",
                              align_corners=False)
        return x

    @staticmethod
    def _pad_to_match(upsampled, skip):
        diff_d = skip.size(2) - upsampled.size(2)
        diff_h = skip.size(3) - upsampled.size(3)
        diff_w = skip.size(4) - upsampled.size(4)
        return F.pad(upsampled, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ])

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        if self.use_pretrained:
            return self._forward_pretrained(x, input_size)
        else:
            return self._forward_custom(x, input_size)

    def _forward_pretrained(self, x, input_size):
        features = self.encoder(x)
        x0, x1, x2, x3, x4 = features

        b = x4
        if self.use_aspp:
            if self.training and x.requires_grad:
                b = grad_checkpoint(self.aspp, b, use_reentrant=False)
            else:
                b = self.aspp(b)
        b = self.bottleneck_proj(b)

        d4 = self._match_size(self.up4(b), x3)
        d4 = self.dec4(self._apply_attention(d4, x3, 4))

        # Oslobodi encoder feature mape čim nisu potrebne
        del x4, b

        d3 = self._match_size(self.up3(d4), x2)
        d3 = self.dec3(self._apply_attention(d3, x2, 3))
        del x3

        d2 = self._match_size(self.up2(d3), x1)
        d2 = self.dec2(self._apply_attention(d2, x1, 2))
        del x2

        d1 = self._match_size(self.up1(d2), x0)
        d1 = self.dec1_a(self._apply_attention(d1, x0, 1))
        del x1, x0

        d1 = self.up0(d1)
        if d1.shape[2:] != input_size:
            d1 = F.interpolate(d1, size=input_size, mode="trilinear",
                               align_corners=False)
        d1 = self.dec1(d1)

        main_out = self.final_conv(d1)
        if self.use_ffm:
            main_out = main_out + self.ffm([d4, d3, d2, d1], input_size)

        return self._make_output(main_out, d4, d3, d2, d1, input_size)

    def _forward_custom(self, x, input_size):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))
        if self.use_aspp:
            b = self.aspp(b)

        d4 = self._pad_to_match(self.up4(b), e4)
        d4 = self.dec4(self._apply_attention(d4, e4, 4))

        d3 = self._pad_to_match(self.up3(d4), e3)
        d3 = self.dec3(self._apply_attention(d3, e3, 3))

        d2 = self._pad_to_match(self.up2(d3), e2)
        d2 = self.dec2(self._apply_attention(d2, e2, 2))

        d1 = self._pad_to_match(self.up1(d2), e1)
        d1 = self.dec1(self._apply_attention(d1, e1, 1))

        main_out = self.final_conv(d1)
        if self.use_ffm:
            main_out = main_out + self.ffm([d4, d3, d2, d1], input_size)

        return self._make_output(main_out, d4, d3, d2, d1, input_size)

    def _make_output(self, main_out, d4, d3, d2, d1, input_size):
        if self.training and (self.use_deep_supervision or self.use_boundary):
            result = {"main": main_out}
            if self.use_deep_supervision:
                result["aux4"] = F.interpolate(
                    self.aux_head4(d4), size=input_size,
                    mode="trilinear", align_corners=False)
                result["aux3"] = F.interpolate(
                    self.aux_head3(d3), size=input_size,
                    mode="trilinear", align_corners=False)
                result["aux2"] = F.interpolate(
                    self.aux_head2(d2), size=input_size,
                    mode="trilinear", align_corners=False)
            if self.use_boundary:
                result["boundary"] = self.boundary_module(d1)
            return result
        return main_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
