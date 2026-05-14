"""
DSBANet (Deep Supervision Boundary-Aware Attention Network) - 2D varijanta.

Nova arhitektura za segmentaciju prostate na MR slikama koja kombinira:
  1. Pretrained ResNet50 enkoder (ili SE-Residual blokovi from scratch)
  2. ASPP modul (Atrous Spatial Pyramid Pooling) u bottlenecku
  3. MSAF (Multi-Scale Attention Fusion) na skip konekcijama
  4. FFM (Feature Fusion Module) — spajanje svih decoder izlaza
  5. Deep Supervision (pomoćni izlazi na svakoj razini dekodera)
  6. Boundary Refinement Module (detekcija rubova za precizniju segmentaciju)

Svaka komponenta se može uključiti/isključiti za ablacijsku studiju.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
# Bazni blokovi
# =============================================================================

class ConvBlock2D(nn.Module):
    """Standardni dvostruki konvolucijski blok (bez SE, bez residuala)."""

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


class SEResConvBlock2D(nn.Module):
    """
    Rezidualni konvolucijski blok s Squeeze-and-Excitation attention mehanizmom
    i opcionim dropout-om za regularizaciju.
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16,
                 drop_rate: float = 0.0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        r = max(1, out_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, out_channels),
            nn.Sigmoid(),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_block(x)
        se_weights = self.se(out).unsqueeze(-1).unsqueeze(-1)
        out = out * se_weights
        return self.relu(out + residual)


# =============================================================================
# Pretrained ResNet50 Encoder
# =============================================================================

class ResNet50Encoder(nn.Module):
    """
    Pretrained ResNet50 enkoder koji izvlači značajke na 4 razine.

    Ulaz: (B, C_in, H, W)  — C_in može biti 1 ili 3
    Izlazi: lista od 4 feature mape na rezolucijama H/2, H/4, H/8, H/16
    te bottleneck na H/32.
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT
                                 if pretrained else None)

        # Prilagodi prvi conv za 1-kanalni ulaz (grayscale MRI)
        if in_channels != 3:
            original_conv = resnet.conv1
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Inicijaliziraj iz pretrained težina: prosječni kanal
                with torch.no_grad():
                    self.conv1.weight.copy_(
                        original_conv.weight.mean(dim=1, keepdim=True).repeat(
                            1, in_channels, 1, 1)
                    )
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 ch, H/4
        self.layer2 = resnet.layer2  # 512 ch, H/8
        self.layer3 = resnet.layer3  # 1024 ch, H/16
        self.layer4 = resnet.layer4  # 2048 ch, H/32

        # Kanali izlaza na svakoj razini
        self.out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        # Razina 0: H/2 (nakon conv1+bn+relu, prije maxpool)
        x0 = self.relu(self.bn1(self.conv1(x)))  # 64 ch, H/2
        # Razina 1: H/4
        x1 = self.layer1(self.maxpool(x0))  # 256 ch, H/4
        # Razina 2: H/8
        x2 = self.layer2(x1)  # 512 ch, H/8
        # Razina 3: H/16
        x3 = self.layer3(x2)  # 1024 ch, H/16
        # Bottleneck: H/32
        x4 = self.layer4(x3)  # 2048 ch, H/32

        return [x0, x1, x2, x3, x4]


# =============================================================================
# ASPP (Atrous Spatial Pyramid Pooling)
# =============================================================================

class ASPP2D(nn.Module):
    """Atrous Spatial Pyramid Pooling za multi-scale kontekst u bottlenecku."""

    def __init__(self, in_channels: int, out_channels: int,
                 rates: tuple = (6, 12, 18)):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate,
                          dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )
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
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)
        features.append(gp)
        return self.project(torch.cat(features, dim=1))


# =============================================================================
# MSAF (Multi-Scale Attention Fusion) — na skip konekcijama
# =============================================================================

class MSAF2D(nn.Module):
    """
    Multi-Scale Attention Fusion modul za skip konekcije.

    Kombinira gate i skip značajke kroz paralelne dilatirane konvolucije
    + global pooling za multi-scale kontekst, te generira attention mapu.
    """

    def __init__(self, gate_channels: int, skip_channels: int,
                 out_channels: int):
        super().__init__()
        # Multi-scale grane na skip značajkama
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.conv3x3_d1 = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 3, padding=1, dilation=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.conv3x3_d2 = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 3, padding=2, dilation=2,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.conv3x3_d3 = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 3, padding=3, dilation=3,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_channels, out_channels, 1, bias=True),
            nn.PReLU(),
        )

        # Gate projekcija
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Transition: spoji 5 multi-scale grana + gate -> attention
        self.transition = nn.Sequential(
            nn.Conv2d(out_channels * 6, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1, bias=False),
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

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        size = skip.shape[2:]

        # Multi-scale grane
        f1 = self.conv1x1(skip)
        f2 = self.conv3x3_d1(skip)
        f3 = self.conv3x3_d2(skip)
        f4 = self.conv3x3_d3(skip)
        f5 = self.global_pool(skip)
        f5 = F.interpolate(f5, size=size, mode="bilinear", align_corners=False)

        # Gate projekcija
        g = self.gate_proj(gate)

        # Spoji sve i generiraj spatial attention mapu
        fused = torch.cat([f1, f2, f3, f4, f5, g], dim=1)
        spatial_att = self.attention(self.transition(fused))

        # Channel attention
        channel_att = self.channel_att(skip).unsqueeze(-1).unsqueeze(-1)

        return skip * spatial_att * channel_att


# =============================================================================
# Dual Attention Gate (Channel + Spatial Attention) — za ablaciju
# =============================================================================

class DualAttentionGate2D(nn.Module):
    """Dual Attention Gate koji kombinira spatial i channel attention."""

    def __init__(self, gate_channels: int, skip_channels: int,
                 inter_channels: int):
        super().__init__()
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
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        spatial_att = self.psi(self.relu(g + s))
        channel_att = self.channel_att(skip).unsqueeze(-1).unsqueeze(-1)
        return skip * spatial_att * channel_att


# =============================================================================
# FFM (Feature Fusion Module)
# =============================================================================

class FFM2D(nn.Module):
    """
    Feature Fusion Module — spaja izlaze svih decoder razina
    u jednu prediktivnu mapu. Svaka razina se projicira na 1 kanal,
    upsampla na ciljanu rezoluciju i zbraja.
    """

    def __init__(self, channel_list: list, out_channels: int = 1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1) for ch in channel_list
        ])

    def forward(self, features: list, target_size: tuple) -> torch.Tensor:
        fused = None
        for proj, feat in zip(self.projections, features):
            out = proj(feat)
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode="bilinear",
                                    align_corners=False)
            fused = out if fused is None else fused + out
        return fused


# =============================================================================
# Boundary Refinement Module
# =============================================================================

class BoundaryRefinementModule2D(nn.Module):
    """Modul za poboljšanje detekcije rubova segmentacije."""

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
# DSBANet 2D — kompletna arhitektura
# =============================================================================

class DSBANet2D(nn.Module):
    """
    DSBANet (Deep Supervision Boundary-Aware Attention Network) za 2D
    binarnu segmentaciju.

    Konfigurabilne komponente za ablacijsku studiju:
      - use_se: SE-Residual blokovi (False = obični konv. blokovi)
      - use_pretrained: Pretrained ResNet50 enkoder
      - use_aspp: ASPP u bottlenecku
      - use_msaf: MSAF multi-scale attention na skip konekcijama
      - use_dual_attention: Dual Attention Gate (zamjena za MSAF)
      - use_ffm: Feature Fusion Module za spajanje decoder izlaza
      - use_deep_supervision: Pomoćni izlazi na razinama dekodera
      - use_boundary: Boundary Refinement Module
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32,
                 use_se: bool = True,
                 use_pretrained: bool = True,
                 use_aspp: bool = True,
                 use_msaf: bool = True,
                 use_dual_attention: bool = True,
                 use_ffm: bool = True,
                 use_deep_supervision: bool = True,
                 use_boundary: bool = True,
                 decoder_dropout: float = 0.1):
        super().__init__()
        self.use_pretrained = use_pretrained
        self.use_aspp = use_aspp
        self.use_msaf = use_msaf
        self.use_dual_attention = use_dual_attention
        self.use_ffm = use_ffm
        self.use_deep_supervision = use_deep_supervision
        self.use_boundary = use_boundary

        dec_drop = decoder_dropout

        if use_pretrained:
            # ---- Pretrained ResNet50 Encoder ----
            self.encoder = ResNet50Encoder(in_channels, pretrained=True)
            enc_ch = self.encoder.out_channels
            s0, s1, s2, s3, fb = enc_ch  # 64, 256, 512, 1024, 2048

            d4_ch = 512
            d3_ch = 256
            d2_ch = 128
            d1_ch = 64

            # Bottleneck
            if use_aspp:
                self.aspp = ASPP2D(fb, fb)
            self.bottleneck_proj = nn.Sequential(
                nn.Conv2d(fb, s3, 1, bias=False),
                nn.BatchNorm2d(s3),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
            )

            # Decoder s dropout regularizacijom
            self.up4 = nn.ConvTranspose2d(s3, s3, kernel_size=2, stride=2)
            self.dec4 = SEResConvBlock2D(s3 + s3, d4_ch, drop_rate=dec_drop) if use_se else ConvBlock2D(s3 + s3, d4_ch)

            self.up3 = nn.ConvTranspose2d(d4_ch, d4_ch, kernel_size=2, stride=2)
            self.dec3 = SEResConvBlock2D(d4_ch + s2, d3_ch, drop_rate=dec_drop) if use_se else ConvBlock2D(d4_ch + s2, d3_ch)

            self.up2 = nn.ConvTranspose2d(d3_ch, d3_ch, kernel_size=2, stride=2)
            self.dec2 = SEResConvBlock2D(d3_ch + s1, d2_ch, drop_rate=dec_drop * 0.5) if use_se else ConvBlock2D(d3_ch + s1, d2_ch)

            self.up1 = nn.ConvTranspose2d(d2_ch, d2_ch, kernel_size=2, stride=2)
            self.dec1_a = SEResConvBlock2D(d2_ch + s0, d1_ch) if use_se else ConvBlock2D(d2_ch + s0, d1_ch)

            self.up0 = nn.ConvTranspose2d(d1_ch, d1_ch, kernel_size=2, stride=2)
            self.dec1 = SEResConvBlock2D(d1_ch, d1_ch) if use_se else ConvBlock2D(d1_ch, d1_ch)

            # MSAF na skip konekcijama (gate_ch, skip_ch, inter_ch)
            if use_msaf:
                self.msaf4 = MSAF2D(s3, s3, s3 // 4)   # gate=1024, skip=1024
                self.msaf3 = MSAF2D(d4_ch, s2, s2 // 4) # gate=512, skip=512
                self.msaf2 = MSAF2D(d3_ch, s1, s1 // 4) # gate=256, skip=256
                self.msaf1 = MSAF2D(d2_ch, s0, max(s0 // 4, 1)) # gate=128, skip=64
            elif use_dual_attention:
                self.att4 = DualAttentionGate2D(s3, s3, s3 // 4)
                self.att3 = DualAttentionGate2D(d4_ch, s2, s2 // 4)
                self.att2 = DualAttentionGate2D(d3_ch, s1, s1 // 4)
                self.att1 = DualAttentionGate2D(d2_ch, s0, max(s0 // 4, 1))

            # Final conv
            self.final_conv = nn.Conv2d(d1_ch, out_channels, kernel_size=1)

            # FFM (spoji izlaze svih decoder razina)
            if use_ffm:
                self.ffm = FFM2D([d4_ch, d3_ch, d2_ch, d1_ch], out_channels)

            # Deep Supervision
            if use_deep_supervision:
                self.aux_head4 = nn.Conv2d(d4_ch, out_channels, kernel_size=1)
                self.aux_head3 = nn.Conv2d(d3_ch, out_channels, kernel_size=1)
                self.aux_head2 = nn.Conv2d(d2_ch, out_channels, kernel_size=1)

            # Boundary
            if use_boundary:
                self.boundary_module = BoundaryRefinementModule2D(d1_ch)

        else:
            # ---- Custom SE-Residual Encoder (from scratch) ----
            f = base_filters
            Block = SEResConvBlock2D if use_se else ConvBlock2D

            self.enc1 = Block(in_channels, f)
            self.enc2 = Block(f, f * 2)
            self.enc3 = Block(f * 2, f * 4)
            self.enc4 = Block(f * 4, f * 8)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.bottleneck = Block(f * 8, f * 16)
            if use_aspp:
                self.aspp = ASPP2D(f * 16, f * 16)

            self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
            self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
            self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
            self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)

            if use_msaf:
                self.msaf4 = MSAF2D(f * 8, f * 8, f * 4)
                self.msaf3 = MSAF2D(f * 4, f * 4, f * 2)
                self.msaf2 = MSAF2D(f * 2, f * 2, f)
                self.msaf1 = MSAF2D(f, f, f // 2)
            elif use_dual_attention:
                self.att4 = DualAttentionGate2D(f * 8, f * 8, f * 4)
                self.att3 = DualAttentionGate2D(f * 4, f * 4, f * 2)
                self.att2 = DualAttentionGate2D(f * 2, f * 2, f)
                self.att1 = DualAttentionGate2D(f, f, f // 2)

            self.dec4 = Block(f * 16, f * 8)
            self.dec3 = Block(f * 8, f * 4)
            self.dec2 = Block(f * 4, f * 2)
            self.dec1 = Block(f * 2, f)

            self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

            if use_ffm:
                self.ffm = FFM2D([f * 8, f * 4, f * 2, f], out_channels)

            if use_deep_supervision:
                self.aux_head4 = nn.Conv2d(f * 8, out_channels, kernel_size=1)
                self.aux_head3 = nn.Conv2d(f * 4, out_channels, kernel_size=1)
                self.aux_head2 = nn.Conv2d(f * 2, out_channels, kernel_size=1)

            if use_boundary:
                self.boundary_module = BoundaryRefinementModule2D(f)

    def _apply_attention(self, gate, skip, level):
        """Primijeni MSAF, DAG ili direktnu konkatenaciju."""
        if self.use_msaf:
            msaf = getattr(self, f"msaf{level}", None)
            if msaf is not None:
                skip = msaf(gate=gate, skip=skip)
        elif self.use_dual_attention:
            att = getattr(self, f"att{level}", None)
            if att is not None:
                skip = att(gate=gate, skip=skip)
        return torch.cat([gate, skip], dim=1)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        if self.use_pretrained:
            return self._forward_pretrained(x, input_size)
        else:
            return self._forward_custom(x, input_size)

    def _forward_pretrained(self, x, input_size):
        # Encoder: [x0(64,H/2), x1(256,H/4), x2(512,H/8), x3(1024,H/16), x4(2048,H/32)]
        features = self.encoder(x)
        x0, x1, x2, x3, x4 = features

        # Bottleneck: 2048 -> 1024
        b = x4
        if self.use_aspp:
            b = self.aspp(b)
        b = self.bottleneck_proj(b)

        # Decoder razina 4: up(1024)->H/16, skip=x3(1024), cat->2048, dec->512
        d4 = self._upsample_to(self.up4(b), x3)
        d4 = self.dec4(self._apply_attention(d4, x3, 4))

        # Decoder razina 3: up(512)->H/8, skip=x2(512), cat->1024, dec->256
        d3 = self._upsample_to(self.up3(d4), x2)
        d3 = self.dec3(self._apply_attention(d3, x2, 3))

        # Decoder razina 2: up(256)->H/4, skip=x1(256), cat->512, dec->128
        d2 = self._upsample_to(self.up2(d3), x1)
        d2 = self.dec2(self._apply_attention(d2, x1, 2))

        # Decoder razina 1: up(128)->H/2, skip=x0(64), cat->192, dec1_a->64
        d1 = self._upsample_to(self.up1(d2), x0)
        d1 = self.dec1_a(self._apply_attention(d1, x0, 1))

        # Zadnji upsampling na originalnu rezoluciju H
        d1 = self.up0(d1)
        if d1.shape[2:] != input_size:
            d1 = F.interpolate(d1, size=input_size, mode="bilinear",
                               align_corners=False)
        d1 = self.dec1(d1)

        # Finalni izlaz
        main_out = self.final_conv(d1)

        # FFM: spoji sve decoder razine
        if self.use_ffm:
            ffm_out = self.ffm([d4, d3, d2, d1], input_size)
            main_out = main_out + ffm_out

        if self.training and (self.use_deep_supervision or self.use_boundary):
            result = {"main": main_out}
            if self.use_deep_supervision:
                result["aux4"] = F.interpolate(
                    self.aux_head4(d4), size=input_size,
                    mode="bilinear", align_corners=False)
                result["aux3"] = F.interpolate(
                    self.aux_head3(d3), size=input_size,
                    mode="bilinear", align_corners=False)
                result["aux2"] = F.interpolate(
                    self.aux_head2(d2), size=input_size,
                    mode="bilinear", align_corners=False)
            if self.use_boundary:
                result["boundary"] = self.boundary_module(d1)
            return result

        return main_out

    @staticmethod
    def _upsample_to(x, target):
        """Upsample x da odgovara prostornim dimenzijama targeta."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode="bilinear",
                              align_corners=False)
        return x

    def _forward_custom(self, x, input_size):
        # Custom encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))
        if self.use_aspp:
            b = self.aspp(b)

        d4 = self.up4(b)
        d4 = self.dec4(self._apply_attention(d4, e4, 4))

        d3 = self.up3(d4)
        d3 = self.dec3(self._apply_attention(d3, e3, 3))

        d2 = self.up2(d3)
        d2 = self.dec2(self._apply_attention(d2, e2, 2))

        d1 = self.up1(d2)
        d1 = self.dec1(self._apply_attention(d1, e1, 1))

        main_out = self.final_conv(d1)

        if self.use_ffm:
            ffm_out = self.ffm([d4, d3, d2, d1], input_size)
            main_out = main_out + ffm_out

        if self.training and (self.use_deep_supervision or self.use_boundary):
            result = {"main": main_out}
            if self.use_deep_supervision:
                result["aux4"] = F.interpolate(
                    self.aux_head4(d4), size=input_size,
                    mode="bilinear", align_corners=False)
                result["aux3"] = F.interpolate(
                    self.aux_head3(d3), size=input_size,
                    mode="bilinear", align_corners=False)
                result["aux2"] = F.interpolate(
                    self.aux_head2(d2), size=input_size,
                    mode="bilinear", align_corners=False)
            if self.use_boundary:
                result["boundary"] = self.boundary_module(d1)
            return result

        return main_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
