"""
Swin-UNet arhitektura za segmentaciju prostate na MR slikama.

Referenca: Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical
Image Segmentation", ECCV 2022.

Potpuno transformer-based U-Net arhitektura koja koristi Swin Transformer
blokove za enkodiranje i dekodiranje, s patch merging/expanding operacijama
umjesto poolinga i transponiranih konvolucija.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Pretvara sliku u sekvencu patcheva pomoću konvolucije."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention s relativnim pozicijskim biasom."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer blok s window-based self-attention.

    Uključuje opciju za shifted window attention (SW-MSA) kada je shift_size > 0.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad ako dimenzije nisu djeljive s window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        # Shifted window
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))

        # Particioniraj u prozore
        x = x.view(B, Hp // self.window_size, self.window_size,
                    Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)

        # Window attention
        x = self.attn(x)

        # Vrati iz prozora
        x = x.view(B, Hp // self.window_size, Wp // self.window_size,
                    self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))

        # Ukloni padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Smanjuje rezoluciju 2x spajanjem 2x2 patcheva (analogno downsamplingu)."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Pad ako neparno
        if H % 2 == 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
        if W % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
        Hp, Wp = x.shape[1], x.shape[2]

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, Hp // 2, Wp // 2


class PatchExpanding(nn.Module):
    """Povećava rezoluciju 2x (analogno upsamplingu)."""

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 4 * (C // 2))
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = self.norm(x)
        x = x.view(B, -1, C // 2)
        return x, H * 2, W * 2


class SwinUNet(nn.Module):
    """
    Swin-UNet za binarnu segmentaciju.

    Potpuno transformer-based U-Net s:
    - Patch embedding na ulazu
    - Swin Transformer blokovi za enkodiranje i dekodiranje
    - Patch merging za downsampling
    - Patch expanding za upsampling
    - Skip konekcije između enkodera i dekodera

    Ulaz: (B, C, H, W) - dimenzije moraju biti djeljive s patch_size * 8
    Izlaz: (B, 1, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32, patch_size: int = 4,
                 window_size: int = 7, num_heads: int = None):
        super().__init__()
        embed_dim = base_filters * 2  # Bazna dimenzija za transformer
        self.patch_size = patch_size

        # Osiguraj minimalne dimenzije
        if num_heads is None:
            num_heads = max(2, embed_dim // 16)

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)

        # Enkoder
        self.enc_blocks1 = nn.ModuleList([
            SwinTransformerBlock(embed_dim, num_heads, window_size, shift_size=0),
            SwinTransformerBlock(embed_dim, num_heads, window_size,
                                shift_size=window_size // 2),
        ])
        self.downsample1 = PatchMerging(embed_dim)

        self.enc_blocks2 = nn.ModuleList([
            SwinTransformerBlock(embed_dim * 2, num_heads * 2, window_size,
                                shift_size=0),
            SwinTransformerBlock(embed_dim * 2, num_heads * 2, window_size,
                                shift_size=window_size // 2),
        ])
        self.downsample2 = PatchMerging(embed_dim * 2)

        self.enc_blocks3 = nn.ModuleList([
            SwinTransformerBlock(embed_dim * 4, num_heads * 4, window_size,
                                shift_size=0),
            SwinTransformerBlock(embed_dim * 4, num_heads * 4, window_size,
                                shift_size=window_size // 2),
        ])
        self.downsample3 = PatchMerging(embed_dim * 4)

        # Bottleneck
        self.bottleneck_blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim * 8, num_heads * 8, window_size,
                                shift_size=0),
            SwinTransformerBlock(embed_dim * 8, num_heads * 8, window_size,
                                shift_size=window_size // 2),
        ])

        # Dekoder
        self.upsample3 = PatchExpanding(embed_dim * 8)
        self.skip_proj3 = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.dec_blocks3 = nn.ModuleList([
            SwinTransformerBlock(embed_dim * 4, num_heads * 4, window_size,
                                shift_size=0),
            SwinTransformerBlock(embed_dim * 4, num_heads * 4, window_size,
                                shift_size=window_size // 2),
        ])

        self.upsample2 = PatchExpanding(embed_dim * 4)
        self.skip_proj2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.dec_blocks2 = nn.ModuleList([
            SwinTransformerBlock(embed_dim * 2, num_heads * 2, window_size,
                                shift_size=0),
            SwinTransformerBlock(embed_dim * 2, num_heads * 2, window_size,
                                shift_size=window_size // 2),
        ])

        self.upsample1 = PatchExpanding(embed_dim * 2)
        self.skip_proj1 = nn.Linear(embed_dim * 2, embed_dim)
        self.dec_blocks1 = nn.ModuleList([
            SwinTransformerBlock(embed_dim, num_heads, window_size, shift_size=0),
            SwinTransformerBlock(embed_dim, num_heads, window_size,
                                shift_size=window_size // 2),
        ])

        # Završni upsampling natrag na originalnu rezoluciju
        self.final_upsample = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * patch_size * patch_size),
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H_orig, W_orig = x.shape

        # Patch embedding
        x, H, W = self.patch_embed(x)

        # Enkoder razina 1
        skip1 = x
        for block in self.enc_blocks1:
            x = block(x, H, W)
        skip1_out = x
        x, H, W = self.downsample1(x, H, W)

        # Enkoder razina 2
        for block in self.enc_blocks2:
            x = block(x, H, W)
        skip2_out = x
        x, H, W = self.downsample2(x, H, W)

        # Enkoder razina 3
        for block in self.enc_blocks3:
            x = block(x, H, W)
        skip3_out = x
        x, H, W = self.downsample3(x, H, W)

        # Bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, H, W)

        # Dekoder razina 3
        x, H, W = self.upsample3(x, H, W)
        x = torch.cat([x, skip3_out], dim=-1)
        x = self.skip_proj3(x)
        for block in self.dec_blocks3:
            x = block(x, H, W)

        # Dekoder razina 2
        x, H, W = self.upsample2(x, H, W)
        x = torch.cat([x, skip2_out], dim=-1)
        x = self.skip_proj2(x)
        for block in self.dec_blocks2:
            x = block(x, H, W)

        # Dekoder razina 1
        x, H, W = self.upsample1(x, H, W)
        x = torch.cat([x, skip1_out], dim=-1)
        x = self.skip_proj1(x)
        for block in self.dec_blocks1:
            x = block(x, H, W)

        # Završni upsampling natrag na originalnu rezoluciju
        x = self.final_norm(x)
        B, L, C = x.shape
        x = self.final_upsample(x)
        x = x.view(B, H, W, self.patch_size, self.patch_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H * self.patch_size, W * self.patch_size)

        # Prilagodi na originalnu rezoluciju ako je potrebno
        if x.shape[2] != H_orig or x.shape[3] != W_orig:
            x = F.interpolate(x, size=(H_orig, W_orig), mode="bilinear",
                              align_corners=False)

        return self.final_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
