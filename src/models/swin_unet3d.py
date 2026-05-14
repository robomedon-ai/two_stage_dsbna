"""
3D Swin-UNet arhitektura za segmentaciju prostate na MR slikama.

Referenca: Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical
Image Segmentation", ECCV 2022. (3D varijanta)

3D verzija potpuno transformer-based U-Net arhitekture koja koristi
3D Swin Transformer blokove s 3D window attention za volumetrijsku segmentaciju.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding3D(nn.Module):
    """Pretvara 3D volumen u sekvencu patcheva pomoću 3D konvolucije."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)  # (B, C, D/P, H/P, W/P)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, D*H*W, C)
        x = self.norm(x)
        return x, D, H, W


class WindowAttention3D(nn.Module):
    """3D Window-based Multi-Head Self-Attention s relativnim pozicijskim biasom."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relativni pozicijski bias za 3D prozore
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) ** 3,
                num_heads,
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Izračunaj relativne pozicijske indekse
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij",
        ))  # (3, ws, ws, ws)
        coords_flatten = torch.flatten(coords, 1)  # (3, ws^3)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 2] += window_size - 1
        relative_coords[:, :, 0] *= (2 * window_size - 1) ** 2
        relative_coords[:, :, 1] *= (2 * window_size - 1)
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

        ws3 = self.window_size ** 3
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws3, ws3, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer blok s window-based self-attention."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 4,
                 shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        ws = self.window_size

        # Pad za djeljivost s window_size
        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = x.shape[1], x.shape[2], x.shape[3]

        # Shifted window
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size,
                                       -self.shift_size), dims=(1, 2, 3))

        # Particioniraj u 3D prozore
        x = x.view(B, Dp // ws, ws, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(-1, ws * ws * ws, C)

        # Window attention
        x = self.attn(x)

        # Vrati iz prozora
        x = x.view(B, Dp // ws, Hp // ws, Wp // ws, ws, ws, ws, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, Dp, Hp, Wp, C)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size,
                                       self.shift_size), dims=(1, 2, 3))

        # Ukloni padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = x.view(B, D * H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging3D(nn.Module):
    """3D Patch Merging - smanjuje rezoluciju 2x spajanjem 2x2x2 patcheva."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)

        # Pad ako neparne dimenzije
        if D % 2 == 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        if H % 2 == 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
        if W % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
        Dp, Hp, Wp = x.shape[1], x.shape[2], x.shape[3]

        # Spoji 2x2x2 patcheve
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, Dp // 2, Hp // 2, Wp // 2


class PatchExpanding3D(nn.Module):
    """3D Patch Expanding - povećava rezoluciju 2x."""

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int):
        B, L, C = x.shape
        x = self.expand(x)  # (B, L, 4*C)
        x = x.view(B, D, H, W, 4 * C)

        # Reorganiziraj u 2x2x2 ekspanziju
        # 4*C -> 8 * (C/2), raspoređujemo u 2x2x2
        C_out = C // 2
        x = x.view(B, D, H, W, 2, 2, 2, C_out)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, D * 2, H * 2, W * 2, C_out)
        x = self.norm(x)
        x = x.view(B, -1, C_out)
        return x, D * 2, H * 2, W * 2


class SwinUNet3D(nn.Module):
    """
    3D Swin-UNet za binarnu volumetrijsku segmentaciju.

    Potpuno transformer-based 3D U-Net s:
    - 3D Patch embedding na ulazu
    - 3D Swin Transformer blokovi za enkodiranje i dekodiranje
    - 3D Patch merging za downsampling
    - 3D Patch expanding za upsampling
    - Skip konekcije između enkodera i dekodera

    Ulaz: (B, C, D, H, W)
    Izlaz: (B, 1, D, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32, patch_size: int = 2,
                 window_size: int = 4, num_heads: int = None):
        super().__init__()
        embed_dim = base_filters * 2
        self.patch_size = patch_size

        if num_heads is None:
            num_heads = max(2, embed_dim // 16)

        # Patch embedding
        self.patch_embed = PatchEmbedding3D(in_channels, embed_dim, patch_size)

        # Enkoder
        self.enc_blocks1 = nn.ModuleList([
            SwinTransformerBlock3D(embed_dim, num_heads, window_size, shift_size=0),
            SwinTransformerBlock3D(embed_dim, num_heads, window_size,
                                  shift_size=window_size // 2),
        ])
        self.downsample1 = PatchMerging3D(embed_dim)

        self.enc_blocks2 = nn.ModuleList([
            SwinTransformerBlock3D(embed_dim * 2, num_heads * 2, window_size,
                                  shift_size=0),
            SwinTransformerBlock3D(embed_dim * 2, num_heads * 2, window_size,
                                  shift_size=window_size // 2),
        ])
        self.downsample2 = PatchMerging3D(embed_dim * 2)

        # Bottleneck
        self.bottleneck_blocks = nn.ModuleList([
            SwinTransformerBlock3D(embed_dim * 4, num_heads * 4, window_size,
                                  shift_size=0),
            SwinTransformerBlock3D(embed_dim * 4, num_heads * 4, window_size,
                                  shift_size=window_size // 2),
        ])

        # Dekoder
        self.upsample2 = PatchExpanding3D(embed_dim * 4)
        self.skip_proj2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.dec_blocks2 = nn.ModuleList([
            SwinTransformerBlock3D(embed_dim * 2, num_heads * 2, window_size,
                                  shift_size=0),
            SwinTransformerBlock3D(embed_dim * 2, num_heads * 2, window_size,
                                  shift_size=window_size // 2),
        ])

        self.upsample1 = PatchExpanding3D(embed_dim * 2)
        self.skip_proj1 = nn.Linear(embed_dim * 2, embed_dim)
        self.dec_blocks1 = nn.ModuleList([
            SwinTransformerBlock3D(embed_dim, num_heads, window_size, shift_size=0),
            SwinTransformerBlock3D(embed_dim, num_heads, window_size,
                                  shift_size=window_size // 2),
        ])

        # Završni upsampling natrag na originalnu rezoluciju
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, embed_dim * patch_size ** 3)
        self.final_conv = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, D_orig, H_orig, W_orig = x.shape

        # Patch embedding
        x, D, H, W = self.patch_embed(x)

        # Enkoder razina 1
        for block in self.enc_blocks1:
            x = block(x, D, H, W)
        skip1 = x
        x, D, H, W = self.downsample1(x, D, H, W)

        # Enkoder razina 2
        for block in self.enc_blocks2:
            x = block(x, D, H, W)
        skip2 = x
        x, D, H, W = self.downsample2(x, D, H, W)

        # Bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, D, H, W)

        # Dekoder razina 2
        x, D, H, W = self.upsample2(x, D, H, W)
        x = torch.cat([x, skip2], dim=-1)
        x = self.skip_proj2(x)
        for block in self.dec_blocks2:
            x = block(x, D, H, W)

        # Dekoder razina 1
        x, D, H, W = self.upsample1(x, D, H, W)
        x = torch.cat([x, skip1], dim=-1)
        x = self.skip_proj1(x)
        for block in self.dec_blocks1:
            x = block(x, D, H, W)

        # Završni upsampling natrag na originalnu rezoluciju
        x = self.final_norm(x)
        B, L, C = x.shape
        ps = self.patch_size
        x = self.final_proj(x)  # (B, L, C * ps^3)
        x = x.view(B, D, H, W, ps, ps, ps, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, D * ps, H * ps, W * ps)

        # Prilagodi na originalnu rezoluciju
        if x.shape[2] != D_orig or x.shape[3] != H_orig or x.shape[4] != W_orig:
            x = F.interpolate(x, size=(D_orig, H_orig, W_orig), mode="trilinear",
                              align_corners=False)

        return self.final_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
