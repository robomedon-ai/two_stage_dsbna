"""
TransUNet arhitektura za segmentaciju prostate na MR slikama.

Referenca: Chen et al., "TransUNet: Transformers Make Strong Encoders for
Medical Image Segmentation", arXiv 2021.

Hibridna arhitektura koja kombinira CNN enkoder s Vision Transformer u
bottlenecku za hvatanje i lokalnih i globalnih kontekstualnih informacija.
"""

import math

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


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mehanizam."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer blok: LayerNorm -> MHSA -> LayerNorm -> MLP"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNet(nn.Module):
    """
    TransUNet za binarnu segmentaciju.

    CNN enkoder izvlači značajke, zatim se feature mapa pretvara u sekvencu
    patcheva i prolazi kroz Transformer blokove u bottlenecku.
    Dekoder koristi skip konekcije iz CNN enkodera.

    Ulaz: (B, C, H, W)
    Izlaz: (B, 1, H, W) logiti (prije sigmoide)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32, num_transformer_layers: int = 6,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        f = base_filters

        # CNN Enkoder
        self.enc1 = ConvBlock2D(in_channels, f)
        self.enc2 = ConvBlock2D(f, f * 2)
        self.enc3 = ConvBlock2D(f * 2, f * 4)
        self.enc4 = ConvBlock2D(f * 4, f * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck konvolucija prije transformera
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(inplace=True),
        )

        # Transformer u bottlenecku
        embed_dim = f * 8
        # Osiguraj da embed_dim bude djeljiv s num_heads
        if embed_dim % num_heads != 0:
            num_heads = 4
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # Projekcija natrag u prostornu domenu
        self.bottleneck_out = ConvBlock2D(f * 8, f * 16)

        # Dekoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock2D(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2D(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN Enkoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck - priprema za transformer
        b = self.pool(e4)
        b = self.bottleneck_conv(b)

        # Pretvori feature mapu u sekvencu: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = b.shape
        b_seq = b.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Pozicijski embedding (sinusoidni, implicitno naučen kroz transformer)
        pos_embed = self._get_positional_encoding(H * W, C, b.device)
        b_seq = b_seq + pos_embed

        # Transformer blokovi
        for block in self.transformer_blocks:
            b_seq = block(b_seq)
        b_seq = self.transformer_norm(b_seq)

        # Vrati u prostornu domenu: (B, H*W, C) -> (B, C, H, W)
        b = b_seq.transpose(1, 2).reshape(B, C, H, W)
        b = self.bottleneck_out(b)

        # Dekoder sa skip konekcijama
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    def _get_positional_encoding(self, num_positions: int, embed_dim: int,
                                 device: torch.device) -> torch.Tensor:
        """Generira sinusoidni pozicijski encoding."""
        pe = torch.zeros(num_positions, embed_dim, device=device)
        position = torch.arange(0, num_positions, dtype=torch.float,
                                device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, N, C)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
