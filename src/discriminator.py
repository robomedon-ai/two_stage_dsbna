"""
PatchGAN diskriminator za adversarial segmentaciju prostate.

Diskriminator prima par (slika, maska) i vraća mapu patch-level
vjerojatnosti da je maska "prava" (GT) ili "lažna" (predikcija modela).

Referenca: Isola et al., "Image-to-Image Translation with Conditional
Adversarial Networks" (pix2pix), CVPR 2017.
"""

import torch
import torch.nn as nn


class PatchDiscriminator2D(nn.Module):
    """
    2D PatchGAN diskriminator.

    Ulaz: konkatenacija slike i maske (B, C_in+1, H, W)
    Izlaz: (B, 1, H', W') mapa patch-level vjerojatnosti
    """

    def __init__(self, in_channels: int = 2, base_filters: int = 64,
                 n_layers: int = 3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2,
                      padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = base_filters
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1,
                          bias=False),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Zadnji sloj — stride 1
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1,
                      bias=False),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Izlazni sloj — 1 kanal
        layers.append(
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W) ulazna slika
            mask: (B, 1, H, W) segmentacijska maska (GT ili predikcija)
        Returns:
            (B, 1, H', W') patch-level logiti
        """
        x = torch.cat([image, mask], dim=1)
        return self.model(x)


class PatchDiscriminator3D(nn.Module):
    """
    3D PatchGAN diskriminator za volumetrijske podatke.

    Ulaz: konkatenacija volumena i maske (B, C_in+1, D, H, W)
    Izlaz: (B, 1, D', H', W') mapa patch-level vjerojatnosti
    """

    def __init__(self, in_channels: int = 2, base_filters: int = 32,
                 n_layers: int = 3):
        super().__init__()

        layers = [
            nn.Conv3d(in_channels, base_filters, kernel_size=4, stride=2,
                      padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = base_filters
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 256)
            layers += [
                nn.Conv3d(nf_prev, nf, kernel_size=4, stride=2, padding=1,
                          bias=False),
                nn.InstanceNorm3d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_prev = nf
        nf = min(nf * 2, 256)
        layers += [
            nn.Conv3d(nf_prev, nf, kernel_size=4, stride=1, padding=1,
                      bias=False),
            nn.InstanceNorm3d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        layers.append(
            nn.Conv3d(nf, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask], dim=1)
        return self.model(x)
