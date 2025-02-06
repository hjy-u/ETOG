import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Fusion(nn.Module):
    """ Base Fusion Class"""

    def __init__(self, input_dim=3):
        super().__init__()
        self.input_dim = input_dim

    def tile_x2(self, x1, x2, x2_proj=None):
        if x2_proj:
            x2 = x2_proj(x2)

        x2 = x2.unsqueeze(-1).unsqueeze(-1)
        # x2 = x2.repeat(x1.shape[0], 1, x1.shape[-2], x1.shape[-1])
        return x2

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        raise NotImplementedError()


class FusionMult(Fusion):
    """ x1 * x2 """

    def __init__(self, input_dim=3):
        super(FusionMult, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return x1 * x2


class FusionAdd(Fusion):
    """ x1 + x2 """

    def __init__(self, input_dim=3):
        super(FusionAdd, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return x1 + x2


class FusionMax(Fusion):
    """ max(x1, x2) """

    def __init__(self, input_dim=3):
        super(FusionMax, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return torch.max(x1, x2)


class FusionConcat(Fusion):
    """ [x1; x2] """

    def __init__(self, input_dim=3):
        super(FusionConcat, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return torch.cat([x1, x2], dim=1)


class FusionConv(Fusion):
    """ 1x1 convs after [x1; x2] """

    def __init__(self, input_dim=3):
        super(FusionConv, self).__init__(input_dim=input_dim)
        self.conv = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1, bias=False)
        )

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        x = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]
        x = self.conv(x)                # [B, C, H, W]
        return x


class FusionConvLat(Fusion):
    """ 1x1 convs after [x1; x2] for lateral fusion """

    def __init__(self, input_dim=3, output_dim=3):
        super(FusionConvLat, self).__init__(input_dim=input_dim)
        self.conv = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        )

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        x = torch.cat([x1, x2], dim=1)  # [B, input_dim, H, W]
        x = self.conv(x)                # [B, output_dim, H, W]
        return x

names = {
    'add': FusionAdd,
    'mult': FusionMult,
    'max': FusionMax,
    'concat': FusionConcat,
    'conv': FusionConv
}