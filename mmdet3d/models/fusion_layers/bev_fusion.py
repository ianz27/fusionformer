import torch
from torch import nn as nn

from ..builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class BEVFusion(nn.Module):
    """Fuse 2d features from 3d seeds.

    """

    def __init__(self, in_channels=512, out_channels=256, out_h=None, out_w=None):
        super(BEVFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_h = out_h
        self.out_w = out_w

        self.bev_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    )

    def forward(self, pts_feats=None, img_feats=None):
        for pts_feat in pts_feats:
            

