import torch
from torch import nn as nn
import torch.nn.functional as F

from ..builder import FUSION_LAYERS
from .lss import LSSTransform


@FUSION_LAYERS.register_module()
class BEVFusion(nn.Module):
    """Fuse 2d features from 3d seeds.

    """

    def __init__(self, 
                 in_channels=512, 
                 out_channels=256, 
                 out_h=None, 
                 out_w=None,
                 vtransform_feat_level=0,
                 vtransform=None,
                 ):
        super(BEVFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_h = out_h
        self.out_w = out_w

        # vtransform
        # self.vtransform = None
        self.vtransform_feat_level = vtransform_feat_level
        self.vtransform = LSSTransform(**vtransform)

        self.bev_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    )

    def forward(self, pts_feats=None, img_feats=None, img_metas=None):
        if pts_feats is not None:
            pts_feat = pts_feats[0]
            pts_feat = F.interpolate(pts_feat, size=(self.out_h, self.out_w), mode='bilinear')
            pts_feat = self.bev_conv(pts_feat)

            return pts_feat

        if img_feats is not None:
            # ## debug
            # for feat in img_feats:
            #     print('img_feat: ', feat.shape)  # img_feat:  torch.Size([12, 256, 29, 50])
            
            img_feat = img_feats[self.vtransform_feat_level]
            BN, C, H, W = img_feat.size()
            N = 6  # TODO: hard coding
            B = BN // N
            img_feat = img_feat.view(B, N, C, H, W)
            dtype = img_feat.dtype
            device = img_feat.device
            camera_intrinsics = torch.tensor([img_meta['cam_intrinsic'] for img_meta in img_metas], dtype=dtype).to(device)
            camera2lidar = torch.tensor([img_meta['cam2lidar'] for img_meta in img_metas], dtype=dtype).to(device)
            img_aug_matrix = torch.tensor([img_meta['img_aug_matrix'] for img_meta in img_metas], dtype=dtype).to(device)
            # print(camera_intrinsics.shape)
            # print(camera_intrinsics)
            # print(camera2lidar.shape)
            # exit()

            # sensor2ego:  torch.float32 torch.Size([1, 6, 4, 4])
            # lidar2ego:  torch.float32 torch.Size([1, 4, 4])
            # lidar2camera:  torch.float32 torch.Size([1, 6, 4, 4])
            # lidar2image:  torch.float32 torch.Size([1, 6, 4, 4])
            # cam_intrinsic:  torch.float32 torch.Size([1, 6, 4, 4])
            # camera2lidar:  torch.float32 torch.Size([1, 6, 4, 4])
            # img_aug_matrix:  torch.float32 torch.Size([1, 6, 4, 4])
            # lidar_aug_matrix:  torch.float32 torch.Size([1, 4, 4])
            # vtransform
            vtransform_feat = self.vtransform(
                img=img_feat,
                points=None,
                radar=None,
                camera2ego=torch.zeros((4, 4)).expand(B, N, -1, -1).to(device),  # unused
                lidar2ego=torch.zeros((4, 4)).expand(B, -1, -1).to(device),  # unused
                lidar2camera=None,
                lidar2image=None,
                camera_intrinsics=camera_intrinsics,
                camera2lidar=camera2lidar,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=torch.eye(4).expand(B, -1, -1).to(device),
                img_metas=img_metas,
                depth_loss=None, 
                gt_depths=None,
            )
            # ## debug
            # print('img_aug_matrix: ', img_aug_matrix)
        
            # ## debug
            # print('vtransform_feat: ', vtransform_feat.shape)  # vtransform_feat:  torch.Size([2, 256, 200, 200])
            # x = self.pts_backbone(x)
            # x = self.pts_neck(x)

            # vtransform_feat = F.interpolate(vtransform_feat, size=(self.out_h, self.out_w), mode='bilinear')
            # vtransform_feat = self.bev_conv(vtransform_feat)

            return vtransform_feat


