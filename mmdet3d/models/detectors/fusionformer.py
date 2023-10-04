# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from mmdet3d.models.utils.bricks import run_time


@DETECTORS.register_module()
class FusionFormer(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 fusion_layer=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FusionFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # x = self.pts_backbone(x)
        # if self.with_pts_neck:
        #     x = self.pts_neck(x)
        return x

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # feature extract, ## debug
        # img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        # print('img: ', img.shape)  # img:  torch.Size([2, 6, 3, 900, 1600])
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None 
        
        # fusion
        ## debug
        x = self.fusion_layer(pts_feats, img_feats, img_metas)

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        # y = x[0] + x[1]
        bev_embed = F.interpolate(x[0], size=(200, 200), mode='bilinear')[:, 0: 256, :, :]
        # x = self.bev_conv(x)

        # head
        outs = self.pts_bbox_head(bev_embed, img_metas=img_metas)

        # loss
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses
        
    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        # feature extract, ## debug
        # img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None 
        
        # fusion
        ## debug
        x = self.fusion_layer(pts_feats, img_feats, img_metas)

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        # y = x[0] + x[1]
        bev_embed = F.interpolate(x[0], size=(200, 200), mode='bilinear')[:, 0: 256, :, :]
        # x = self.bev_conv(x)
        # print('bev_embed:')
        # print(bev_embed.shape)
        # print(bev_embed)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(bev_embed, img_metas, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def simple_test_pts(self, bev_embed, img_metas, rescale=False):
        """Test function of point cloud branch."""
        # head
        outs = self.pts_bbox_head(bev_embed, img_metas=img_metas)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
