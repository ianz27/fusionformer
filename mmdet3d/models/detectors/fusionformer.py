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
                 use_grid_mask=None,
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
                 aux_head=None,
                 aux_weight=0.5,
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
        
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

        if aux_head is not None:
            aux_head.update(train_cfg=train_cfg.pts)
            aux_head.update(test_cfg=test_cfg.pts)
            self.aux_head = builder.build_head(aux_head)
            self.aux_weight = aux_weight

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
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
        # feature extract
        img_feats, pts_feats = self.extract_feat(points, img, img_metas=img_metas)
        # ## debug
        # # torch.Size([4, 256, 128, 128])
        # # torch.Size([4, 256, 64, 64])
        # # torch.Size([4, 256, 32, 32])
        # # torch.Size([4, 256, 16, 16])
        # for pts_feat in pts_feats:
        #     print(pts_feat.shape)
        # exit()

        # head
        # TODO: mvl feature
        outs = self.pts_bbox_head(pts_feats[0], img_feats, img_metas=img_metas)

        # loss
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        if self.aux_head is not None:
            # TODO: mvl feature
            aux_outs = self.aux_head(pts_feats[0], img_feats, img_metas=img_metas)
            aux_loss_inputs = [gt_bboxes_3d, gt_labels_3d, aux_outs]
            aux_losses = self.pts_bbox_head.aux_head.loss(*aux_loss_inputs)
            for k, v in aux_losses.items():
                losses[f'aux_{k}'] = v * self.aux_weight

        return losses
        
    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        # feature extract
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts_img(pts_feats[0], img_feats, img_metas, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def simple_test_pts_img(self, pts_feat, img_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        # head
        outs = self.pts_bbox_head(pts_feat, img_feats, img_metas=img_metas)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
