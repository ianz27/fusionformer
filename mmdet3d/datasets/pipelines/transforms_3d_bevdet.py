# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import cv2
import numpy as np
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from mmdet3d.datasets.pipelines.compose import Compose
from mmdet.datasets.pipelines import RandomCrop, RandomFlip, Rotate
from ..builder import OBJECTSAMPLERS, PIPELINES
from .data_augment_utils import noise_per_object_v3_


@PIPELINES.register_module()
class MultiViewWrapper(object):
    """Wrap transformation from single-view into multi-view.

    The wrapper processes the images from multi-view one by one. For each
    image, it constructs a pseudo dict according to the keys specified by the
    'process_fields' parameter. After the transformation is finished, desired
    information can be collected by specifying the keys in the 'collected_keys'
    parameter. Multi-view images share the same transformation parameters
    but do not share the same magnitude when a random transformation is
    conducted.

    Args:
        transforms (list[dict]): A list of dict specifying the transformations
            for the monocular situation.
        process_fields (dict): Desired keys that the transformations should
            be conducted on. Default to dict(img_fields=['img']).
        collected_keys (list[str]): Collect information in transformation
            like rotate angles, crop roi, and flip state.
    """

    def __init__(self,
                 transforms,
                 process_fields=dict(img_fields=['img']),
                 collected_keys=[]):
        self.transform = Compose(transforms)
        self.collected_keys = collected_keys
        self.process_fields = process_fields

    def __call__(self, input_dict):
        for key in self.collected_keys:
            input_dict[key] = []
        for img_id in range(len(input_dict['img'])):
            process_dict = self.process_fields.copy()
            for field in self.process_fields:
                for key in self.process_fields[field]:
                    process_dict[key] = input_dict[key][img_id]
            process_dict = self.transform(process_dict)
            for field in self.process_fields:
                for key in self.process_fields[field]:
                    input_dict[key][img_id] = process_dict[key]
            for key in self.collected_keys:
                input_dict[key].append(process_dict[key])
        return input_dict


@PIPELINES.register_module()
class RangeLimitedRandomCrop(RandomCrop):
    """Randomly crop image-view objects under a limitation of range.

    Args:
        relative_x_offset_range (tuple[float]): Relative range of random crop
            in x direction. (x_min, x_max) in [0, 1.0]. Default to (0.0, 1.0).
        relative_y_offset_range (tuple[float]): Relative range of random crop
            in y direction. (y_min, y_max) in [0, 1.0]. Default to (0.0, 1.0).
    """

    def __init__(self,
                 relative_x_offset_range=(0.0, 1.0),
                 relative_y_offset_range=(0.0, 1.0),
                 **kwargs):
        super(RangeLimitedRandomCrop, self).__init__(**kwargs)
        for range in [relative_x_offset_range, relative_y_offset_range]:
            assert 0 <= range[0] <= range[1] <= 1
        self.relative_x_offset_range = relative_x_offset_range
        self.relative_y_offset_range = relative_y_offset_range

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images.

        Modified from RandomCrop in mmdet==2.25.0

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_range_h = (margin_h * self.relative_y_offset_range[0],
                              margin_h * self.relative_y_offset_range[1] + 1)
            offset_h = np.random.randint(*offset_range_h)
            offset_range_w = (margin_w * self.relative_x_offset_range[0],
                              margin_w * self.relative_x_offset_range[1] + 1)
            offset_w = np.random.randint(*offset_range_w)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
            results['crop'] = (crop_x1, crop_y1, crop_x2, crop_y2)
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results


@PIPELINES.register_module()
class RandomRotate(Rotate):
    """Randomly rotate images.

    The ratation angle is selected uniformly within the interval specified by
    the 'range'  parameter.

    Args:
        range (tuple[float]): Define the range of random rotation.
            (angle_min, angle_max) in angle.
    """

    def __init__(self, range, **kwargs):
        super(RandomRotate, self).__init__(**kwargs)
        self.range = range

    def __call__(self, results):
        self.angle = np.random.uniform(self.range[0], self.range[1])
        super(RandomRotate, self).__call__(results)
        results['rotate'] = self.angle
        return results


@PIPELINES.register_module()
class AffineResize(object):
    """Get the affine transform matrices to the target size.

    Different from :class:`RandomAffine` in MMDetection, this class can
    calculate the affine transform matrices while resizing the input image
    to a fixed size. The affine transform matrices include: 1) matrix
    transforming original image to the network input image size. 2) matrix
    transforming original image to the network output feature map size.

    Args:
        img_scale (tuple): Images scales for resizing.
        down_ratio (int): The down ratio of feature map.
            Actually the arg should be >= 1.
        bbox_clip_border (bool, optional): Whether clip the objects
            outside the border of the image. Defaults to True.
    """

    def __init__(self, img_scale, down_ratio, bbox_clip_border=True):

        self.img_scale = img_scale
        self.down_ratio = down_ratio
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, results):
        """Call function to do affine transform to input image and labels.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after affine resize, 'affine_aug', 'trans_mat'
                keys are added in the result dict.
        """
        # The results have gone through RandomShiftScale before AffineResize
        if 'center' not in results:
            img = results['img']
            height, width = img.shape[:2]
            center = np.array([width / 2, height / 2], dtype=np.float32)
            size = np.array([width, height], dtype=np.float32)
            results['affine_aug'] = False
        else:
            # The results did not go through RandomShiftScale before
            # AffineResize
            img = results['img']
            center = results['center']
            size = results['size']

        trans_affine = self._get_transform_matrix(center, size, self.img_scale)

        img = cv2.warpAffine(img, trans_affine[:2, :], self.img_scale)

        if isinstance(self.down_ratio, tuple):
            trans_mat = [
                self._get_transform_matrix(
                    center, size,
                    (self.img_scale[0] // ratio, self.img_scale[1] // ratio))
                for ratio in self.down_ratio
            ]  # (3, 3)
        else:
            trans_mat = self._get_transform_matrix(
                center, size, (self.img_scale[0] // self.down_ratio,
                               self.img_scale[1] // self.down_ratio))

        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['trans_mat'] = trans_mat

        self._affine_bboxes(results, trans_affine)

        if 'centers2d' in results:
            centers2d = self._affine_transform(results['centers2d'],
                                               trans_affine)
            valid_index = (centers2d[:, 0] >
                           0) & (centers2d[:, 0] <
                                 self.img_scale[0]) & (centers2d[:, 1] > 0) & (
                                     centers2d[:, 1] < self.img_scale[1])
            results['centers2d'] = centers2d[valid_index]

            for key in results.get('bbox_fields', []):
                if key in ['gt_bboxes']:
                    results[key] = results[key][valid_index]
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][
                            valid_index]
                    if 'gt_masks' in results:
                        raise NotImplementedError(
                            'AffineResize only supports bbox.')

            for key in results.get('bbox3d_fields', []):
                if key in ['gt_bboxes_3d']:
                    results[key].tensor = results[key].tensor[valid_index]
                    if 'gt_labels_3d' in results:
                        results['gt_labels_3d'] = results['gt_labels_3d'][
                            valid_index]

            results['depths'] = results['depths'][valid_index]

        return results

    def _affine_bboxes(self, results, matrix):
        """Affine transform bboxes to input image.

        Args:
            results (dict): Result dict from loading pipeline.
            matrix (np.ndarray): Matrix transforming original
                image to the network input image size.
                shape: (3, 3)
        """

        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            bboxes[:, :2] = self._affine_transform(bboxes[:, :2], matrix)
            bboxes[:, 2:] = self._affine_transform(bboxes[:, 2:], matrix)
            if self.bbox_clip_border:
                bboxes[:,
                       [0, 2]] = bboxes[:,
                                        [0, 2]].clip(0, self.img_scale[0] - 1)
                bboxes[:,
                       [1, 3]] = bboxes[:,
                                        [1, 3]].clip(0, self.img_scale[1] - 1)
            results[key] = bboxes

    def _affine_transform(self, points, matrix):
        """Affine transform bbox points to input image.

        Args:
            points (np.ndarray): Points to be transformed.
                shape: (N, 2)
            matrix (np.ndarray): Affine transform matrix.
                shape: (3, 3)

        Returns:
            np.ndarray: Transformed points.
        """
        num_points = points.shape[0]
        hom_points_2d = np.concatenate((points, np.ones((num_points, 1))),
                                       axis=1)
        hom_points_2d = hom_points_2d.T
        affined_points = np.matmul(matrix, hom_points_2d).T
        return affined_points[:, :2]

    def _get_transform_matrix(self, center, scale, output_scale):
        """Get affine transform matrix.

        Args:
            center (tuple): Center of current image.
            scale (tuple): Scale of current image.
            output_scale (tuple[float]): The transform target image scales.

        Returns:
            np.ndarray: Affine transform matrix.
        """
        # TODO: further add rot and shift here.
        src_w = scale[0]
        dst_w = output_scale[0]
        dst_h = output_scale[1]

        src_dir = np.array([0, src_w * -0.5])
        dst_dir = np.array([0, dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2, :] = self._get_ref_point(src[0, :], src[1, :])
        dst[2, :] = self._get_ref_point(dst[0, :], dst[1, :])

        get_matrix = cv2.getAffineTransform(src, dst)

        matrix = np.concatenate((get_matrix, [[0., 0., 1.]]))

        return matrix.astype(np.float32)

    def _get_ref_point(self, ref_point1, ref_point2):
        """Get reference point to calculate affine transform matrix.

        While using opencv to calculate the affine matrix, we need at least
        three corresponding points separately on original image and target
        image. Here we use two points to get the the third reference point.
        """
        d = ref_point1 - ref_point2
        ref_point3 = ref_point2 + np.array([-d[1], d[0]])
        return ref_point3

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'down_ratio={self.down_ratio}) '
        return repr_str


@PIPELINES.register_module()
class RandomShiftScale(object):
    """Random shift scale.

    Different from the normal shift and scale function, it doesn't
    directly shift or scale image. It can record the shift and scale
    infos into loading pipelines. It's designed to be used with
    AffineResize together.

    Args:
        shift_scale (tuple[float]): Shift and scale range.
        aug_prob (float): The shifting and scaling probability.
    """

    def __init__(self, shift_scale, aug_prob):

        self.shift_scale = shift_scale
        self.aug_prob = aug_prob

    def __call__(self, results):
        """Call function to record random shift and scale infos.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after random shift and scale, 'center', 'size'
                and 'affine_aug' keys are added in the result dict.
        """
        img = results['img']

        height, width = img.shape[:2]

        center = np.array([width / 2, height / 2], dtype=np.float32)
        size = np.array([width, height], dtype=np.float32)

        if random.random() < self.aug_prob:
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)
            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)
            results['affine_aug'] = True
        else:
            results['affine_aug'] = False

        results['center'] = center
        results['size'] = size

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(shift_scale={self.shift_scale}, '
        repr_str += f'aug_prob={self.aug_prob}) '
        return repr_str
