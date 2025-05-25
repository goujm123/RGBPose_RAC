# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import cv2
import torch

from ViTpose.base_pose import BasePose
from ViTpose.vit_backbone import ViT
from ViTpose.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead


# Top-down pose detectors.
class TopDown(BasePose):

    def __init__(self, model_cfg):
        super().__init__()
        self.fp16_enabled = False

        self.test_cfg = model_cfg['test_cfg']

        # 使用传入的配置参数字典，删除掉 type 说明
        backbone_args = model_cfg['backbone']
        del backbone_args['type']
        self.backbone = ViT(**backbone_args)

        head_args = model_cfg['keypoint_head']
        del head_args['type']
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_args)

        self.init_weights(pretrained=None)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights()
        self.keypoint_head.init_weights()

    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas, **kwargs)

        return self.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)

        output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        keypoint_losses = self.keypoint_head.get_loss(
            output, target, target_weight)
        losses.update(keypoint_losses)

        keypoint_accuracy = self.keypoint_head.get_accuracy(
            output, target, target_weight)
        losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)

        output_heatmap = self.keypoint_head.inference_model(features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)

            output_flipped_heatmap = self.keypoint_head.inference_model(features_flipped, img_metas[0]['flip_pairs'])
            output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

        keypoint_result = self.keypoint_head.decode(
            img_metas, output_heatmap, img_size=[img_width, img_height])
        result.update(keypoint_result)

        if not return_heatmap:
            output_heatmap = None

        result['output_heatmap'] = output_heatmap

        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)

        output = self.keypoint_head(output)

        return output

    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            # imshow_bboxes(
            #     img,
            #     bboxes,
            #     labels=bbox_labels,
            #     colors=bbox_color,
            #     text_color=text_color,
            #     thickness=bbox_thickness,
            #     font_scale=font_scale,
            #     show=False)

        if pose_result:
            # imshow_keypoints(img, pose_result, skeleton, kpt_score_thr, pose_kpt_color, pose_link_color, radius, thickness)
            pass

        if show:
            cv2.imshow(win_name, img)

        if out_file is not None:
            cv2.imwrite(out_file, img)

        return img
