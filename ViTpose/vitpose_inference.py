import cv2
import numpy as np
import torch
from ViTpose.top_down import TopDown
from mmcv.runner import load_checkpoint

# 配置来源于 configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py ,
# 注意和使用的预训练模型 ViTPose_small_coco_256x192.pth 对应
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

model_cfg = dict(
    type='TopDown',
    pretrained="../pretrained_models/ViTPose_small_coco_256x192.pth",
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
    ),

    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=384,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),

    train_cfg=dict(),

    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type='GaussianHeatmap',
        modulate_kernel=11,
        use_udp=True))


def init_pose_model(filename=model_cfg['pretrained'], device='cpu'):
    model = TopDown(model_cfg)
    load_checkpoint(model, filename, map_location=device)

    model.to(device)
    model.eval()

    return model


def inference_test(pose_model, img, metas):
    # forward test
    with torch.no_grad():
        result = pose_model.forward(img, img_metas=metas, return_loss=False, return_heatmap=True)

    return result

if __name__ == '__main__':
    device = "cuda:0"
    model = init_pose_model(device=device)

    image_name = 'c:/coco/images/val2017/000000397133.jpg'
    img = cv2.imread(image_name).astype(np.float32) / 255.0  # to 0.0 ~ 1.0

    # image 的维度需要调整为 [1, 3, 256, 192]
    img = cv2.resize(img, (192, 256))
    img_tensor = torch.tensor(img, dtype=torch.float).permute((2, 0, 1)).unsqueeze(0).to(device)

    metas = [{
        'bbox_id': 0,
        'bbox_score': 1,
        'center': [442.865, 208.23],
        'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
        'image_file': 'c:/coco/images/val2017/000000397133.jpg',
        'rotation': 0,
        'scale': [1.2966563, 1.728875]
    }]

    inference_test(model, img_tensor, metas)

    print("inference done!")
