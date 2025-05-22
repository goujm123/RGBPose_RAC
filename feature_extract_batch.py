import numpy as np
import os
import torch
import cv2
from preprocess_data_loader import Rep_count
from decord import VideoReader, cpu, gpu
from video_mae_cross_full_attention import SupervisedMAE
from util.config import load_config
import argparse
import tqdm
from torchvision.transforms import Resize, CenterCrop, Normalize, Compose

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

from mediapipe.python.solutions import pose as mp_pose
import einops

torch.manual_seed(0)

BLAZEPOSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7),
                                   (0, 4), (4, 5), (5, 6), (6, 8),
                                   (9, 10),
                                   (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                                   (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)])

# 生成的 npz 文件存放位置
target_dir = 'd:/datasets/RepCount/tokens_batch'
# 窗口覆盖 64帧，从窗口中采样 16帧 构成一个 clip（或 segment）
window_size = 64
sample_frames = 16
# 根据 GPU内存情况，进行批处理。设置 16个clip构成一个chunk，但最后一个chunk可能不足16个clip
chunk_size = 16


def get_args_parser():
    parser = argparse.ArgumentParser('MAE encoding', add_help=False)
    parser.add_argument('--use_v1', default=False, help='use the v1 variant of the encoder')
    parser.add_argument('--config', default='configs/pretrain_config.yaml', help="config file")

    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--pretrained_encoder', default='pretrained_models/VIT_B_16x4_MAE_PT.pth', type=str)
    parser.add_argument('--dataset', default='RepCount', help='choose from [RepCount, Countix, UCFRep]', type=str)
    parser.add_argument('--model', default='VideoMAE', help="VideoMAE, VideoSwin")
    parser.add_argument('--encodings', default='mae', help="mae, swin, resnext")
    parser.add_argument('--data_path', default='D:/datasets/RepCount/video', help='data path for the dataset')
    return parser


def preprocess(tensor, min_size=224, crop_size=224, video_mean=[0.485, 0.456, 0.406], video_std=[0.229, 0.224, 0.225]):
    T, C, H, W = tensor.shape

    crop = CenterCrop(crop_size)
    resize = Resize(min_size)
    normalize = Normalize(mean=video_mean, std=video_std)

    data = normalize(tensor)
    data = resize(data)
    data = crop(data)

    return data


def preprocess_pose(tensor, min_size=224, crop_size=224):
    T, C, H, W = tensor.shape

    crop = CenterCrop(crop_size)
    resize = Resize(min_size)

    data = resize(tensor)
    data = crop(data)

    return data


def draw_blazepose_landmarks(image, landmarks, draw_connections=True):
    """
    在RGB图像上绘制BlazePose关键点和骨架连线
    :param image: 原始RGB图像 (H, W, 3)
    :param landmarks: 33个landmark的list, 每个为[x, y, z, visibility]
    :param draw_connections: 是否绘制骨架连线
    :return: 带可视化的图像
    """
    h = image.shape[0]
    w = image.shape[1]
    img = np.zeros((h, w, 3), dtype=np.uint8)  # 当前是三通道
    landmarks = np.array(landmarks)

    # 绘制关键点
    for idx, landmark in enumerate(landmarks):
        x, y, z, v = landmark
        if v < 0.5:
            continue
        cv2.circle(img, (int(x), int(y)), radius=3, color=[255, 255, 255], thickness=-1)

    # 绘制骨架连线
    if draw_connections:
        for start, end in BLAZEPOSE_CONNECTIONS:
            v_start = landmarks[start][3]
            v_end = landmarks[end][3]
            if v_start < 0.5 or v_end < 0.5:
                continue
            x1, y1 = int(landmarks[start][0]), int(landmarks[start][1])
            x2, y2 = int(landmarks[end][0]), int(landmarks[end][1])
            cv2.line(img, (x1, y1), (x2, y2), [255, 255, 255], 5)

    return img


def pose_feature_extract(clip, pose_estimator):
    B, C, H, W = clip.shape
    images = []

    # 这个 pose_estimator 一次只能处理一帧
    for i in range(clip.shape[0]):
        frame = clip[i, :]  # C, H, W
        frame = frame.permute(1, 2, 0).numpy()  # H, W, C
        pose_result = pose_estimator.process(image=frame)
        pose_landmarks = pose_result.pose_landmarks

        # 如果有 pose landmark，则转换。否者，返回空白帧
        if pose_landmarks:
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            # landmarks 维度：33*4
            landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width, lmk.visibility] for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(landmarks.shape)
            vis_img = draw_blazepose_landmarks(frame, landmarks)  # ndarray
            images.append(vis_img)

            # 可视化检查
            # cv2.imshow("BlazePose Visualization", vis_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            images.append(np.zeros((H, W, C), dtype=np.uint8))

    # 为了对接后续的 Compose 处理流程，需要调整维度格式(N,H,W,C-->N,C,H,W)
    result = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)

    return result


def pose_feature_extract_vitpose(clip):
    B, C, H, W = clip.shape
    images = []
    


def save_tokens(dataloaders, model, args):
    '''
    This function extracts the encodings for each video using windows of 64 frames and then sampling 16 frames uniformly from these windows.
    We save the encodings in npz format. The input to the encoder is B*3x16xHxW, where B is the batch size and each batch comprises overlapping windows in each video.
    The output is spatio-temporal tokens of shape Bx(T'H'W')xC. We save these encodings as BxCxT'xH'xW'.
    inputs: a dict consisting of 'train', 'val' and 'test' dataloaders,
         the pretrained model,
         other parameters needed
    '''

    splits = ['train', 'val', 'test']

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # cuda内存使用统计复位
    # torch.cuda.reset_peak_memory_stats()

    pose_estimator = mp_pose.Pose()
    if pose_estimator is None:
        raise RuntimeError("create pose estimator failed")

    for split in splits:
        for item in tqdm.tqdm(dataloaders[split], total=len(dataloaders[split])):
            C, Total, H, W = item[0]  # Total: # of total frames

            if Total == 0:
                continue

            video_name = item[-1][0]
            base_name = os.path.basename(video_name)[:-4]
            print(f"{split}: {base_name}, total frames: {Total.numpy()}")

            # 如果之前已经转换完成一部分，则保留，从剩下的开始处理
            if os.path.exists('{}/{}_rgb.npz'.format(target_dir, base_name)) and os.path.exists('{}/{}_pose.npz'.format(target_dir, base_name)):
                continue

            vr = VideoReader(video_name, ctx=cpu(0))
            rgb_encoded = []
            pose_encoded = []

            rgb_chunk = []
            pose_chunk = []

            for j in range(0, Total, 16):  #### 75% overlap， 每个窗口覆盖64帧，滑动窗口step为16
                idx = np.linspace(j, j + window_size, sample_frames + 1)[:sample_frames].astype(int)  ### sample 16 frames from windows of 64 frames

                valid_idx = [i for i in idx if i < Total]
                padding_idx = [i for i in idx if i >= Total]
                batch_clip = vr.get_batch(valid_idx).asnumpy()  # (B, H, W, 3)

                clip_valid = torch.from_numpy(batch_clip)  # (B, H, W, C)
                clip_valid = clip_valid.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
                clip_padding = torch.zeros([len(padding_idx), C, H, W]).to(dtype=torch.uint8)

                clip = torch.cat((clip_valid, clip_padding), dim=0) if len(padding_idx) > 0 else clip_valid

                # pose_feature_extract（注意 BlazePose一次只能处理一帧RGB图像）
                pose_input = pose_feature_extract(clip, pose_estimator)
                pose_input = preprocess_pose(pose_input / 255.)
                pose_input = pose_input.permute(1, 0, 2, 3)  # (T,C,H,W-->C,T,H,W)

                # rgb_feature_extract
                rgb_input = preprocess(clip / 255.)  # Norm, resize & crop valid frames first. shape: H*W -> 224*224
                rgb_input = rgb_input.permute(1, 0, 2, 3)

                # 批量处理提高效率，一个chunk包含多个clip
                rgb_chunk.append(rgb_input)
                pose_chunk.append(pose_input)

                if len(rgb_chunk) >= chunk_size or j + window_size >= Total:
                    rgb_input = torch.stack(rgb_chunk, dim=0).to("cuda")  # (C,T,H,W-->B,C,T,H,W)
                    pose_input = torch.stack(pose_chunk, dim=0).to("cuda")

                    # 使用 VideoMAE 预训练模型分别对两种输入编码
                    with torch.no_grad():
                        try:
                            encoded_pose, thw_pose = model(pose_input)
                            encoded_rgb, thw_rgb = model(rgb_input)
                        except:
                            print(f"VideoMAE encoding: {video_name}, exception")
                            break

                        encoded_rgb = encoded_rgb.transpose(1, 2).reshape(encoded_rgb.shape[0], encoded_rgb.shape[-1], thw_rgb[0], thw_rgb[1], thw_rgb[2])  # reshape to B x C x T x H x W
                        encoded_pose = encoded_pose.transpose(1, 2).reshape(encoded_pose.shape[0], encoded_pose.shape[-1], thw_pose[0], thw_pose[1], thw_pose[2])  # reshape to B x C x T x H x W

                    rgb_encoded.append(encoded_rgb.cpu().numpy())
                    pose_encoded.append(encoded_pose.cpu().numpy())

                    rgb_chunk.clear()
                    pose_chunk.clear()

                torch.cuda.empty_cache()
                del clip, clip_valid, clip_padding, batch_clip, pose_input, rgb_input

            # 按照 float存放，每个clip的16帧图像数据编码为 (768,8,14,14)，约4.8MB
            merged = np.concatenate(rgb_encoded, 0)
            np.savez('{}/{}_rgb.npz'.format(target_dir, base_name), merged)  ### saving as npz

            merged = np.concatenate(pose_encoded, 0)
            np.savez('{}/{}_pose.npz'.format(target_dir, base_name), merged)

            # max_reserved = torch.cuda.max_memory_reserved()
            # max_alloc = torch.cuda.max_memory_allocated()
            # print(f" {video_name}: shape[{C}, {T}, {H}, {W}], max reserved: {max_reserved / 1e9:.2f} GB, max allocated: {max_alloc / 1e9:.2f} GB")
            # torch.cuda.reset_peak_memory_stats()


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.opts = None

    cfg = load_config(args)

    model = SupervisedMAE(cfg=cfg, just_encode=True, use_precomputed=False).cuda()
    if args.pretrained_encoder:
        state_dict = torch.load(args.pretrained_encoder)
        if 'model_state' in state_dict.keys():
            state_dict = state_dict['model_state']
        else:
            state_dict = state_dict['model']
    else:
        print("You should download VIT_B_16x4_MAE_PT.pyth manually.")

    for name in model.state_dict().keys():
        if 'decode' in name:
            continue
        matched = 0

        for name_, param in state_dict.items():
            if name_ == name:
                model.state_dict()[name].copy_(param)
                matched = 1
                break
            elif '.qkv.' in name and 'blocks' in name:
                q_name = name.replace('.qkv.', '.q.').replace('module.', '')
                k_name = name.replace('.qkv.', '.k.').replace('module.', '')
                v_name = name.replace('.qkv.', '.v.').replace('module.', '')
                params = torch.cat([state_dict[q_name], state_dict[k_name], state_dict[v_name]])
                model.state_dict()[name].copy_(params)
                matched = 1
                break

    model.eval()

    if args.dataset == 'RepCount':
        dataset_train = Rep_count(cfg=cfg, split="train", data_dir=args.data_path, sampling_interval=1, encode_only=True)
        dataset_val = Rep_count(cfg=cfg, split="valid", data_dir=args.data_path, sampling_interval=1, encode_only=True)
        dataset_test = Rep_count(cfg=cfg, split="test", data_dir=args.data_path, sampling_interval=1, encode_only=True)

    dataloaders = {'train': torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                        num_workers=1,
                                                        shuffle=False,
                                                        pin_memory=True,
                                                        drop_last=False),
                   'val': torch.utils.data.DataLoader(dataset_val,
                                                      batch_size=args.batch_size,
                                                      num_workers=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      drop_last=False),
                   'test': torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=args.batch_size,
                                                       num_workers=1,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=False),
                   }

    save_tokens(dataloaders, model, args)


if __name__ == '__main__':
    main()
