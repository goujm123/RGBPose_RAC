import pathlib
from random import randint
import torch.utils.data
import os, sys, math
import numpy as np
import pandas as pd
from numpy.f2py.auxfuncs import throw_error
from tqdm import tqdm
import random
from scipy import integrate
from scipy import ndimage

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import einops


class Rep_count(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 add_noise=False,
                 num_frames=512,
                 video_tokens_dir="./tokens",
                 pose_tokens_dir="./tokens",
                 select_rand_segment=True,
                 compact=False,
                 peak_at_random_location=False,
                 get_overlapping_segments=False,
                 density_peak_width=1.0,
                 threshold=0.0):

        self.num_frames = num_frames
        self.video_tokens_dir = video_tokens_dir
        self.pose_tokens_dir = pose_tokens_dir
        self.compact = compact
        self.select_rand_segment = select_rand_segment
        self.split = split  # set the split to load
        self.add_noise = add_noise  # add noise to frames (augmentation)
        self.peak_at_random_location = peak_at_random_location
        self.get_overlapping_segments = get_overlapping_segments
        self.density_peak_width = density_peak_width
        self.threshold = threshold  ## cutoff to decide if we should select exemplar from other videos of same class

        # ESCount使用的是自己处理过的标注文件，和原始标注有格式上的差别
        csv_path = f'datasets/repcount/{self.split}_with_fps.csv'

        self.df = pd.read_csv(csv_path)

        # 过滤掉不符合要求的文件
        self.df = self.df[self.df['count'].notna()]
        self.df = self.df[self.df['num_frames'] > 64]
        self.df = self.df[self.df['count'] > 0]  # remove no reps

        # 显存较小，超长视频训练时会导致 OOM（RTX2070 上能支持 30个分段， 1920帧），过滤掉超长的视频
        # self.df = self.df.drop(self.df.loc[self.df['num_frames'] > 1920].index)

        # 过滤标注错误（最常见的是count 或者 cycle标注错误）
        # train集
        self.df = self.df.drop(self.df.loc[self.df['name'] == 'stu1_10.mp4'].index)
        self.df = self.df.drop(self.df.loc[self.df['name'] == 'stu4_3.mp4'].index)
        self.df = self.df.drop(self.df.loc[self.df['name'] == 'test118.mp4'].index)

        # 指定文件，用于调试 (这个文件有 2000多帧)
        self.df = self.df[(self.df['name'] == 'stu1_22.mp4') | (self.df['name'] == 'stu2_57.mp4')]
        

        # test集（标注错误，但为了和别的论文对比，先保留着）
        # self.df = self.df.drop(self.df.loc[self.df['name'] == 'stu4_5.mp4'].index)

        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- ")

    def load_tokens(self, path, bounds=None, get_overlapping_segments=False):
        """
        loading video or exemplar tokens. 
        input: path -> the path for the saved video/exemplar tokens
               bounds -> (st, end) to trim video given the start and end timestamps.

        output:
               video/pose tokens
        """

        try:
            tokens = np.load(path)['arr_0']  # Load in format B x 768 x 8 x 14 x 14
        except:
            print(f'Could not load {path}')
            exit(-1)

        if bounds is not None:
            low_bound = bounds[0] // 8
            up_bound = math.ceil(bounds[1] / 8)

        if get_overlapping_segments:  # 默认不走这个
            if self.split != 'test':
                tokens1 = tokens[0::4]  ### concatenating tokens for non-overlapping windows
                tokens1 = einops.rearrange(tokens1, 'S C T H W -> C (S T) H W')
                tokens1 = tokens1[:, max(low_bound, 0): max(up_bound, 0)]
                tokens1 = torch.from_numpy(tokens1)
                tokens2 = None
            else:
                tokens1 = tokens[0::4]
                tokens2 = tokens[2::4]

                tokens1 = einops.rearrange(tokens1, 'S C T H W -> C (S T) H W')
                tokens2 = einops.rearrange(tokens2, 'S C T H W -> C (S T) H W')
                tokens1 = tokens1[:, low_bound:up_bound]
                tokens2 = tokens2[:, max(low_bound - 4, 0): max(up_bound - 4, 0)]
                tokens1 = torch.from_numpy(tokens1)
                tokens2 = torch.from_numpy(tokens2)

            if self.split != 'test':
                tokens = tokens1
            else:
                tokens = (tokens1, tokens2)

        else:
            tokens = tokens[0::4]  # non overlapping segments，注意每个window 覆盖64帧，step为16。这里就是取不重叠的windows
            tokens = einops.rearrange(tokens, 'S C T H W -> C (S T) H W')
            tokens = tokens[:, low_bound:up_bound]  # 最后一个window可能包含较多的padding，丢弃 up_bound之后的 padding
            tokens = torch.from_numpy(tokens)

        return tokens

    def __getitem__(self, index):
        video_name = self.df.iloc[index]['name'].replace('.mp4', '_rgb.npz')
        pose_name = self.df.iloc[index]['name'].replace('.mp4', '_pose.npz')

        action_type = self.df.iloc[index]['type']

        row = self.df.iloc[index]
        # 这个视频总的帧数记录在row['num_frames']，动作计数记录在 row['count']
        # duration = row['num_frames']
        count = row['count']

        # 取每个动作开始和结束的标注（帧id）。
        cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])]  ### repetition start-end timestamps

        segment_start = row['segment_start']  # 0
        segment_end = row['segment_end']  # 总帧数
        num_frames = row['num_frames']  # 总帧数

        ### --- Creating density maps ---
        # 论文4.1节，M = 1568*R/64 = (8*14*14)*R/64，相当于 1/8 下采样。R为原始总帧数，每个窗口64帧，每个窗口内的帧采样后编码为1568个tokens
        # 下面的代码按照1/8下采样处理。对预处理的tokens来说，由于window覆盖64帧，最后一个window包含了较多的padding，1/8下采样时去掉多余的 padding
        frame_ids = np.arange(num_frames)  # 帧序列编号
        low = (segment_start // 8) * 8  # 0，标注中的segment_start为 0
        up = math.ceil(segment_end / 8) * 8  # 最后一个段虽然不足 8帧，但还是要采样，对齐

        # 建索引 [0, 8, 16...]
        select_frame_ids = frame_ids[low:up][0::8]  # 0, 8, 16, 24...
        density_map_alt = np.zeros(len(select_frame_ids))

        actual_counts = 0
        for i in range(0, len(cycle), 2):
            if cycle[i] == cycle[i + 1]:  # 如果一个动作周期的开始和结束帧号相同
                continue

            actual_counts += 1
            st = (cycle[i] // 8) * 8  # 将每个动作周期的开始帧号和结束帧号对齐到 8的倍数
            end = min(np.ceil(cycle[i + 1] / 8) * 8, select_frame_ids[-1])

            # 上一步已经对齐，因此这个条件一般是满足的
            if st in select_frame_ids and end in select_frame_ids:
                start_id = np.where(select_frame_ids == st)[0][0]  # 这是下采样后的索引，不是原始帧号（索引*8 对应帧号）
                end_id = np.where(select_frame_ids == end)[0][0]
                mid = (start_id + end_id) // 2  ### get the middle of the repetitions
                density_map_alt[mid] = 1  ### assign 1 to the middle of repetitions

        gt_density = ndimage.gaussian_filter1d(density_map_alt, sigma=self.density_peak_width, order=0)  ### gaussian smoothing
        count_gt = gt_density.sum()  # 看看和标注的差别（很小）

        # 标注的帧号，每个周期的开始和结束，计算每个动作的持续帧数
        starts = np.array(cycle[0::2])
        ends = np.array(cycle[1::2])
        durations = ends - starts
        durations = durations.astype(np.float32)
        durations[durations == 0] = np.inf

        ### Load video tokens
        video_path = f"{self.video_tokens_dir}/{video_name}"
        vid_tokens = self.load_tokens(video_path, (segment_start, segment_end), get_overlapping_segments=self.get_overlapping_segments)

        ### Load pose tokens
        pose_path = f"{self.pose_tokens_dir}/{pose_name}"
        pose_tokens = self.load_tokens(pose_path, (segment_start, segment_end), get_overlapping_segments=self.get_overlapping_segments)

        # 默认走这个流程
        if not self.select_rand_segment:
            vid_tokens = vid_tokens
            gt_density = torch.from_numpy(gt_density).half()
            return vid_tokens, pose_tokens, gt_density, gt_density.sum(), self.df.iloc[index]['name'][:-4], list(vid_tokens[0].shape[-3:]), num_frames, count

        # 默认不会走下面的流程
        T = row['num_frames']  ### number of frames in the video
        if T <= self.num_frames:
            start, end = 0, T
        else:
            start = random.choice(np.arange(0, T - self.num_frames, 64))
            end = start + self.num_frames  ## for taking 8 segments

        sampled_segments = vid_tokens[(start // 64): (end // 64)]
        thw = sampled_segments.shape()[-3:]
        sampled_segments = einops.rearrange(sampled_segments, 'C t h w -> (t h w) C')
        gt = gt_density[(start // 4): (end // 4)]

        return sampled_segments, pose_tokens, gt, gt.sum(), self.df.iloc[index]['name'][:-4], thw, num_frames, count

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        from torch.nn.utils.rnn import pad_sequence

        # [1 x T1 x .... ], [1 x T2 x ....] => [2 x T2 x ....] (T2 > T1) 
        if len(batch[0][0]) == 2:
            vids = pad_sequence([einops.rearrange(x[0][0], 'C T H W -> T C H W') for x in batch])
            vids1 = pad_sequence([einops.rearrange(x[0][1], 'C T H W -> T C H W') for x in batch])
            if self.compact:
                vids = einops.rearrange(vids, 'T B C H W -> B (T H W) C')
                vids1 = einops.rearrange(vids1, 'T B C H W -> B (T H W) C')
            else:
                vids = einops.rearrange(vids, 'T B C H W -> B C T H W')
                vids1 = einops.rearrange(vids1, 'T B C H W -> B C T H W')
            vids = (vids, vids1)
        else:
            vids = pad_sequence([einops.rearrange(x[0], 'C T H W -> T C H W') for x in batch])
            if self.compact:
                vids = einops.rearrange(vids, 'T B C H W -> B (T H W) C')
            else:
                vids = einops.rearrange(vids, 'T B C H W -> B C T H W')

        poses = torch.stack([x[1] for x in batch]).squeeze(1)
        if self.compact:
            poses = einops.rearrange(poses, 'B C T H W -> B (T H W) C')
        gt_density = einops.rearrange(pad_sequence([x[2] for x in batch]), 'S B -> B S')
        gt_density_sum = torch.tensor([x[3] for x in batch], dtype=torch.float)
        names = [x[4] for x in batch]
        thw = [x[5] for x in batch]
        num_frames = [x[6] for x in batch]
        count = [x[7] for x in batch]

        return vids, poses, gt_density, gt_density_sum, names, thw, num_frames, count


## testing
if __name__ == '__main__':
    from tqdm import tqdm

    dat = Rep_count(select_rand_segment=False, compact=False, pool_tokens_factor=0.5, get_overlapping_segments=False)
    print('--- dataset created ---')
    device = torch.device("cpu")
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,
                                             batch_size=1,
                                             num_workers=1,
                                             shuffle=False,
                                             pin_memory=False,
                                             drop_last=True,
                                             collate_fn=dat.collate_fn)

    sum_clip_dur = []
    sum_tot_dur = []
    sum_clip_counts = []
    sum_tot_counts = []

    density_maps_sum = {}
    counts = {}
    density_map_sum = []

    fps = []

    for i, item in enumerate(tqdm(dataloader)):
        print(f"It. {i} \n vid tokens: {item[0][0].shape} \n exem tokens: {item[1].shape} \n density map: {item[2].shape}:{item[3]} \n \n")
        density_map_sum.append(item[3][0].item())
