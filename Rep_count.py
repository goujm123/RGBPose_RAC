from random import randint
import torch.utils.data
import os
import numpy as np
import pandas as pd

from decord import VideoReader, cpu, gpu
from pytorchvideo.transforms import create_video_transform
from itertools import cycle, islice
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def read_video_timestamps(video_filename, timestamps, duration=0):
    """ 
    summary

    Args:
        video_filename (string): full filepath of the video
        timestamps (list): list of ints for the temporal points to load from the file

    Returns:
        frames: tensor of shape C x T x H x W
        totfps: float for the video segment length (in secs) 实际上返回的是一个最后面的帧号
    """
    try:
        assert os.path.isfile(video_filename), f"VideoLoader: {video_filename} does not exist"
    except:
        print(f"{video_filename} does not exist")

    # 用 decord 实现, cpu结果是准确的，但 gpu 还有问题
    vr = VideoReader(video_filename, ctx=cpu(0))
    frames2 = vr.get_batch(timestamps)
    frames2 = torch.from_numpy(frames2.asnumpy())
    video_frames = frames2.permute((3, 0, 1, 2)).to(torch.float32)

    return video_frames, timestamps[-1]


class Rep_count(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 cfg=None,
                 jittering=False,
                 add_noise=False,
                 sampling='uniform',
                 encode_only=False,
                 sampling_interval=4,
                 data_dir="data/RepCount/"):

        self.sampling = sampling
        self.sampling_interval = sampling_interval
        self.encode_only = encode_only
        self.data_dir = data_dir
        self.split = split  # set the split to load
        self.jittering = jittering  # temporal jittering (augmentation)
        self.add_noise = add_noise  # add noise to frames (augmentation)

        # 根据标注文件读取视频
        csv_path = f"datasets/repcount/{self.split}_with_fps.csv"
        self.df = pd.read_csv(csv_path)

        # 过滤掉不符合要求的文件
        self.df = self.df[self.df['count'].notna()]
        self.df = self.df[self.df['num_frames'] > 64]
        self.df = self.df[self.df['count'] > 0]  # remove no reps

        # 过滤标注错误（最常见的是count 或者 cycle标注错误）
        # train集
        self.df = self.df.drop(self.df.loc[self.df['name'] == 'stu1_10.mp4'].index)
        self.df = self.df.drop(self.df.loc[self.df['name'] == 'stu4_3.mp4'].index)
        self.df = self.df.drop(self.df.loc[self.df['name'] == 'test118.mp4'].index)

        # test集（标注错误，但为了和别的论文对比，先保留着）
        # self.df = self.df.drop(self.df.loc[self.df['name'] == 'stu4_5.mp4'].index)

        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- ")

        if cfg is not None:
            self.sample_frames = cfg.DATA.NUM_FRAMES
        else:
            self.sample_frames = 16

    def __getitem__(self, index):
        # 返回视频的元信息，而不是返回具体视频帧的内容数据。视频帧数据在 train 或者 test 流程中进行读取，规避 OOM
        video_name = f"{self.data_dir}/{self.split}/{self.df.iloc[index]['name']}"
        row = self.df.iloc[index]

        # 每个视频总的帧数记录在row['num_frames']，重复动作次数记录在 row['count']
        count = row['count']
        num_frames = row['num_frames']

        # 取每个动作开始和结束的标注（帧id）。
        cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])]
        starts = cycle[0::2]
        ends = cycle[1::2]

        try:
            assert os.path.isfile(video_name), f"VideoLoader: {video_name} does not exist"
        except:
            print(f"{video_name} does not exist")
            return (0, 0, 0, 0), None, None, video_name

        try:
            vr = VideoReader(video_name, ctx=cpu(0))
            total_frames = len(vr)
            if count != len(starts):
                print(f"{self.split}/{video_name} error:  action count={count}, cycles=", len(starts))
            if num_frames != total_frames:
                print(f"{self.split}/{video_name} error:  num_frames={num_frames}, total_frames={total_frames}")

            frame = vr.get_batch([0])  # 取第一帧的信息即可
            del vr
        except:
            print("open file failed:", video_name)
            return (0, 0, 0, 0), None, None, video_name

        _, H, W, c = frame.asnumpy().shape
        shape = (c, total_frames, H, W)

        return shape, starts, ends, video_name

    def __len__(self):
        return len(self.df)


## testing
if __name__ == '__main__':
    from tqdm import tqdm

    dat = Rep_count(data_dir="D:/datasets/RepCount/video")
    print('dataset created')

    dataloader = torch.utils.data.DataLoader(dat, batch_size=1, num_workers=1, shuffle=False, pin_memory=False, drop_last=True)

    for i, item in enumerate(tqdm(dataloader)):
        pass
