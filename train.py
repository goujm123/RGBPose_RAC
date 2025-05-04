import torch
import torch.nn as nn
import numpy as np
import os, sys
from train_data_loader import Rep_count
from tqdm import tqdm
from video_mae_cross_full_attention import SupervisedMAE
from util.config import load_config
import timm.optim.optim_factory as optim_factory
import argparse
import wandb
import torch.optim as optim
import random
import time

torch.manual_seed(0)
torch.cuda.empty_cache()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    # worker_seed = 0
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--config', default='configs/pretrain_config.yaml', type=str)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--encodings', default='mae', type=str, help="['swin', 'mae']")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--only_test', default=False, help='Only testing')
    parser.add_argument('--trained_model', default='./saved_models_repcount/best_1.pyth', type=str, help='path to a trained model')
    parser.add_argument('--scale_counts', default=100, type=int, help='scaling the counts')

    parser.add_argument('--dataset', default='RepCount', type=str, help='Repcount, Countix, UCFRep')

    parser.add_argument('--get_overlapping_segments', default=False, help='whether to get overlapping segments')

    parser.add_argument('--peak_at_random_locations', default=False, type=bool, help='whether to have density peaks at random locations')

    parser.add_argument('--multishot', default=True, action='store_true')

    parser.add_argument('--iterative_shots', default=True, action='store_true', help='will show the examples one by one')

    parser.add_argument('--density_peak_width', default=0.5, type=float, help='sigma for the peak of density maps, lesser sigma gives sharp peaks')

    # Model parameters
    parser.add_argument('--save_path', default='./saved_models_repcountfull', type=str, help="Path to save the model")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate (peaklr)')
    parser.add_argument('--eval_freq', default=2, type=int)

    # Dataset parameters
    parser.add_argument('--precomputed', default=True, type=lambda x: (str(x).lower() == 'true'), help='flag to specify if precomputed tokens will be loaded')
    parser.add_argument('--data_path', default='', type=str, help='dataset path')
    parser.add_argument('--slurm_job_id', default=None, type=str, help='job id')

    parser.add_argument('--video_tokens_dir', default='D:/datasets/RepCount/tokens', type=str, help='ground truth density map directory')
    parser.add_argument('--pose_tokens_dir', default='D:/datasets/RepCount/tokens', type=str, help='tokens of poses')

    # 选取因子p，用于设置是否在同类动作的不同视频中取数据
    parser.add_argument('--threshold', default=0, type=float, help='p, cut off to decide if select exemplar from different video')

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Training parameters
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--pretrained_encoder', default='pretrained_models/VIT_B_16x4_MAE_PT.pth', type=str)

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus')

    # Logging parameters
    parser.add_argument('--log_dir', default='./logs/fim6_dir', help='path where to tensorboard log')
    parser.add_argument("--title", default="", type=str)
    parser.add_argument("--use_wandb", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--wandb", default="", type=str)
    parser.add_argument("--team", default="", type=str)
    parser.add_argument("--wandb_id", default='', type=str)

    parser.add_argument("--token_pool_ratio", default=1.0, type=float)
    parser.add_argument("--rho", default=0.7, type=float)
    parser.add_argument("--window_size", default=(4, 7, 7), type=int, nargs='+', help='window size for windowed self attention')

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    args.window_size = tuple(args.window_size)
    args.opts = None
    g = torch.Generator()
    g.manual_seed(args.seed)

    cfg = load_config(args)

    # create dataloaders
    dataset_train = Rep_count(split="train",
                              video_tokens_dir=args.video_tokens_dir,
                              pose_tokens_dir=args.pose_tokens_dir,
                              select_rand_segment=False,
                              compact=True,
                              pool_tokens_factor=args.token_pool_ratio,
                              peak_at_random_location=args.peak_at_random_locations,
                              get_overlapping_segments=args.get_overlapping_segments,
                              multishot=args.multishot,
                              threshold=args.threshold)

    dataset_valid = Rep_count(split="valid",
                              video_tokens_dir=args.video_tokens_dir,
                              pose_tokens_dir=args.pose_tokens_dir,
                              select_rand_segment=False,
                              compact=True,
                              pool_tokens_factor=args.token_pool_ratio,
                              peak_at_random_location=args.peak_at_random_locations,
                              get_overlapping_segments=args.get_overlapping_segments,
                              multishot=args.multishot,
                              density_peak_width=args.density_peak_width)

    dataset_test = Rep_count(split="test",
                             video_tokens_dir=args.video_tokens_dir,
                             pose_tokens_dir=args.pose_tokens_dir,
                             select_rand_segment=False,
                             compact=True,
                             pool_tokens_factor=args.token_pool_ratio,
                             peak_at_random_location=args.peak_at_random_locations,
                             get_overlapping_segments=args.get_overlapping_segments,
                             multishot=args.multishot,
                             density_peak_width=args.density_peak_width)

    # Create dict of dataloaders for train and val
    dataloaders = {'train': torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        shuffle=True,
                                                        pin_memory=False,
                                                        drop_last=False,
                                                        collate_fn=dataset_train.collate_fn,
                                                        worker_init_fn=seed_worker,
                                                        persistent_workers=True,
                                                        generator=g),
                   'val': torch.utils.data.DataLoader(dataset_valid,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=False,
                                                      drop_last=False,
                                                      collate_fn=dataset_valid.collate_fn,
                                                      worker_init_fn=seed_worker,
                                                      generator=g),
                   'test': torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=1,
                                                       num_workers=args.num_workers,
                                                       shuffle=False,
                                                       pin_memory=False,
                                                       drop_last=False,
                                                       collate_fn=dataset_valid.collate_fn,
                                                       worker_init_fn=seed_worker,
                                                       generator=g)}

    # scaler = torch.cuda.amp.GradScaler() # use mixed percision for efficiency
    # scaler = NativeScaler()
    model = SupervisedMAE(cfg=cfg, use_precomputed=args.precomputed, token_pool_ratio=args.token_pool_ratio, iterative_shots=args.iterative_shots,
                          encodings="mae", window_size=args.window_size).cuda()

    train_step = 0
    val_step = 0

    # only for testing
    if args.only_test:
        model.load_state_dict(torch.load(args.trained_model)['model_state_dict'])  ### load trained model
        videos = []
        loss = []
        model.eval()
        print(f"Testing")
        dataloader = dataloaders['test']
        gt_counts = list()
        predictions = list()
        predict_mae = list()
        predict_mse = list()
        clips = list()

        bformat = '{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
        with tqdm(total=len(dataloader), bar_format=bformat, ascii='░▒█') as pbar:
            for i, item in enumerate(dataloader):
                if args.get_overlapping_segments:
                    rgb, data2 = item[0][0], item[0][1]
                else:
                    rgb = item[0].cuda().type(torch.cuda.FloatTensor)  # B x (THW) x C
                example = item[1].cuda().type(torch.cuda.FloatTensor)  # B x (THW) x C
                density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts
                actual_counts = item[3].cuda()  # B x 1
                video_name = item[4]

                videos.append(video_name[0])

                shot_num = item[6][0]
                b, n, c = rgb.shape

                thw = item[5]
                with torch.no_grad():
                    if args.get_overlapping_segments:
                        rgb = rgb.cuda().type(torch.cuda.FloatTensor)
                        data2 = data2.cuda().type(torch.cuda.FloatTensor)
                        pred1 = model(rgb, example, thw, shot_num=shot_num)
                        pred2 = model(data2, example, thw, shot_num=shot_num)
                        if pred1.shape != pred2.shape:
                            pred2 = torch.cat([torch.zeros(1, 4).cuda(), pred2], 1)
                        else:
                            print('equal')
                        pred = (pred1 + pred2) / 2
                    else:
                        pred = model(rgb, example, thw, shot_num=shot_num)  ### predict the density maps

                mse = ((pred - density_map) ** 2).mean(-1)
                predict_counts = torch.sum(pred, dim=1).type(torch.FloatTensor).cuda() / args.scale_counts  #### scaling down by args.scale_counts
                predict_counts = predict_counts.round()
                predictions.extend(predict_counts.detach().cpu().numpy())
                gt_counts.extend(actual_counts.detach().cpu().numpy())
                mae = torch.div(torch.abs(predict_counts - actual_counts), actual_counts + 1e-1)
                predict_mae.extend(mae.cpu().numpy())
                predict_mse.extend(np.sqrt(mse.cpu().numpy()))
                loss.append(mse.cpu().numpy())
                pbar.update()

        predict_mae = np.array(predict_mae)
        predictions = np.array(predictions).round()
        gt_counts = np.array(gt_counts)
        predict_mse = np.array(predict_mse)
        diff = np.abs(predictions.round() - gt_counts)
        diff_z = np.abs(predictions.round() - gt_counts.round())
        print(f'Overall MAE: {predict_mae.mean()}')  ### calculating mae
        print(f'OBO: {(diff <= 1).sum() / len(diff)}')  ### calculating obo
        print(f'OBZ: {(diff_z == 0).sum() / len(diff)}')  ### calculating obz
        print(f'RMSE: {np.sqrt((diff ** 2).mean())}')  ### calculating rmse
        return

    # train
    if args.use_wandb:
        wandb_run = wandb.init(
            config=args,
            resume="allow",
            project=args.wandb,
            anonymous="allow",
            mode="offline",
            entity=args.team,
            id=f"{args.wandb_id}_{args.dataset}_{args.encodings}_{args.lr}_{args.threshold}",
        )

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    milestones = [i for i in range(0, args.epochs, 60)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.8)  ### reduce learning rate by 0.8 every 60 epochs
    lossMSE = nn.MSELoss().cuda()
    lossSL1 = nn.SmoothL1Loss().cuda()
    best_loss = np.inf

    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        # scheduler.step()   # 新版本的torch中， scheduler.step() 要放到 optimizer.step() 后
        start_time = time.time()

        print(f"Epoch: {epoch:02d}")
        for phase in ['train']:
            if phase == 'val':
                if epoch % args.eval_freq != 0:
                    continue
                model.eval()
                ground_truth = list()
                predictions = list()
            else:
                model.train()

            with torch.set_grad_enabled(phase == 'train'):
                total_loss_all = 0
                total_loss1 = 0
                total_loss2 = 0
                total_loss3 = 0
                off_by_zero = 0
                off_by_one = 0
                mse = 0
                count = 0
                mae = 0

                bformat = '{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
                dataloader = dataloaders[phase]

                with tqdm(total=len(dataloader), bar_format=bformat, ascii='░▒█') as pbar:
                    for i, item in enumerate(dataloader):
                        if phase == 'train':
                            train_step += 1
                        elif phase == 'val':
                            val_step += 1

                        with torch.cuda.amp.autocast(enabled=True):
                            rgb = item[0].cuda().type(torch.cuda.FloatTensor)  # B x (THW) x C
                            pose = item[1].cuda().type(torch.cuda.FloatTensor)  # B x (THW) x C
                            density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts  ###scaling up args.scale_counts.This helps in magnifying the loss
                            actual_counts = item[3].cuda()  # B x 1
                            thw = item[5]
                            shot_num = item[6][0]  ## number of shots
                            b, n, c = rgb.shape
                            y = model(rgb, pose, thw, shot_num=shot_num)

                            if phase == 'train':
                                mask = np.random.binomial(n=1, p=0.8, size=[1, density_map.shape[1]])  ### random masking of 20% density map
                            else:
                                mask = np.ones([1, density_map.shape[1]])

                            masks = np.tile(mask, (density_map.shape[0], 1))

                            masks = torch.from_numpy(masks).cuda()
                            loss = ((y - density_map) ** 2)
                            loss = ((loss * masks) / density_map.shape[1]).sum() / density_map.shape[0]  ### mse

                            predict_count = torch.sum(y, dim=1).type(torch.cuda.FloatTensor) / args.scale_counts  # sum density map
                            # loss_mse = torch.mean((predict_count - actual_counts)**2)

                            if phase == 'val':
                                ground_truth.append(actual_counts.detach().cpu().numpy())
                                predictions.append(predict_count.detach().cpu().numpy())

                            loss2 = lossSL1(predict_count, actual_counts)  ###L1 loss between count and predicted count
                            loss3 = torch.sum(torch.div(torch.abs(predict_count - actual_counts), actual_counts + 1e-1)) / \
                                    predict_count.flatten().shape[0]  #### reduce the mean absolute error (mae loss)

                            if phase == 'train':
                                loss1 = (loss + 1.0 * loss3) / args.accum_iter  ### mse between density maps + mae loss (loss3)
                                loss1.backward()  ### call backward
                                if (i + 1) % args.accum_iter == 0:  ### accumulate gradient
                                    optimizer.step()  ##update parameters
                                    optimizer.zero_grad()
                                    torch.cuda.empty_cache()

                            epoch_loss = loss.item()
                            count += b
                            total_loss_all += loss.item() * b
                            total_loss1 += loss.item() * b
                            total_loss2 += loss2.item() * b
                            total_loss3 += loss3.item() * b
                            off_by_zero += (torch.abs(actual_counts.round() - predict_count.round()) == 0).sum().item()  ## off by zero
                            off_by_one += (torch.abs(actual_counts.round() - predict_count.round()) <= 1).sum().item()  ## off by one
                            mse += ((actual_counts - predict_count.round()) ** 2).sum().item()
                            mae += torch.sum(torch.div(torch.abs(predict_count.round() - actual_counts), (actual_counts) + 1e-1)).item()  ##mean absolute error

                            pbar.set_description(f"EPOCH: {epoch:02d} | PHASE: {phase} ")
                            pbar.set_postfix_str(
                                f" LOSS: {total_loss_all / count:.2f} | MAE:{mae / count:.2f} | LOSS ITER: {loss.item():.2f} | OBZ: {off_by_zero / count:.2f} | OBO: {off_by_one / count:.2f} | RMSE: {np.sqrt(mse / count):.3f}")
                            pbar.update()

                if args.use_wandb:
                    if phase == 'train':
                        wandb.log({"epoch": epoch,
                                   # "lr": lr,
                                   "train_loss": total_loss_all / float(count),
                                   "train_loss1": total_loss1 / float(count),
                                   "train_loss2": total_loss2 / float(count),
                                   "train_loss3": total_loss3 / float(count),
                                   "train_obz": off_by_zero / count,
                                   "train_obo": off_by_one / count,
                                   "train_rmse": np.sqrt(mse / count),
                                   "train_mae": mae / count
                                   })

                    if phase == 'val':
                        if not os.path.isdir(args.save_path):
                            os.makedirs(args.save_path)
                        wandb.log({"epoch": epoch,
                                   "val_loss": total_loss_all / float(count),
                                   "val_loss1": total_loss1 / float(count),
                                   "val_loss2": total_loss2 / float(count),
                                   "val_loss3": total_loss3 / float(count),
                                   "val_obz": off_by_zero / count,
                                   "val_obo": off_by_one / count,
                                   "val_mae": mae / count,
                                   "val_rmse": np.sqrt(mse / count)
                                   })

                        ### Savind checkpoints
                        if total_loss_all / float(count) < best_loss:
                            best_loss = total_loss_all / float(count)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }, os.path.join(args.save_path, 'best_1.pyth'))
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(args.save_path, 'epoch_{}.pyth'.format(str(epoch).zfill(3))))

        scheduler.step()
        used_time = time.time() - start_time

        print(f"Time elapsed: {int(used_time)} sec")

    if args.use_wandb:
        wandb_run.finish()


if __name__ == '__main__':
    main()
