import os
import time
import yaml
import random
import logging
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

import torchvision
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
#from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.distributed

import argparse
from thop import profile

from models.submodules.layers import Conv1x1, Conv3x3, Linear, SpikingMatmul
import models.spikingresformer
from utils.augment import DVSAugment
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter
from utils.utils import RecordDict, GlobalTimer, Timer
from utils.utils import count_convNd, count_linear, count_matmul
from utils.utils import DatasetSplitter, DatasetWarpper, CriterionWarpper, DVStransform, SOPMonitor
from utils.utils import is_main_process, save_on_master, tb_record, accuracy, safe_makedirs
from spikingjelly.activation_based import functional, layer, base
from timm.data import FastCollateMixup, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model
from models.spikingresformer import DSSAWithSSDP


def parse_args():
    config_parser = argparse.ArgumentParser(description="Training Config", add_help=False)

    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description='Training')

    # training options
    parser.add_argument('--seed', default=12450, type=int)
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--model', default='spikingresformer_ti', help='model type')
    parser.add_argument('--dataset', default='ImageNet', help='dataset type')
    parser.add_argument('--augment', type=str, help='data augmentation')
    parser.add_argument('--mixup', type=bool, default=False, help='Mixup')
    parser.add_argument('--cutout', type=bool, default=False, help='Cutout')
    parser.add_argument('--label-smoothing', type=float, default=0, help='Label smoothing')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')

    parser.add_argument('--print-freq', default=5, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--data-path', default='./datasets')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--transfer', type=str, help='transfer from pretrained checkpoint')
    parser.add_argument('--input-size', type=int, nargs='+', default=[])
    parser.add_argument('--distributed-init-mode', type=str, default='env://')

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    parser.add_argument('--save-latest', action='store_true')
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', type=bool, default=True, help='Use AMP training')
    parser.add_argument('--sync-bn', action='store_true', help='Use SyncBN training')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args


def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(
    dataset_dir: str,
    batch_size: int,
    workers: int,
    num_classes: int,
    dataset_type: str,
    input_size: Tuple[int],
    distributed: bool,
    augment: str,
    mixup: bool,
    cutout: bool,
    label_smoothing: float,
    T: int,
):

    if dataset_type == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=True,
                                                     download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.0,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.5, cutmix_alpha=0.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )
    elif dataset_type == 'CIFAR100':
        dataset_train = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True,
                                                      download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.0,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.5, cutmix_alpha=0.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[n / 255. for n in [129.3, 124.1, 112.4]],
            std=[n / 255. for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[n / 255. for n in [129.3, 124.1, 112.4]],
            std=[n / 255. for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        if augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True),
                DVSAugment()]))
        else:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True)]))
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))

        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        dataset_train, dataset_test = DatasetSplitter(dataset, 0.9,
                                                      True), DatasetSplitter(dataset, 0.1, False)
        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
                dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)  # type:ignore
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                        sampler=train_sampler, num_workers=workers,
                                                        pin_memory=True, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       sampler=test_sampler, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'DVS128Gesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        if augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True),
                DVSAugment()]))
        else:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True)]))
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))

        dataset_train = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T,
                                      split_by='number')
        dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T,
                                     split_by='number')
        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
                dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)  # type:ignore
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                        sampler=train_sampler, num_workers=workers,
                                                        pin_memory=True, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       sampler=test_sampler, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'ImageNet' or dataset_type == 'ImageNet100':
        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')
        dataset_train = torchvision.datasets.ImageFolder(traindir)
        dataset_test = torchvision.datasets.ImageFolder(valdir)
        augment_args = dict(
            scale=[0.08, 1.0],
            ratio=[3. / 4., 4. / 3.],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.4,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.2, cutmix_alpha=1.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            distributed=distributed,
            crop_pct=0.95,
            pin_memory=True,
        )
    else:
        raise ValueError(dataset_type)

    return dataset_train, dataset_test, data_loader_train, data_loader_test


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader_train: torch.utils.data.DataLoader,
    logger: logging.Logger,
    print_freq: int,
    factor: int,
    scheduler_per_iter=None,
    scaler: Optional[GradScaler] = None,
    one_hot: Optional[int] = None,
):
    model.train()
    from collections import defaultdict
    import time

    class RecordDict:
        def __init__(self, init_dict):
            self.sum_dict = defaultdict(float)
            self.cnt_dict = defaultdict(int)
            for k in init_dict.keys():
                self.sum_dict[k] = 0.0
                self.cnt_dict[k] = 0
        def update(self, key, val, n=1):
            self.sum_dict[key] += val * n
            self.cnt_dict[key] += n
        def __getitem__(self, key):
            if self.cnt_dict[key] == 0:
                return 0.0
            return self.sum_dict[key] / self.cnt_dict[key]
        def sync(self):
            pass

    def accuracy(output, target, topk=(1, 5)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    metric_dict = RecordDict({'loss': 0.0, 'acc@1': 0.0, 'acc@5': 0.0})
    t_start = time.time()

    model.zero_grad()
    for idx, (image, target) in enumerate(data_loader_train):
        image, target = image.cuda(), target.cuda()
        if one_hot:
            target = F.one_hot(target, one_hot).float()

        # ---------- forward + loss ----------
        if scaler is not None:
            with autocast():
                output = model(image)          # logits [B, C]
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)
        metric_dict.update('loss', loss.item(), image.size(0))

        # ---------- standard back‑prop update ----------
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            model.zero_grad()

        # optional LR‑scheduler (per‑iteration)
        if scheduler_per_iter is not None:
            scheduler_per_iter.step()

        # -------------------------------------------------------
        # (A) SSDP update on the *classifier* weight matrix
        # -------------------------------------------------------
        model_without_ddp = model.module if hasattr(model, 'module') else model
        # Only start applying SSDP after the user‑defined warm‑up epochs
        if hasattr(model_without_ddp, 'current_epoch') and \
           model_without_ddp.current_epoch >= model_without_ddp.start_epoch:

            # Binary spike indicators for current mini‑batch
            pre_spike  = (model_without_ddp.out > 0).float().detach()   # presynaptic spikes  [B, C_in]
            post_spike = (output > 0).float().detach()                  # postsynaptic spikes [B, C_out]

            # Spike times recorded during forward pass
            t_pre  = model_without_ddp.t_pre    # [B, C_in]
            t_post = model_without_ddp.t_post   # [B, C_out]

            # Pairwise absolute timing differences Δt_ij = |t_post_j − t_pre_i|
            t_pre_exp  = t_pre.unsqueeze(1)     # [B, 1, C_in]
            t_post_exp = t_post.unsqueeze(2)    # [B, C_out, 1]
            delta_t    = (t_post_exp - t_pre_exp).abs()  # [B, C_out, C_in]

            # Weight increment Δw_ij from SSDP kernel (no grad, online Hebbian‑style)
            with torch.no_grad():
                delta_w = model_without_ddp.ssdp(pre_spike, post_spike, delta_t)

                # Safety check: make sure shapes match classifier weight
                if delta_w.shape != model_without_ddp.classifier.weight.shape:
                    print("[SSDP Debug] shape mismatch:",
                          delta_w.shape, model_without_ddp.classifier.weight.shape)

                # In‑place weight update
                model_without_ddp.classifier.weight.add_(delta_w)

        # -----------------------------------------------------------------
        # (B) SSDP update on Wproj of the *last* DSSA attention block
        # -----------------------------------------------------------------
        if hasattr(model_without_ddp, 'current_epoch') and \
           model_without_ddp.current_epoch >= model_without_ddp.start_epoch:

            last_stage = model_without_ddp.layers[-1]  # nn.Sequential (last stage)
            dssa_layer = None
            for m in last_stage:                       # find first DSSAWithSSDP
                if isinstance(m, DSSAWithSSDP):
                    dssa_layer = m
                    break

            if dssa_layer is not None:
                # x_in / x_out were cached inside DSSAWithSSDP during forward()
                # Shapes: [T, B, C, H, W]
                x_in  = dssa_layer.x_in.detach()
                x_out = dssa_layer.x_out.detach()
                T, B, C, H, W = x_in.shape

                # ---------- collapse spatial dims (H, W) ----------
                # For each channel c and time step t: did *any* neuron in (H,W) spike?
                x_in_channels  = (x_in  > 0).any(dim=4).any(dim=3).float()  # [T, B, C]
                x_out_channels = (x_out > 0).any(dim=4).any(dim=3).float()  # [T, B, C]

                # Convert spike trains to first‑spike times
                pre_exist  = (x_in_channels  > 0)
                post_exist = (x_out_channels > 0)

                # Index of first spike along temporal axis, fallback to T if no spike
                t_pre_  = pre_exist.float().cumsum(dim=0).argmax(dim=0).float()
                t_pre_[~pre_exist.any(dim=0)]  = float(T)   # no presyn spike
                t_post_ = post_exist.float().cumsum(dim=0).argmax(dim=0).float()
                t_post_[~post_exist.any(dim=0)] = float(T)  # no postsyn spike

                # Collapse temporal dimension to single spike indicator per channel
                pre_spike  = (x_in_channels.sum(dim=0)  > 0).float()  # [B, C_in]
                post_spike = (x_out_channels.sum(dim=0) > 0).float()  # [B, C_out]

                # Pairwise Δt between channel‑wise first spikes
                t_pre_exp  = t_pre_.unsqueeze(1)   # [B, 1, C_in]
                t_post_exp = t_post_.unsqueeze(2)  # [B, C_out, 1]
                delta_t    = (t_post_exp - t_pre_exp).abs()  # [B, C_out, C_in]

                # Hebbian update for projection matrix Wproj
                with torch.no_grad():
                    delta_w = model_without_ddp.ssdp_dssa(pre_spike, post_spike, delta_t)

                    wproj = dssa_layer.Wproj.weight           # [C_out, C_in, 1, 1]
                    c_out, c_in, _, _ = wproj.shape
                    if delta_w.shape != (c_out, c_in):
                        print("[SSDP DSSA Debug] shape mismatch:",
                              delta_w.shape, wproj.shape)

                    # In‑place add, reshaped to 4‑D conv kernel
                    wproj.data.add_(delta_w.view(c_out, c_in, 1, 1))

        # ---------- reset membrane potentials etc. ----------
        functional.reset_net(model)

        # ---------- bookkeeping ----------
        if target.dim() > 1:
            target = target.argmax(-1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_dict.update('acc@1', acc1.item(), image.size(0))
        metric_dict.update('acc@5', acc5.item(), image.size(0))

        if print_freq != 0 and ((idx + 1) % int(len(data_loader_train) / (print_freq))) == 0:
            metric_dict.sync()
            speed = (idx + 1) * image.size(0) * factor / (time.time() - t_start)
            logger.debug(
                ' [{}/{}] it/s: {:.5f}, loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader_train),
                    speed, metric_dict['loss'], metric_dict['acc@1'], metric_dict['acc@5']
                )
            )

    metric_dict.sync()
    return metric_dict['loss'], metric_dict['acc@1'], metric_dict['acc@5']



def evaluate(model, criterion, data_loader, print_freq, logger, one_hot=None):
    model.eval()
    from utils.utils import RecordDict, accuracy
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image, target = image.cuda(), target.cuda()
            if one_hot:
                target = F.one_hot(target, one_hot).float()
            output = model(image) # [B,C]
            loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            if target.dim() > 1:
                target = target.argmax(-1)
            acc1, acc5 = accuracy(output, target, topk=(1,5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % int(len(data_loader) / print_freq)) == 0:
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave

def test(
    model: nn.Module,
    data_loader_test: torch.utils.data.DataLoader,
    input_size: Tuple[int],
    args: argparse.Namespace,
    logger: logging.Logger,
):
    logger.info('[Test]')
    from utils.utils import RecordDict, accuracy, SOPMonitor
    mon = SOPMonitor(model)
    model.eval()
    mon.enable()
    logger.debug('Test start')
    metric_dict = RecordDict({'acc@1': None, 'acc@5': None}, test=True)
    with torch.no_grad():
        t = time.time()
        for idx, (image, target) in enumerate(data_loader_test):
            image, target = image.cuda(), target.cuda()
            output = model(image) # [B,C]
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1,5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if args.print_freq != 0 and ((idx + 1) % int(len(data_loader_test) / args.print_freq)) == 0:
                logger.debug('Test: [{}/{}]'.format(idx + 1, len(data_loader_test)))
        logger.info('Throughput: {:.5f} it/s'.format(
            len(data_loader_test) * args.batch_size / (time.time() - t)))
    metric_dict.sync()
    logger.info('Acc@1: {:.5f}, Acc@5: {:.5f}'.format(metric_dict['acc@1'].ave,
                                                      metric_dict['acc@5'].ave))

    step_mode = 's'
    for m in model.modules():
        if isinstance(m, base.StepModule):
            if m.step_mode == 'm':
                step_mode = 'm'
            else:
                step_mode = 's'
            break

    from utils.utils import count_convNd, count_linear, count_matmul
    ops, params = profile(
        model, inputs=(torch.rand(input_size).cuda().unsqueeze(0), ), verbose=False, custom_ops={
            layer.Conv2d: count_convNd,
            Conv3x3: count_convNd,
            Conv1x1: count_convNd,
            Linear: count_linear,
            SpikingMatmul: count_matmul, })[0:2]
    if step_mode == 'm':
        ops, params = (ops / (1000**3)) / args.T, params / (1000**2)
    else:
        ops, params = (ops / (1000**3)), params / (1000**2)
    functional.reset_net(model)
    logger.info('MACs: {:.5f} G, params: {:.2f} M.'.format(ops, params))

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        sop = torch.cat(sublist).mean().item()
        sops = sops + sop
    sops = sops / (1000**3)
    sops = sops / args.batch_size
    if step_mode == 's':
        sops = sops * args.T
    logger.info('Avg SOPs: {:.5f} G, Power: {:.5f} mJ.'.format(sops, 0.9 * sops))
    logger.info('A/S Power Ratio: {:.6f}'.format((4.6 * ops) / (0.9 * sops)))

def cos_annealing_factor(epoch, total_epochs, start_factor, end_factor):
    """
    返回某个[ start_factor -> end_factor ]的余弦退火插值系数
    公式示例: factor = end_factor + 0.5 * (start_factor - end_factor)
                         * (1 + cos(pi * epoch / total_epochs))
    """
    import math
    cos_part = 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    factor = end_factor + (start_factor - end_factor) * cos_part
    return factor


def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))

    # load data

    dataset_type = args.dataset
    one_hot = None
    if dataset_type == 'CIFAR10':
        num_classes = 10
        input_size = (3, 32, 32)
    elif dataset_type == 'CIFAR10DVS':
        one_hot = 10
        num_classes = 10
        input_size = (3, 64, 64)
    elif dataset_type == 'DVS128Gesture':
        one_hot = 11
        num_classes = 11
        input_size = (3, 64, 64)
    elif dataset_type == 'CIFAR100':
        num_classes = 100
        input_size = (3, 32, 32)
    elif dataset_type == 'ImageNet':
        num_classes = 1000
        input_size = (3, 224, 224)
    elif dataset_type == 'ImageNet100':
        num_classes = 100
        input_size = (3, 224, 224)
    else:
        raise ValueError(dataset_type)
    if len(args.input_size) != 0:
        input_size = args.input_size

    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path, args.batch_size, args.workers, num_classes, dataset_type, input_size,
        distributed, args.augment, args.mixup, args.cutout, args.label_smoothing, args.T)
    logger.info('dataset_train: {}, dataset_test: {}'.format(len(dataset_train), len(dataset_test)))

    # model

    model = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size[-1],
    ).cuda()

    # transfer
    if args.transfer:
        checkpoint = torch.load(args.transfer, map_location='cpu')
        model.transfer(checkpoint['model'])

    # optimzer

    optimizer = create_optimizer_v2(
        model,
        opt=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # loss_fn

    if args.mixup:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)
    criterion_eval = nn.CrossEntropyLoss()
    criterion_eval = CriterionWarpper(criterion_eval)

    # amp speed up

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # lr scheduler

    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched='cosine',
        num_epochs=args.epochs,
        cooldown_epochs=10,
        min_lr=1e-5,
        warmup_lr=1e-5,
        warmup_epochs=3,
    )

    # Sync BN
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP

    model_without_ddp = model
    if distributed and not args.test_only:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=False)
        model_without_ddp = model.module

    # custom scheduler

    scheduler_per_iter = None
    scheduler_per_epoch = None

    # resume

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
        # custom scheduler
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   test only
    ##################################################

    if args.test_only:
        if distributed:
            logger.error('Using distribute mode in test, abort')
            return
        test(model_without_ddp, data_loader_test, input_size, args, logger)
        return

    ##################################################
    #                   Train
    ##################################################

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'),
                                  purge_step=start_epoch)

    logger.info("[Train]")
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch, optimizer.param_groups[0]["lr"]))

        # 如果模型内有 self.current_epoch 需更新：
        if hasattr(model_without_ddp, 'current_epoch'):
            model_without_ddp.current_epoch = epoch

        with Timer(' Train', logger):
            train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer,
                                                                 data_loader_train, logger,
                                                                 args.print_freq, world_size,
                                                                 scheduler_per_iter, scaler,
                                                                 one_hot)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()

        with Timer(' Test', logger):
            test_loss, test_acc1, test_acc5 = evaluate(model, criterion_eval, data_loader_test,
                                                       args.print_freq, logger, one_hot)

        if is_main_process() and tb_writer is not None:
            tb_record(tb_writer, train_loss, train_acc1, train_acc5, test_loss, test_acc1,
                      test_acc5, epoch)

        logger.info(' Test loss: {:.5f}, Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
            test_loss, test_acc1, test_acc5))

        # ----------在此处手动更新SSD**P**的lr-----------
        model_without_ddp = model.module if hasattr(model, 'module') else model
        if hasattr(model_without_ddp, 'ssdp'):
            with torch.no_grad():
                # 1) 计算每个参数在epoch时的"目标值"
                #    例如A_plus从 0.0002退火到0.00005
                #    A_minus从 0.00015 -> 0.00003
                #    A_baseline从0.0001 -> 0.0
                A_plus_target = cos_annealing_factor(
                    epoch=epoch,
                    total_epochs=args.epochs,
                    start_factor=0.0001,  # initial
                    end_factor=0.00005  # final
                )
                A_minus_target = cos_annealing_factor(
                    epoch=epoch,
                    total_epochs=args.epochs,
                    start_factor=0.00015,
                    end_factor=0.00003
                )
                A_baseline_target = cos_annealing_factor(
                    epoch=epoch,
                    total_epochs=args.epochs,
                    start_factor=0.0001,
                    end_factor=0.00005
                )

                # 2) 采用“线性插值”方式，逐渐逼近target，而非一次性跳到target
                #    例如  alpha=0.3 代表做 30% 的插值前进；你可根据需求调节
                alpha = 0.3

                current_Ap = model_without_ddp.ssdp.A_plus.item()
                new_Ap = (1 - alpha) * current_Ap + alpha * A_plus_target

                current_Am = model_without_ddp.ssdp.A_minus.item()
                new_Am = (1 - alpha) * current_Am + alpha * A_minus_target

                current_Ab = model_without_ddp.ssdp.A_baseline.item()
                new_Ab = (1 - alpha) * current_Ab + alpha * A_baseline_target

                # 3) clamp 避免极端情况
                new_Ap = max(min(new_Ap, 1e-2), 1e-7)  # demo: clamp in [1e-7, 1e-2]
                new_Am = max(min(new_Am, 1e-2), 1e-7)
                new_Ab = max(min(new_Ab, 1e-2), 0.0)  # baseline不应为负？

                # 4) 写回
                model_without_ddp.ssdp.A_plus.fill_(new_Ap)
                model_without_ddp.ssdp.A_minus.fill_(new_Am)
                model_without_ddp.ssdp.A_baseline.fill_(new_Ab)
        # ========== end SSDP update ===========
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1, }
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        # custom scheduler

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'))

    logger.info('Training completed.')

    ##################################################
    #                   test
    ##################################################

    ##### reset utils #####

    # reset model

    del model, model_without_ddp

    model = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size[-1],
    )

    try:
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    except:
        logger.warning('Cannot load max acc1 model, skip test.')
        logger.warning('Exit.')
        return

    # reload data

    del dataset_train, dataset_test, data_loader_train, data_loader_test
    _, _, _, data_loader_test = load_data(args.data_path, args.batch_size, args.workers,
                                          num_classes, dataset_type, input_size, False,
                                          args.augment, args.mixup, args.cutout,
                                          args.label_smoothing, args.T)

    ##### test #####

    model = model.cuda()

    if is_main_process():
        test(model, data_loader_test, input_size, args, logger)
    logger.info('All Done.')


if __name__ == "__main__":
    main()
