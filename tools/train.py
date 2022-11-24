from __future__ import division
import argparse
import os
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import torch
from mmcv import Config
import torch.nn.functional as F

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

import warnings
from mmdet.datasets import DATASETS, build_dataloader
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def select_training_param(model):

    for v in model.parameters():
        v.requires_grad = False

    model.bbox_head.fc_cls.weight.requires_grad = True
    model.bbox_head.fc_cls.bias.requires_grad = True

    return model


def select_head(model):

    for v in model.parameters():
        v.requires_grad = False

    for v in model.bbox_head.parameters():
        v.requires_grad = True

    return model

def select_cascade_cls_params(model):

    for v in model.parameters():
        v.requires_grad = False

    for child in model.bbox_head.children():
        for v in child.fc_cls.parameters():
            v.requires_grad = True

    return model

def select_mask_params(model):

    for v in model.parameters():
        v.requires_grad = False

    for v in model.bbox_head.parameters():
        v.requires_grad = True
    for v in model.mask_head.parameters():
        v.requires_grad = True

    return model

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    tune_part = cfg.get('selectp', 0)
    if tune_part == 1:
        print('Train fc_cls only.')
        model = select_training_param(model)
    elif tune_part == 2:
        print('Train bbox head only.')
        model = select_head(model)
    elif tune_part == 3:
        print('Train cascade fc_cls only.')
        model = select_cascade_cls_params(model)
    elif tune_part == 4:
        print('Train bbox and mask head.')
        model = select_mask_params(model)
    else:
        print('Train all params.')




    dataset = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    # put model on gpus

    
    logger.info('Loading checkpoint: {} ...'.format(cfg.load_from))
    checkpoint = torch.load(cfg.load_from)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad=False
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    model.eval()
    aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
    aggregation_weight.data.fill_(1/3) 
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    optimizer = torch.optim.SGD([aggregation_weight], lr=0.02, momentum=0.9, weight_decay=0.0001)

    for epoch in range(12):
        for i, data_batch in enumerate(data_loaders[0]):
            
          
            cls_scores0 = model(**data_batch[0])
            cls_scores1 = model(**data_batch[1])
            expert1_logits_output0 = cls_scores0[0]
            expert2_logits_output0 = cls_scores0[1]
            expert3_logits_output0 = cls_scores0[2]
            expert1_logits_output1 = cls_scores1[0]
            expert2_logits_output1 = cls_scores1[1]
            expert3_logits_output1 = cls_scores1[2]

            aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
            aggregation_output0 = aggregation_softmax[0].cuda() * expert1_logits_output0 + aggregation_softmax[1].cuda() * expert2_logits_output0 + aggregation_softmax[2].cuda() * expert3_logits_output0
            aggregation_output1 = aggregation_softmax[0].cuda() * expert1_logits_output1 + aggregation_softmax[1].cuda() * expert2_logits_output1 + aggregation_softmax[2].cuda() * expert3_logits_output1
            softmax_aggregation_output0 = F.softmax(aggregation_output0, dim=1) 
            softmax_aggregation_output1 = F.softmax(aggregation_output1, dim=1)
        
        # SSL loss: similarity maxmization
            cos_similarity = cos(softmax_aggregation_output0, softmax_aggregation_output1).mean()
            ssl_loss =  cos_similarity
        
        # Entropy regularizer: entropy maxmization
            loss =  -ssl_loss 
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%10==0:
                print("Aggregation weight: Expert 1 is {0:.2f}, Expert 2 is {1:.2f}, Expert 3 is {2:.2f}".format(aggregation_weight[0], aggregation_weight[1], aggregation_weight[2]))



    # train_detector(
    #     model,
    #     datasets,
    #     cfg,
    #     distributed=distributed,
    #     validate=args.validate,
    #     logger=logger)


if __name__ == '__main__':
    main()
