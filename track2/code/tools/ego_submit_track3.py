# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_root_logger, setup_multi_processes,
                         update_data_root)
from mmdet.datasets.pipelines import Compose
from devkit_tools.benchmarks import challenge_instance_detection_benchmark
from devkit_tools.datasets.challenge_detection_dataset import ChallengeDetectionDataset
import glob, shutil

DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/track2_datasets'

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        'workdir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        'datadir',
        help='the directory of ego dataset')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    workdir = args.workdir
    config = glob.glob(os.path.join(workdir,'*.py'))[-1]
    checkpoints = glob.glob(os.path.join(workdir,'exp_*.pth'))
    assert len(checkpoints)==5,'exp pth file save error!!!'
    checkpoints = sorted(checkpoints)
    
    cfg = Config.fromfile(config)
    # set multi-process settings
    setup_multi_processes(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()


    train_transform = Compose(cfg.data.train.pipeline)
    eval_transform = Compose(cfg.data.val.pipeline)
    dataset = ChallengeDetectionDataset(
        args.datadir,
        train=False,
        transform=eval_transform,
        eval_submit=True)
    dataset.CLASSES = list(range(1110))
    '''
    if rank == 0: print(f'loading benchmark ...')
    benchmark = challenge_instance_detection_benchmark(
        dataset_path=args.datadir,
        train_transform=train_transform,
        eval_transform=eval_transform,
        eval_submit=True
    )

    cls_num = benchmark.n_classes
    if rank == 0: print(f'Benchmark class number : {cls_num}')

    dataset = benchmark.test_stream[0].dataset
    '''
    dataset_size = len(dataset)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    for i,ckpt in enumerate(checkpoints):
        if rank == 0: print(f'test experience : {i}, test dataset: {dataset_size}') 
        
        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        
        checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model) 
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                      args.show_score_thr)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)

        rank, _ = get_dist_info()
        if rank == 0:
            kwargs = {}
            dataset.format_results(outputs, **kwargs)        
            cur_json = './test.json' 
            save_json = os.path.join(args.workdir,f'track3_output_eval_exp{i}.json') 
            shutil.move(cur_json,save_json)          
            print(f'exp results save in {save_json}!!!')

if __name__ == '__main__':
    main()
