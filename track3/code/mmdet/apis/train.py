# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv_custom.runner import CustomEpochBasedRunner as EpochBasedRunner
from mmcv.runner import load_checkpoint
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
from mmdet.models import build_detector
from mmcv import Config
from mmdet.datasets.dataset_wrappers import ClassBalancedDataset, MultiImageMixDataset


def build_teacher(teacher_cfg_path,teacher_checkpoint):
    #teacher_cfg = Config.fromfile("configs/clv-challenge/vfnet_r50_fpn_1x_clv_json.py")
    teacher_cfg = Config.fromfile(teacher_cfg_path)
    teacher = build_detector(
        teacher_cfg.model, train_cfg=teacher_cfg.get('train_cfg'), test_cfg=teacher_cfg.get('test_cfg'))
    #load_checkpoint(teacher,
    #                "/youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/checkpoints/vfnet_r50_fpn_1x_clv_json.py_20220420_122421/exp_4.pth",
    #                map_location='cpu')
    load_checkpoint(teacher,
                    teacher_checkpoint,
                    map_location='cpu')
    return teacher

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True


def train_detector(cfg, 
                   model,
                   dataset,
                   val_dataset,
                   logger,
                   distributed,
                   meta,
                   timestamp,
                   exp_id,
                   teacher_cfg_path=None,
                   teacher_checkpoint=None):

    if teacher_cfg_path!=None:
        print("build teacher...")
        print("teacher_cfg_path=", teacher_cfg_path)
        print("teacher_checkpoint=", teacher_checkpoint)
        teacher = build_teacher(teacher_cfg_path,teacher_checkpoint)
    else:
        teacher = None

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    '''
    # ClassBalancedDataset
    oversample_thr = 1e-3
    dataset = [
        ClassBalancedDataset(ds, oversample_thr, filter_empty_gt=True)
        for ds in dataset
    ]
    '''
    '''
    # MultiImageMixDataset
    dataset = [
        MultiImageMixDataset(ds, cfg.train_pipeline2)
        for ds in dataset
    ]
    '''
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]
    # put model on gpus
    if teacher!=None:
        if distributed:
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            teacher = MMDistributedDataParallel(
                teacher.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                #find_unused_parameters=find_unused_parameters,
                find_unused_parameters=True)
        else:
            teacher = MMDataParallel(
                teacher.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    # build runner
    #if exp_id > 0: # finetuning
    #    cfg.optimizer.lr = 0.005
    #    cfg.runner.max_epochs = 24
    optimizer = build_optimizer(model, cfg.optimizer)
    if teacher_cfg_path==None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        print("run with kd...")
        runner = EpochBasedRunner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
            max_iters=None,
            max_epochs=cfg.runner.max_epochs,
            teacher=teacher,
            exp_id=exp_id
        )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp
    

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())
      
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    
    if not exp_id:
        lr_config = cfg.lr_config        
    else:        
        lr_config = dict(warmup=None,
                         warmup_iters=0,
                         warmup_ratio=0,
                         step=cfg.lr_config['step'],
                         policy='step')
     
    # register hooks
    runner.register_training_hooks(
        lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if not exp_id:
        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)

    # register eval hooks
    if val_dataset:
        dataset_size = len(val_dataset)
        logger.info(f'Start of experience : {exp_id}, val dataset: {dataset_size}') 
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')
 
    runner.run(data_loaders, cfg.workflow)
