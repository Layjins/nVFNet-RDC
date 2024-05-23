# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__

from mmdet.apis.train_v2 import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_root_logger, setup_multi_processes,
                         update_data_root)

from devkit_tools.challenge_constants import \
    DEFAULT_DEMO_TEST_JSON, \
    DEFAULT_DEMO_TRAIN_JSON, DEMO_DETECTION_EXPERIENCES, \
    CHALLENGE_DETECTION_EXPERIENCES, CHALLENGE_DETECTION_FORCED_TRANSFORMS, \
    DEFAULT_CHALLENGE_TRAIN_JSON, DEFAULT_CHALLENGE_TEST_JSON, \
    DEFAULT_CHALLENGE_CLASS_ORDER_SEED
from merge_json_clv import merge_json_clv
from split_json_clv import split_json_clv

import pdb
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('dataDir', help='ego dataset root path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.work_dir = cfg.work_dir + '_' + str(timestamp)
    #print(f'work_dir={cfg.work_dir}')
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    #timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # TRAINING LOOP
    logger.info(f'Starting experiment for track2...')
    # benchmark 
    cfg.data.val.ann_file = os.path.join(args.dataDir, DEFAULT_CHALLENGE_TEST_JSON)
    cfg.data.val.img_prefix = os.path.join(args.dataDir,'images/')
    val_dataset = build_dataset(cfg.data.val)

    cls_num = len(val_dataset.CLASSES)
    logger.info(f'Benchmark class number : {cls_num}') 
    

    if len(cfg.gpu_ids) > 1:
        local_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
    # generate replay json
    if cfg.replay.replay_flag:
        replay_json_dir = 'replay_json_dir/' + timestamp
        replay_json_path = args.dataDir + '/' + replay_json_dir
    if cfg.replay.replay_flag and local_rank==0:
        weight_scale = 1.0
        os.makedirs(replay_json_path)
        for exp_id in range(5):
            if exp_id ==0:
                # split0
                train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0.json')
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0_split.json')
                splited_json_info = split_json_clv(args.dataDir + '/' + train_json_name, cfg.replay.replay_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
            elif exp_id ==1:
                # merge0
                train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1.json')
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0_split.json')
                merged_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1_merge.json')
                merged_json_info = merge_json_clv(args.dataDir + '/' + train_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # split1
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1_split.json')
                split1_img_num = int(cfg.replay.replay_img_num//2//weight_scale)
                splited_json_info = split_json_clv(args.dataDir + '/' + train_json_name, split1_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # split0
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0_split.json')
                split0_img_num = cfg.replay.replay_img_num - split1_img_num
                splited_json_info = split_json_clv(replay_json_path + '/' + splited_json_name, split0_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # out train json
                train_json_name = replay_json_dir + '/' + merged_json_name
            elif exp_id ==2:
                # merge0
                train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp2.json')
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0_split.json')
                merged_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp2_merge.json')
                merged_json_info = merge_json_clv(args.dataDir + '/' + train_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # merge1
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1_split.json')
                merged_json_info = merge_json_clv(replay_json_path + '/' + merged_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # split2
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp2_split.json')
                split2_img_num = int(cfg.replay.replay_img_num//3//weight_scale)
                split2_donate_num = int(cfg.replay.replay_img_num//3) - split2_img_num
                splited_json_info = split_json_clv(args.dataDir + '/' + train_json_name, split2_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # split1
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1_split.json')
                split1_img_num = int(cfg.replay.replay_img_num//3//weight_scale) + int(split2_donate_num//2)
                splited_json_info = split_json_clv(replay_json_path + '/' + splited_json_name, split1_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # split0
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0_split.json')
                split0_img_num = cfg.replay.replay_img_num - split1_img_num - split2_img_num
                splited_json_info = split_json_clv(replay_json_path + '/' + splited_json_name, split0_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # out train json
                train_json_name = replay_json_dir + '/' + merged_json_name
            elif exp_id ==3:
                # merge0
                train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp3.json')
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0_split.json')
                merged_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp3_merge.json')
                merged_json_info = merge_json_clv(args.dataDir + '/' + train_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # merge1
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1_split.json')
                merged_json_info = merge_json_clv(replay_json_path + '/' + merged_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # merge2
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp2_split.json')
                merged_json_info = merge_json_clv(replay_json_path + '/' + merged_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # split3
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp3_split.json')
                split3_img_num = int(cfg.replay.replay_img_num//4//weight_scale)
                split3_donate_num = int(cfg.replay.replay_img_num//4) - split3_img_num
                splited_json_info = split_json_clv(args.dataDir + '/' + train_json_name, split3_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # split2
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp2_split.json')
                split2_img_num = int(cfg.replay.replay_img_num//4//weight_scale)
                split2_donate_num = int(cfg.replay.replay_img_num//4) - split2_img_num
                split2_img_num = split2_img_num + int(split3_donate_num//3)
                splited_json_info = split_json_clv(replay_json_path + '/' + splited_json_name, split2_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # split1
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1_split.json')
                split1_img_num = int(cfg.replay.replay_img_num//4//weight_scale)
                split1_donate_num = int(cfg.replay.replay_img_num//4) - split1_img_num
                split1_img_num = split1_img_num + int(split3_donate_num//3)  + int(split2_donate_num//2)
                splited_json_info = split_json_clv(replay_json_path + '/' + splited_json_name, split1_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # split0
                splited_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0_split.json')
                split0_img_num = cfg.replay.replay_img_num - split1_img_num - split2_img_num - split3_img_num
                splited_json_info = split_json_clv(replay_json_path + '/' + splited_json_name, split0_img_num, splited_json_path=replay_json_path + '/' + splited_json_name, split_mode=cfg.replay.replay_mode)
                # out train json
                train_json_name = replay_json_dir + '/' + merged_json_name
            elif exp_id ==4:
                # merge0
                train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp4.json')
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp0_split.json')
                merged_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp4_merge.json')
                merged_json_info = merge_json_clv(args.dataDir + '/' + train_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # merge1
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1_split.json')
                merged_json_info = merge_json_clv(replay_json_path + '/' + merged_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # merge2
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp2_split.json')
                merged_json_info = merge_json_clv(replay_json_path + '/' + merged_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # merge3
                splited_json_name =  DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp3_split.json')
                merged_json_info = merge_json_clv(replay_json_path + '/' + merged_json_name, replay_json_path + '/' + splited_json_name, merged_json_path=replay_json_path + '/' + merged_json_name)
                # out train json
                train_json_name = replay_json_dir + '/' + merged_json_name


    for exp_id in range(CHALLENGE_DETECTION_EXPERIENCES):
        if CHALLENGE_DETECTION_EXPERIENCES == 5:
            train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp' + str(exp_id)+'.json')

            if cfg.replay.replay_flag:
                if exp_id ==0:
                    train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp' + str(exp_id)+'.json')
                elif exp_id ==1:
                    train_json_name = replay_json_dir + '/' + DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp1_merge.json')
                elif exp_id ==2:
                    train_json_name = replay_json_dir + '/' + DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp2_merge.json')
                elif exp_id ==3:
                    train_json_name = replay_json_dir + '/' + DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp3_merge.json')
                elif exp_id ==4:
                    train_json_name = replay_json_dir + '/' + DEFAULT_CHALLENGE_TRAIN_JSON.replace('.json', '_exp4_merge.json')
            
            logger.info(f'train_json_name={train_json_name}')
            if cfg.usingClassBalanced:
                cfg.data.train.dataset.ann_file = os.path.join(args.dataDir, train_json_name)
                cfg.data.train.dataset.img_prefix = os.path.join(args.dataDir,'images/')
            else:
                cfg.data.train.ann_file = os.path.join(args.dataDir, train_json_name)
                cfg.data.train.img_prefix = os.path.join(args.dataDir,'images/')
            train_dataset = build_dataset(cfg.data.train)
        else:
            if cfg.usingClassBalanced:
                cfg.data.train.dataset.ann_file = os.path.join(args.dataDir, DEFAULT_CHALLENGE_TRAIN_JSON)
                cfg.data.train.dataset.img_prefix = os.path.join(args.dataDir,'images/')
            else:
                cfg.data.train.ann_file = os.path.join(args.dataDir, DEFAULT_CHALLENGE_TRAIN_JSON)
                cfg.data.train.img_prefix = os.path.join(args.dataDir,'images/')
            train_dataset = build_dataset(cfg.data.train)
            
        train_dataset_size = len(train_dataset)
        logger.info(f'Start of experience : {exp_id}, train dataset: {train_dataset_size}') 

        if cfg.knowledge_distill.kd_flag and exp_id>0:
            train_detector(
                        model,
                        [train_dataset],
                        cfg,
                        distributed=distributed,
                        validate=(not args.no_validate),
                        timestamp=timestamp,
                        meta=meta,
                        exp_id=exp_id,
                        teacher_cfg_path=args.config,
                        teacher_checkpoint=cfg.work_dir + '/' + 'exp_'+str(exp_id-1)+'.pth')
        else:
            train_detector(
                        model,
                        [train_dataset],
                        cfg,
                        distributed=distributed,
                        validate=(not args.no_validate),
                        timestamp=timestamp,
                        meta=meta,
                        exp_id=exp_id)
        ckpt_path = os.path.join(cfg.work_dir,'latest.pth')
        exp_ckpt = ckpt_path.replace('latest.pth','exp_'+str(exp_id)+'.pth')
        shutil.copy(ckpt_path,exp_ckpt)
    

if __name__ == '__main__':
    main()
