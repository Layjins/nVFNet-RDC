#!/usr/bin/env bash

export TORCH_HOME=/youtu/fuxi-team2/persons/niceliu/models
export PYTHONPATH=/youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/mmdetection_track3/mmdetection/code/avalanche:PYTHONPATH

DATASET_PATH=/youtu/fuxi-team2-2/CLVision/submit_datasets
CKPT_DIR=/youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/checkpoints

CONFIG=$1
GPUS=$2
PORT=${PORT:-26500}
checkpoint_dir=$CKPT_DIR/${CONFIG##*/}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --cfg-options work_dir=$checkpoint_dir \
    --launcher pytorch ${@:3}
