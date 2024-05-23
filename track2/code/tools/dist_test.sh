#!/usr/bin/env bash

export TORCH_HOME=/youtu/fuxi-team2/persons/niceliu/models
export PYTHONPATH=/youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/mmdetection/code/avalanche:PYTHONPATH
DATASET_PATH=/youtu/fuxi-team2-2/CLVision/track2_datasets

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/ego_test.py $CONFIG $CHECKPOINT $DATASET_PATH --launcher pytorch ${@:4}
