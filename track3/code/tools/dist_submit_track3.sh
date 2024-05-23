#!/usr/bin/env bash

export TORCH_HOME=/youtu/fuxi-team2/persons/niceliu/models
export PYTHONPATH=/youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/mmdetection_track3/mmdetection/code/avalanche:PYTHONPATH

DATASET_PATH=/youtu/fuxi-team2-2/CLVision/submit_datasets

WORKDIR=$1
GPUS=$2
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/ego_submit_track3.py $WORKDIR $DATASET_PATH --launcher pytorch ${@:4}

