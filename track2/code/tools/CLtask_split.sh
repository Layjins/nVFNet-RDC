#!/usr/bin/env bash

export TORCH_HOME=/youtu/fuxi-team2/persons/niceliu/models
export PYTHONPATH=/youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/mmdetection/code/avalanche:PYTHONPATH

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 tools/CLtask_split.py 