#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
RESUME=$3
PORT=${PORT:-28509}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#   --use_env $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env $(dirname "$0")/train.py $CONFIG --resume-from=$RESUME --launcher pytorch ${@:3} --deterministic >> nohup_small.out 2>&1 &


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python $(dirname "$0")/train.py $CONFIG --gpus=$GPUS --resume-from=$RESUME 
