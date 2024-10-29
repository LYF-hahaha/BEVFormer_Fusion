#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

# torch 1.9.0以上版本改launch为run了
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
