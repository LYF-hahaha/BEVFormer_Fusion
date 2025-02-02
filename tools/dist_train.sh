#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
RESUME=$3
PORT=${PORT:-28509}

# 在命令行最开始加下面这个，即仅有2、3两个硬件可见
# python会把2，3当成0，1
CUDA_VISIBLE_DEVICES=2,3,4,5

# 原工作默认
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#   --use_env $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic

# 首次开始训练
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env $(dirname "$0")/train.py $CONFIG --resume-from=$3 --launcher pytorch ${@:3} --deterministic >> nohup_base_semi_fusion_e6d6.out 2>&1 &


# 出幺蛾子后需要恢复的话
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env $(dirname "$0")/train.py $CONFIG --resume-from='data_nas/LYF_bu/semi_fusion_e6d6/epoch_7.pth' --launcher pytorch ${@:3} --deterministic >> nohup_base_semi_fusion_e6d6.out 2>&1 &

