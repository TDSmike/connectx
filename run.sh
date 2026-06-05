#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1          # 改成你要用的空卡
PY=/home/ubuntu/miniconda3/envs/mingxiang/bin/python
STEPS=300000

$PY train.py --exp ppo  --steps $STEPS
$PY train.py --exp grpo --steps $STEPS
$PY compare.py --kaggle
