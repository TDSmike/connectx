#!/usr/bin/env bash
# One-command pipeline:
#   1. train PPO and GRPO in parallel,
#   2. write training logs/checkpoints/curves,
#   3. generate plots and faithful Kaggle built-in comparisons.
#
# Usage:
#   bash run.sh
#   STEPS=500000 EVAL_GAMES=200 bash run.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY=${PY:-/home/ubuntu/miniconda3/envs/mingxiang/bin/python}
STEPS=${STEPS:-300000}
EVAL_EVERY=${EVAL_EVERY:-20000}
EVAL_GAMES=${EVAL_GAMES:-100}
NEG_DEPTH=${NEG_DEPTH:-4}
COMPARE_GAMES=${COMPARE_GAMES:-400}
KAGGLE_EPISODES=${KAGGLE_EPISODES:-100}
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-course}
PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
export MPLCONFIGDIR PYTHONDONTWRITEBYTECODE

mkdir -p checkpoints results "$MPLCONFIGDIR"

GPU_A=${GPU_A:-}
GPU_B=${GPU_B:-}
if [[ -z "$GPU_A" || -z "$GPU_B" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t, -k2 -nr | cut -d, -f1 | tr -d ' ')
        GPU_A=${GPU_A:-${GPUS[0]:-}}
        GPU_B=${GPU_B:-${GPUS[1]:-${GPU_A}}}
    fi
fi

run_train() {
    local exp=$1
    local gpu=$2
    local log=$3
    local -a cmd=("$PY" train.py --exp "$exp" --steps "$STEPS"
        --eval-every "$EVAL_EVERY" --eval-games "$EVAL_GAMES"
        --negamax-depth "$NEG_DEPTH")

    if [[ -n "$gpu" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1
    else
        "${cmd[@]}" > "$log" 2>&1
    fi
}

echo "python=$PY"
echo "steps=$STEPS eval_every=$EVAL_EVERY eval_games=$EVAL_GAMES neg_depth=$NEG_DEPTH"
echo "compare_games=$COMPARE_GAMES kaggle_episodes=$KAGGLE_EPISODES"
echo "PPO -> GPU ${GPU_A:-cpu} | GRPO -> GPU ${GPU_B:-cpu}"

run_train ppo "$GPU_A" results/train_ppo.log &
PID_PPO=$!
run_train grpo "$GPU_B" results/train_grpo.log &
PID_GRPO=$!

echo "ppo  pid=$PID_PPO -> results/train_ppo.log"
echo "grpo pid=$PID_GRPO -> results/train_grpo.log"

status=0
if wait "$PID_PPO"; then
    echo "[done] PPO"
else
    echo "[failed] PPO; see results/train_ppo.log" >&2
    status=1
fi

if wait "$PID_GRPO"; then
    echo "[done] GRPO"
else
    echo "[failed] GRPO; see results/train_grpo.log" >&2
    status=1
fi

if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi

echo "=== comparing and plotting ==="
if [[ -n "$GPU_A" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_A" "$PY" compare.py --games "$COMPARE_GAMES" \
        --kaggle --kaggle-episodes "$KAGGLE_EPISODES" 2>&1 | tee results/compare.log
else
    "$PY" compare.py --games "$COMPARE_GAMES" \
        --kaggle --kaggle-episodes "$KAGGLE_EPISODES" 2>&1 | tee results/compare.log
fi

echo "ALL DONE"
echo "plots: results/compare.png results/loss_curves.png"
echo "summary: results/compare.log"
