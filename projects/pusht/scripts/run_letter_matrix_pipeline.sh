#!/bin/bash
# Sequential TDMPC2 PushT training: T, L, K, S at 500k each, then DR at 2M.
# Match SAC matrix env (contact_gated + log_barrier + AR=2).
set -e
cd /home/stevenman/Desktop/Work/Research/jax-learning
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

COMMON_KWARGS='{"obs_type":"keypoints","reward_mode":"contact_gated","coverage_shape":"log_barrier","coverage_eps":0.01}'

run_shape() {
  local shape=$1
  local steps=$2
  local ckpt=".temp/tdmpc2_pusht_${shape}_${steps}"
  local kwargs=$(echo "$COMMON_KWARGS" | python3 -c "import json,sys; d=json.load(sys.stdin); d['block_shape']='$shape'; print(json.dumps(d))")
  echo "=== [$(date +%H:%M:%S)] tdmpc2 PushT shape=$shape steps=$steps ckpt=$ckpt ==="
  uv run python scripts/train_tdmpc2.py \
    --env PushT \
    --total-timesteps $steps \
    --num-envs 8 \
    --seed 0 \
    --eval-every 50000 \
    --ckpt-dir "$ckpt" \
    --env-kwargs "$kwargs" \
    --wandb --wandb-project pusht-tdmpc2-letter \
    2>&1 | tee ".temp/logs/tdmpc2_pusht_${shape}_${steps}.log"
  echo "=== [$(date +%H:%M:%S)] done $shape ==="
}

mkdir -p .temp/logs

run_shape tee 500000
run_shape l 500000
run_shape k 500000
run_shape s 500000
run_shape dr 2000000

echo "=== ALL TRAINING DONE [$(date +%H:%M:%S)] ==="
