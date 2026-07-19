#!/bin/bash
# Mid-flight gate eval for the Delta radio-reproduction run.
# Usage: gate_eval_ckpt.sh <config_name> <exp_name> <step>
# e.g.:  gate_eval_ckpt.sh pi05_b1k_delta radio_repro 10000
# Pulls the checkpoint from Delta scratch, serves it with replan-32 on :8020,
# and runs the 20-episode full-task eval (instances 0-4). Targets:
#   official baseline @10-50K: ~40% @ replan-32 (12/30), 15% @ replan-16.
set -euo pipefail
CFG=$1; EXP=$2; STEP=$3
LOCAL=/mnt/nvme/repro_ckpts/${EXP}_${STEP}
mkdir -p "$LOCAL"
echo "== pulling checkpoint step $STEP from Delta (params only)..."
rsync -az --info=progress2 "delta:/scratch/bchy/shin1/b1k/ckpts/${CFG}/${EXP}/${STEP}/params" "$LOCAL/"
# serve needs assets/norm_stats alongside: reuse official (bitwise-identical)
mkdir -p "$LOCAL/assets/turning_on_radio"
cp /mnt/nvme/pi05_pretrained/pi05_turn_on_the_radio/assets/turning_on_radio/norm_stats.json "$LOCAL/assets/turning_on_radio/"

echo "== serving on :8020 (replan-32)..."
cd /mnt/nvme/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 nohup .venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio \
  --policy.config pi05_b1k --policy.dir "$LOCAL" \
  --control_mode receding_horizon --action_horizon 32 --port 8020 \
  > /mnt/nvme/repro_ckpts/serve_${EXP}_${STEP}.log 2>&1 &
echo "serve pid $!; wait ~60s for load, then:"
echo "OMNIGIBSON_DATA_PATH=/mnt/nvme/behavior_data OMNIGIBSON_HEADLESS=1 conda run -n behavior python -m omnigibson.eval.eval \\"
echo "  --task-name turning_on_radio --robot-config OmniGibson/omnigibson/eval/r1pro.yaml \\"
echo "  --mode train --instance-indices 0 1 2 3 4 --host 127.0.0.1 --port 8020 \\"
echo "  --output-dir outputs/gate_${EXP}_${STEP} --write-video"
