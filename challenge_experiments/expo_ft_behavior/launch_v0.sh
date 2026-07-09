#!/bin/bash
# EXPO-FT BEHAVIOR run v0 orchestrator: waits for demo conversion, then
# launches env server + learner. All logs under challenge_experiments/.
set -u
CE=/home/yosub/co/BEHAVIOR-1K/challenge_experiments
DEMOS=/mnt/nvme/expoft_demos/turning_on_radio

echo "[orch] waiting for demo conversion..."
until grep -q "CONVERT_FULL_EXIT=0" "$CE/convert_full.log" 2>/dev/null; do sleep 60; done
N_EP=$(ls "$DEMOS" | grep -cE '^[0-9]+$')
echo "[orch] conversion done: $N_EP episodes at $DEMOS ($(du -sh $DEMOS | cut -f1))"

echo "[orch] launching env server on :8102..."
(
  export OMNIGIBSON_DATA_PATH=/mnt/nvme/behavior_data OMNIGIBSON_HEADLESS=1
  cd "$CE/expo_ft_behavior"
  exec ~/miniconda3/condabin/conda run -n behavior --no-capture-output \
    python behavior_env_server.py --port 8102 \
    --task-name turning_on_radio --instance-ids 0 1 2 3 4 --perturb-pose
) > "$CE/v0_env_server.log" 2>&1 &
SERVER_PID=$!
echo "[orch] env server pid=$SERVER_PID"

# wait for the ws port to bind (server binds before the heavy scene load)
for i in $(seq 1 60); do
  ss -tln 2>/dev/null | grep -q ":8102 " && { echo "[orch] port 8102 up"; break; }
  sleep 5
done

echo "[orch] launching learner..."
(
  cd /mnt/nvme/expo-ft
  export WANDB_MODE=offline
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55   # leave room for the sim (~15GB) on the shared GPU
  exec .venv/bin/python train_pi_robo.py \
    --config configs/model/expo_ft_b1k_config.py \
    --config_task configs/task/behavior_radio.py \
    --dataset_path "$DEMOS" \
    --run_name expoft_b1k_radio_v0 \
    --output_dir /mnt/nvme/expoft_runs \
    --client_host localhost --client_port 8102 \
    --replan_steps 8 --batch_size 64 --utd_ratio 20 \
    --update_type episode --num_updates 50 \
    --checkpoint_model --checkpoint_interval 5000 \
    --max_steps 100000
) > "$CE/v0_learner.log" 2>&1 &
LEARNER_PID=$!
echo "[orch] learner pid=$LEARNER_PID"
echo "[orch] both launched. Tailing for liveness..."

# liveness summary loop (every 10 min, 12 hours max)
for i in $(seq 1 72); do
  sleep 600
  S=$(kill -0 $SERVER_PID 2>/dev/null && echo up || echo DOWN)
  L=$(kill -0 $LEARNER_PID 2>/dev/null && echo up || echo DOWN)
  STEPS=$(grep -oE "^ *[0-9]+%.*| [0-9]+/100000" "$CE/v0_learner.log" | tail -1)
  echo "[orch][t+$((i*10))min] server=$S learner=$L progress: ${STEPS:-n/a}"
  [ "$L" = "DOWN" ] && { echo "[orch] learner exited; stopping."; break; }
done
echo "ORCH_DONE"
