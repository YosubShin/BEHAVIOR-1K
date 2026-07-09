#!/bin/bash
# Multi-process eval throughput benchmark (careful K=2, DefaultWrapper).
# Isolates stepping throughput from scene-load by subtracting a load calibration.
# NOTE: GPU is shared with a running lerobot-train job -> numbers are throughput-under-load.
set -u
DP=/mnt/nvme/behavior_data
BASE=/home/yosub/co/BEHAVIOR-1K
OUT=$BASE/challenge_experiments/parallel_bench
mkdir -p "$OUT"
CONDA=~/miniconda3/condabin/conda
STEPS=2000

run_eval() {  # name inst steps
  local name="$1" inst="$2" steps="$3"
  OMNIGIBSON_DATA_PATH=$DP OMNIGIBSON_HEADLESS=1 \
  $CONDA run -n behavior --no-capture-output python -m omnigibson.eval.eval \
    --task-name turning_on_radio --env-wrapper omnigibson.eval.wrappers.DefaultWrapper \
    --host 127.0.0.1 --port 8010 --mode public_test --instance-indices "$inst" \
    --max-steps "$steps" --output-dir "$OUT/out_$name" > "$OUT/$name.log" 2>&1
}
now(){ date +%s.%N; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -dc '0-9'; }

# background free-mem logger
( while true; do echo "$(date +%T) free_MiB=$(free_mib)"; sleep 5; done ) > "$OUT/gpu_mem.log" 2>&1 &
MON=$!
trap "kill $MON 2>/dev/null" EXIT

echo "=== CALIBRATION (load only: inst 0, 20 steps) ==="
t0=$(now); run_eval calib 0 20; t1=$(now)
T_LOAD=$(awk "BEGIN{print $t1-$t0}"); echo "T_LOAD=${T_LOAD}s"

echo "=== SINGLE (inst 0, ${STEPS} steps) ==="
t0=$(now); run_eval single 0 $STEPS; t1=$(now)
T_SINGLE=$(awk "BEGIN{print $t1-$t0}"); echo "T_SINGLE=${T_SINGLE}s"

# safety guard before parallel
FREE=$(free_mib); echo "free before parallel: ${FREE} MiB"
if [ "$FREE" -lt 18000 ]; then
  echo "ABORT parallel: only ${FREE} MiB free (<18000) — too risky for training job."
  T_PAR=""
else
  echo "=== PARALLEL K=2 (inst 0 & 1, ${STEPS} steps each) ==="
  t0=$(now)
  run_eval par0 0 $STEPS & q0=$!
  run_eval par1 1 $STEPS & q1=$!
  wait $q0 $q1
  t1=$(now)
  T_PAR=$(awk "BEGIN{print $t1-$t0}"); echo "T_PAR=${T_PAR}s"
fi

echo "=== RESULTS ==="
python3 - "$T_LOAD" "$T_SINGLE" "${T_PAR:-0}" "$STEPS" <<'PY'
import sys
tl,ts,tp,STEPS=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])
N=2
ss=ts-tl
print(f"load~={tl:.0f}s | single_total={ts:.0f}s")
print(f"stepping {STEPS} steps single-env: {ss:.0f}s -> {STEPS/ss:.1f} steps/s (1 env)")
if tp>0:
    sp=tp-tl
    print(f"parallel(K={N})_total={tp:.0f}s | stepping per-env: {sp:.0f}s")
    print(f"aggregate throughput: {N*STEPS/sp:.1f} steps/s")
    print(f"THROUGHPUT BENEFIT = {N*ss/sp:.2f}x  (ideal {N}.0, none 1.0)")
    print(f"end-to-end for {N} rollouts: serial~={tl+N*ss:.0f}s vs parallel={tp:.0f}s -> {(tl+N*ss)/tp:.2f}x speedup")
else:
    print("parallel phase skipped (memory guard)")
PY
echo "min free MiB during run:"; awk -F= '/free_MiB/{print $2}' "$OUT/gpu_mem.log" | sort -n | head -1
echo "PARBENCH_DONE"
