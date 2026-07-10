"""Direct (in-process) open-loop replay probe — no websocket layer.

Instantiates BehaviorEnvOps in this process, resets to each demo's extracted
start state, replays the demo's recorded actions, reports success/timeout.
Heartbeats every 100 steps so hangs are localizable; run with
PYTHONFAULTHANDLER=1 and an outer `timeout` for hard kills.

    OMNIGIBSON_DATA_PATH=... python -u replay_probe_direct.py \
        --starts /mnt/nvme/expoft_snapshots/demo_starts_radiorest \
        --trims /mnt/nvme/expoft_demos/turning_on_radio_radiorest --n 5
"""

import argparse
import glob
import time

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--starts", required=True)
    p.add_argument("--trims", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--extra-steps", type=int, default=120)
    args = p.parse_args()

    from behavior_env_server import BehaviorEnvOps

    print("[direct] creating env (scene load ~2min)...", flush=True)
    env = BehaviorEnvOps(
        task_name="turning_on_radio",
        instance_ids=[0],
        max_steps=2000,
        snapshot_record_dir=None,
        start_snapshot_dir=None,
    )
    print("[direct] env ready", flush=True)

    starts = sorted(glob.glob(f"{args.starts}/*.pt"))
    results = []
    for k in range(args.n):
        snap = starts[k]
        with h5py.File(f"{args.trims}/{k}/traj.hdf5") as f:
            actions = np.asarray(f["action"]["joint_full"])
        print(f"[direct] demo {k}: reset to {snap.split('/')[-1]} ({len(actions)} demo steps)", flush=True)
        t0 = time.time()
        env.reset(snapshot_path=snap)
        print(f"[direct]   reset done in {time.time()-t0:.1f}s; replaying...", flush=True)
        outcome, steps = "timeout", 0
        for t in range(len(actions) + args.extra_steps):
            a = actions[min(t, len(actions) - 1)]
            env.step(a.tolist())
            info = env.get_info_for_step()
            steps = t + 1
            if t % 100 == 99:
                print(f"[direct]   ...step {t+1} r={info['reward']:.3f}", flush=True)
            if info["done"]:
                outcome = "SUCCESS" if info["success"] else "done-fail"
                break
        results.append(outcome == "SUCCESS")
        print(f"[direct] demo {k}: {outcome} at step {steps}/{len(actions)}", flush=True)
    print(f"[direct] REPLAY RESULT: {sum(results)}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
