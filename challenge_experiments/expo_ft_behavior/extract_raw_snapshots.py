"""Extract exact T-450 sim states from raw episode files -> mini-task start snapshots.

Verifies raw<->LeRobot episode mapping via action-stream equality, loads each raw
serialized state into the live sim, and saves og.sim.dump_state(serialized=False)
dicts compatible with behavior_env_server's --start-snapshot-dir loader.

Run in the behavior conda env with OMNIGIBSON_DATA_PATH set (sim must be free):
    python extract_raw_snapshots.py --episodes 20 --back 450 \
        --out /mnt/nvme/expoft_snapshots/miniradio_exact --probe /tmp/miniprobe
"""

import argparse
import glob
import os

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="/mnt/nvme/b1k_rawdata/task-0000")
    p.add_argument("--mini-dir", default="/mnt/nvme/expoft_demos/turning_on_radio_mini450")
    p.add_argument("--task", default="turning_on_radio")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--back", type=int, default=450, help="steps before episode end")
    p.add_argument("--out", required=True)
    p.add_argument("--probe", default=None, help="save a rendered probe PNG per episode here")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.probe:
        os.makedirs(args.probe, exist_ok=True)

    raw_files = sorted(glob.glob(f"{args.raw_dir}/episode_*.hdf5"))
    print(f"[extract] {len(raw_files)} raw files")

    # boot sim via the same env class the server uses
    from behavior_env_server import BehaviorEnvOps

    env = BehaviorEnvOps(task_name=args.task, instance_ids=[0], max_steps=450)
    env.reset()

    import cv2
    import omnigibson as og
    import torch as th

    ok = 0
    for i, raw_path in enumerate(raw_files[: args.episodes]):
        with h5py.File(raw_path) as f:
            demo = f["data"][list(f["data"].keys())[0]]
            raw_actions = np.asarray(demo["action"])
            states = demo["state"]
            state_sizes = np.asarray(demo["state_size"])
            T = raw_actions.shape[0]
            idx = max(0, T - args.back)
            state_row = np.asarray(states[idx][: state_sizes[idx]], dtype=np.float32)

        # mapping check: raw actions == converted lerobot actions (same episode)?
        mini_path = f"{args.mini_dir}/{i}/traj.hdf5"
        if os.path.exists(mini_path):
            with h5py.File(mini_path) as mf:
                mini_act = np.asarray(mf["action"]["joint_full"])  # last `back` steps of full ep
            match = np.allclose(raw_actions[-mini_act.shape[0] :], mini_act, atol=1e-5)
        else:
            match = None

        try:
            og.sim.load_state(th.from_numpy(state_row), serialized=True)
            for _ in range(5):
                og.sim.step_physics()
            for _ in range(10):
                og.sim.render()
            snap = og.sim.dump_state(serialized=False)
            th.save(snap, f"{args.out}/{args.task}_rawep{i:03d}_t{idx:05d}.pt")
            ok += 1
            status = "OK"
            if args.probe:
                obs, _ = env.env.get_obs()
                env.evaluator.obs = env.evaluator._preprocess_obs(obs)
                img = env._observation()["base_image"]
                cv2.imwrite(f"{args.probe}/exact_ep{i}_base.png", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        except Exception as e:
            status = f"LOAD FAILED: {e}"
        print(f"[extract] ep{i} ({os.path.basename(raw_path)}): T={T} idx={idx} "
              f"state_dim={state_row.shape[0]} action_match={match} -> {status}", flush=True)

    print(f"[extract] DONE: {ok}/{min(args.episodes, len(raw_files))} snapshots -> {args.out}")


if __name__ == "__main__":
    main()
