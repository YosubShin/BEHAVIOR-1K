"""Convert BEHAVIOR-1K LeRobot v3 demos -> EXPO-FT offline demo format.

EXPO-FT's process_droid_dataset (expo_ft/env/droid_utils.py) expects:
    <out_dir>/<episode_idx>/traj.hdf5 with
        saved_observation/base_image        (T, 224, 224, 3) uint8
        saved_observation/left_wrist_image  (T, 224, 224, 3) uint8
        saved_observation/right_wrist_image (T, 224, 224, 3) uint8
        saved_observation/state             (T, 23) float32
        action/<action_space>               (T, A1)
        action/gripper_<gripper_action_space> (T, A2)   [concatenated -> action]

We store the full 23-dim B1K action as action/joint_full and a zero-width
action/gripper_empty so the loader's concat([a1, a2]) yields (T, 23) without
forking their loader. Task config must set:
    config.action_space = "joint_full"; config.gripper_action_space = "empty"

Rewards/dones are NOT stored — process_droid_dataset synthesizes terminal
reward 1.0 (all demos are successes).

Run in the openpi venv (has the wensi-ai lerobot fork that reads these demos):
    /mnt/nvme/openpi/.venv/bin/python convert_demos_to_expoft.py \
        --dataset-root /mnt/nas/2026-challenge-demos \
        --task turning_on_radio --episodes 20 \
        --out /mnt/nvme/expoft_demos/turning_on_radio

Sizing: ~2k steps/radio episode x 3 cams x 224^2 uint8 ~= 0.9 GB/ep raw,
~0.3-0.5 GB gzipped -> 20 eps ~= 6-10 GB. Keep on nvme.
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd


def extract_state_23(proprio: np.ndarray) -> np.ndarray:
    """61-dim proprio -> 23-dim state (same layout as openpi b1k RobotConfig)."""
    parts = [
        proprio[..., 0:3],
        proprio[..., 53:57],
        proprio[..., 3:10],
        proprio[..., 24:26].sum(axis=-1, keepdims=True),
        proprio[..., 28:35],
        proprio[..., 49:51].sum(axis=-1, keepdims=True),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def to_uint8_hwc(img) -> np.ndarray:
    """LeRobot video frames come as torch float32 CHW in [0,1]; normalize to uint8 HWC."""
    import torch

    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return img[..., :3]


def resize_224(img: np.ndarray) -> np.ndarray:
    import cv2

    if img.shape[0] == 224 and img.shape[1] == 224:
        return img
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


CAM_KEYS = {
    "base_image": "observation.rgb.zed_link_camera_0",
    "left_wrist_image": "observation.rgb.left_realsense_link_camera_0",
    "right_wrist_image": "observation.rgb.right_realsense_link_camera_0",
}


def find_task_episodes(dataset_root: str, task_name: str) -> list[int]:
    tasks = pd.read_parquet(os.path.join(dataset_root, "meta", "tasks.parquet"))
    # tasks.parquet: index or column mapping task string -> task_index
    if "task_index" in tasks.columns:
        # task string may be the index or a column
        if task_name in tasks.index:
            task_index = int(tasks.loc[task_name, "task_index"])
        else:
            col = [c for c in tasks.columns if tasks[c].dtype == object][0]
            task_index = int(tasks.loc[tasks[col].str.contains(task_name, case=False), "task_index"].iloc[0])
    else:
        raise RuntimeError(f"Unrecognized tasks.parquet schema: {tasks.columns.tolist()} / index={tasks.index[:3]}")

    ep_dir = os.path.join(dataset_root, "meta", "episodes")
    frames = []
    for root, _, files in os.walk(ep_dir):
        for f in sorted(files):
            if f.endswith(".parquet"):
                frames.append(pd.read_parquet(os.path.join(root, f)))
    episodes = pd.concat(frames, ignore_index=True)
    # episode-level task association: column name differs across versions
    for col in ("task_index", "tasks"):
        if col in episodes.columns:
            if col == "task_index":
                sel = episodes.loc[episodes[col] == task_index, "episode_index"]
            else:  # list/str of task names
                sel = episodes.loc[
                    episodes[col].astype(str).str.contains(task_name, case=False), "episode_index"
                ]
            return sorted(int(e) for e in sel.tolist())
    raise RuntimeError(f"No task column in episodes meta: {episodes.columns.tolist()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="/mnt/nas/2026-challenge-demos")
    p.add_argument("--task", default="turning_on_radio")
    p.add_argument("--episodes", type=int, default=20, help="number of demo episodes to convert")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ep_indices = find_task_episodes(args.dataset_root, args.task)
    print(f"[convert] task '{args.task}': {len(ep_indices)} episodes in dataset; converting first {args.episodes}")
    ep_indices = ep_indices[: args.episodes]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="behavior-1k/2026-challenge-demos", root=args.dataset_root, episodes=ep_indices)
    print(f"[convert] LeRobotDataset loaded: {ds.num_episodes} episodes, {ds.num_frames} frames")

    os.makedirs(args.out, exist_ok=True)
    # group frame indices by episode
    cur_ep, buf, out_idx = None, None, 0

    def flush(ep_buf, idx):
        ep_dir = os.path.join(args.out, str(idx))
        os.makedirs(ep_dir, exist_ok=True)
        T = len(ep_buf["state"])
        with h5py.File(os.path.join(ep_dir, "traj.hdf5"), "w") as f:
            g = f.create_group("saved_observation")
            for k in ("base_image", "left_wrist_image", "right_wrist_image"):
                g.create_dataset(
                    k, data=np.stack(ep_buf[k]), compression="gzip", compression_opts=4, chunks=(1, 224, 224, 3)
                )
            g.create_dataset("state", data=np.stack(ep_buf["state"]))
            a = f.create_group("action")
            a.create_dataset("joint_full", data=np.stack(ep_buf["action"]).astype(np.float32))
            a.create_dataset("gripper_empty", data=np.zeros((T, 0), dtype=np.float32))
        print(f"[convert] wrote episode {idx}: {T} steps -> {ep_dir}/traj.hdf5")

    for i in range(len(ds)):
        item = ds[i]
        ep = int(item["episode_index"])
        if ep != cur_ep:
            if buf is not None:
                flush(buf, out_idx)
                out_idx += 1
            cur_ep, buf = ep, {k: [] for k in ("base_image", "left_wrist_image", "right_wrist_image", "state", "action")}
        for out_key, ds_key in CAM_KEYS.items():
            buf[out_key].append(resize_224(to_uint8_hwc(item[ds_key])))
        state61 = np.asarray(item["observation.state"], dtype=np.float32)
        buf["state"].append(extract_state_23(state61))
        buf["action"].append(np.asarray(item["action"], dtype=np.float32))
    if buf is not None:
        flush(buf, out_idx)

    print(f"[convert] DONE: {out_idx + 1} episodes -> {args.out}")


if __name__ == "__main__":
    main()
