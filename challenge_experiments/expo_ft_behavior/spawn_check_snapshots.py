"""Render the spawn view of every snapshot in a pool for visual verification.

For each .pt in --snapshot-dir: reset the env server to that exact state and
save the head-camera first observation. Output: one PNG per snapshot plus a
labeled contact sheet — spot-check that starts are base-parked, radio in
reach, arms mid-manipulation (the nav-end regime) BEFORE training on them.

Run in the `behavior` conda env against a live behavior_env_server:
    python spawn_check_snapshots.py --env-port 8102 \
        --snapshot-dir /mnt/nvme/expoft_snapshots/miniradio_navend_own \
        --out /mnt/nvme/expoft_debug/navend_own_views
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import websockets.sync.client

from omnigibson.eval.utils.network_utils import Packer, unpackb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-host", default="localhost")
    p.add_argument("--env-port", type=int, default=8102)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ws = websockets.sync.client.connect(
        f"ws://{args.env_host}:{args.env_port}", max_size=None, open_timeout=600
    )
    packer = Packer()

    def call(operation: str, **kwargs) -> dict:
        ws.send(packer.pack({"operation": operation, **kwargs}))
        return unpackb(ws.recv())

    env_id = call("create_env")["env_id"]
    snaps = sorted(glob.glob(f"{args.snapshot_dir}/*.pt"))
    print(f"[spawn-check] {len(snaps)} snapshots")

    import cv2

    tiles = []
    for sp in snaps:
        call("reset", env_id=env_id, snapshot_path=sp)
        obs = call("get_observation", env_id=env_id)["observation"]
        img = np.concatenate(
            [obs["left_wrist_image"], obs["base_image"], obs["right_wrist_image"]], axis=1
        )
        name = os.path.basename(sp).replace(".pt", "")
        cv2.imwrite(f"{args.out}/{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        labeled = img.copy()
        cv2.rectangle(labeled, (0, 0), (330, 22), (0, 0, 0), -1)
        cv2.putText(labeled, name[-22:], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(labeled)
        print(f"[spawn-check] {name}", flush=True)

    rows = [np.concatenate(tiles[i : i + 1], axis=1) for i in range(len(tiles))]
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(f"{args.out}/contact_sheet.png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[spawn-check] contact sheet: {args.out}/contact_sheet.png", flush=True)


if __name__ == "__main__":
    main()
