"""Grab the live env's current observation via a second ws connection and save PNGs."""

import sys
import time

import numpy as np
import websockets.sync.client

try:
    from openpi_client import msgpack_numpy

    Packer, unpackb = msgpack_numpy.Packer, msgpack_numpy.unpackb
except ImportError:
    from omnigibson.eval.utils.network_utils import Packer, unpackb

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe"
packer = Packer()
conn = websockets.sync.client.connect("ws://127.0.0.1:8102", compression=None, max_size=None, ping_interval=None)
# idempotent create_env returns the existing env id
conn.send(packer.pack({"operation": "create_env"}))
r = unpackb(conn.recv())
env_id = r["env_id"]
for i in range(3):  # a few samples a second apart
    conn.send(packer.pack({"operation": "get_observation", "env_id": env_id}))
    obs = unpackb(conn.recv())["observation"]
    import cv2

    for k in ("base_image", "left_wrist_image", "right_wrist_image"):
        cv2.imwrite(f"{out}_{i}_{k}.png", cv2.cvtColor(np.asarray(obs[k]), cv2.COLOR_RGB2BGR))
    print(f"sample {i}: state[:4]={np.asarray(obs['state'])[:4].round(3)} prompt={obs.get('prompt')!r}")
    time.sleep(1.0)
conn.close()
print("saved to", out)
