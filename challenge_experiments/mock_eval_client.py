"""Mock evaluator client -- verifies minimal_policy_server.py without OmniGibson.

Reproduces what omnigibson.eval.utils.network_utils.WebsocketClientPolicy does:
  1. GET /healthz until 200.
  2. Open websocket, receive metadata frame.
  3. Send a few synthetic obs dicts shaped like the real R1Pro challenge obs,
     receive an action each time, assert shape == (action_dim,).
  4. Send a {"reset": True} frame (no response expected).

Exit code 0 on success. Run inside an env that has websockets + msgpack + torch
(e.g. `conda run -n env_isaaclab python mock_eval_client.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.client

import msgpack
import numpy as np
import torch as th

# Reuse the exact codec from the server module.
from minimal_policy_server import packb, unpackb


def make_fake_obs() -> dict:
    """Synthetic obs matching the flattened R1Pro challenge schema (RGBD full-res)."""
    return {
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros((720, 720, 4), dtype=np.uint8),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros((480, 480, 4), dtype=np.uint8),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros((480, 480, 4), dtype=np.uint8),
        "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": np.zeros((720, 720), dtype=np.float32),
        "robot_r1::cam_rel_poses": np.zeros(7 * 3, dtype=np.float32),
        "task_id": np.array([0], dtype=np.int64),
    }


def wait_for_healthz(host: str, port: int, timeout_s: float = 20.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/healthz")
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            if resp.status == 200 and body.strip() == b"OK":
                print(f"[client] healthz OK ({resp.status})")
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("healthz never returned 200")


async def run(host: str, port: int, steps: int, action_dim: int) -> None:
    import websockets

    wait_for_healthz(host, port)
    uri = f"ws://{host}:{port}"
    async with websockets.connect(uri, compression=None, max_size=None) as ws:
        metadata = unpackb(await ws.recv())
        print(f"[client] connected; server metadata = {metadata}")

        for step in range(steps):
            await ws.send(packb(make_fake_obs()))
            reply = unpackb(await ws.recv())
            assert "action" in reply, f"reply missing 'action': {list(reply)}"
            action = th.from_numpy(np.asarray(reply["action"])).to(th.float32)
            assert action.shape == (action_dim,), f"bad action shape {tuple(action.shape)} != ({action_dim},)"
            if step == 0:
                print(
                    f"[client] step 0 action shape={tuple(action.shape)} "
                    f"dtype={action.dtype} timing={reply.get('server_timing')}"
                )

        # reset frame (no response), then confirm the socket still serves actions
        await ws.send(packb({"reset": True}))
        await ws.send(packb(make_fake_obs()))
        reply = unpackb(await ws.recv())
        assert reply["action"].shape == (action_dim,)
        print(f"[client] ran {steps} steps + reset roundtrip OK; action_dim={action_dim}")
    print("[client] PASS")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--action-dim", type=int, default=23)
    args = p.parse_args()
    asyncio.run(run(args.host, args.port, args.steps, args.action_dim))


if __name__ == "__main__":
    main()
