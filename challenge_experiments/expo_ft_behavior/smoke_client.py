"""Smoke test for behavior_env_server.py — exercises the 5-op protocol like
EXPO-FT's train loop does (get_observation -> get_info_for_step -> step).

Run in the `behavior` conda env (has websockets + the msgpack codec):
    python smoke_client.py --port 8102 --steps 30
"""

import argparse
import time

import numpy as np
import websockets.sync.client

try:
    from openpi_client import msgpack_numpy

    Packer, unpackb = msgpack_numpy.Packer, msgpack_numpy.unpackb
except ImportError:
    from omnigibson.eval.utils.network_utils import Packer, unpackb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--task", default="turning_on_radio")
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()

    packer = Packer()
    print("[smoke] connecting (server may take minutes to come up)...")
    conn = None
    for _ in range(240):
        try:
            conn = websockets.sync.client.connect(
                f"ws://{args.host}:{args.port}", compression=None, max_size=None, ping_interval=None
            )
            break
        except Exception:
            time.sleep(5)
    assert conn is not None, "could not connect to env server"
    print("[smoke] connected")

    def call(op, **kw):
        conn.send(packer.pack({"operation": op, **kw}))
        resp = unpackb(conn.recv())
        assert resp.get("status") != "error", f"{op} error: {resp.get('message')}"
        return resp

    t0 = time.time()
    r = call("create_env", task_name=args.task, instance_ids=[0, 1], max_steps=200, dense_reward=True,
             perturb_pose=True, seed=0)
    env_id = r["env_id"]
    print(f"[smoke] create_env OK in {time.time()-t0:.0f}s | env_id={env_id} | prompt={r['task_description']!r}")

    t0 = time.time()
    r = call("reset", env_id=env_id)
    obs = r["observation"]
    print(f"[smoke] reset OK in {time.time()-t0:.0f}s | done={r['done']}")
    for k, v in obs.items():
        print(f"    obs[{k!r}]: {type(v).__name__}", getattr(v, "shape", ""), getattr(v, "dtype", ""))
    assert obs["state"].shape == (23,), f"state shape {obs['state'].shape} != (23,)"
    assert obs["base_image"].ndim == 3 and obs["base_image"].shape[-1] == 3

    rewards, t0 = [], time.time()
    for i in range(args.steps):
        o = call("get_observation", env_id=env_id)["observation"]
        info = call("get_info_for_step", env_id=env_id)
        a = np.zeros(23, dtype=np.float32)
        call("step", env_id=env_id, action=a.tolist())
        rewards.append(info["reward"])
        if i == 0:
            print(f"[smoke] step-loop wired: done={info['done']} success={info['success']} "
                  f"reward={info['reward']:.4f} mask={info['mask']}")
        if info["done"]:
            print(f"[smoke] episode done at step {i} (expected only if max_steps hit)")
            break
    dt = time.time() - t0
    print(f"[smoke] {len(rewards)} steps in {dt:.1f}s = {len(rewards)/dt:.1f} env-steps/s (3 ops per step)")
    print(f"[smoke] reward stats: min={min(rewards):.4f} max={max(rewards):.4f} sum={sum(rewards):.4f} "
          f"(zero-action policy -> expect ~0.0)")

    # second reset — exercises instance cycling + pose perturbation
    t0 = time.time()
    call("reset", env_id=env_id)
    print(f"[smoke] second reset OK in {time.time()-t0:.0f}s (instance cycled, pose perturbed)")
    conn.close()
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
