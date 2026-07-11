"""Obs-provenance A/B probe: drive OUR env server with the REFERENCE serve policy.

Bisection logic (expo pristine = 0/49; serve-eval = 2/10 on 301-310):
  - serve-policy-through-our-env ~= serve-eval  -> our env/obs side is clean;
    the residual bug is inside expo's inference internals.
  - serve-policy-through-our-env ~= 0           -> our env server's obs (or env
    stepping) differ from the challenge eval harness; hunt there.

Run in the `behavior` conda env (uses omnigibson's own eval-client classes for
byte-level fidelity with yesterday's serve eval):
    python probe_serve_via_envserver.py --env-port 8102 --policy-port 8010 --episodes 20
Requires: behavior_env_server.py on --env-port (with get_eval_obs op),
          serve_b1k.py on --policy-port (receding_horizon, action_horizon 16).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch as th
import websockets.sync.client

from omnigibson.eval.utils.network_utils import Packer, unpackb
from omnigibson.eval.utils.network_utils import WebsocketClientPolicy


class EnvOpsClient:
    """Minimal client for behavior_env_server's 5-op msgpack protocol."""

    def __init__(self, host: str, port: int):
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}", max_size=None, open_timeout=600
        )
        self._packer = Packer()

    def call(self, operation: str, **kwargs) -> dict:
        self._ws.send(self._packer.pack({"operation": operation, **kwargs}))
        return unpackb(self._ws.recv())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-host", default="localhost")
    p.add_argument("--env-port", type=int, default=8102)
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=8010)
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    env = EnvOpsClient(args.env_host, args.env_port)
    created = env.call("create_env")
    env_id = created["env_id"]
    print(f"[probe] env {env_id}: {created.get('task_description')}", flush=True)
    policy = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)

    results = []
    for ep in range(args.episodes):
        env.call("reset", env_id=env_id)
        policy.reset()
        t0, steps, success = time.time(), 0, False
        while True:
            eval_obs = env.call("get_eval_obs", env_id=env_id)["eval_obs"]
            obs = {
                k: th.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in eval_obs.items()
            }
            action = policy.act(obs)
            action = action.detach().cpu().numpy() if isinstance(action, th.Tensor) else np.asarray(action)
            env.call("step", env_id=env_id, action=action.astype(np.float32).tolist())
            info = env.call("get_info_for_step", env_id=env_id)
            steps += 1
            if info["done"]:
                success = bool(info["success"])
                break
        results.append(success)
        print(
            f"[probe] ep {ep + 1}/{args.episodes}: {'SUCCESS' if success else 'fail'} "
            f"({steps} steps, {time.time() - t0:.0f}s) | running {sum(results)}/{len(results)}",
            flush=True,
        )

    print(f"[probe] FINAL: {sum(results)}/{len(results)} successes", flush=True)


if __name__ == "__main__":
    main()
