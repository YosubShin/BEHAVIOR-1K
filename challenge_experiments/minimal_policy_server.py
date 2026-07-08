"""Minimal BEHAVIOR-1K challenge policy server.

This is a *reference* websocket policy server that speaks the exact protocol the
OmniGibson evaluator (`python -m omnigibson.eval.eval`) expects. Drop your own
model into `MyPolicy.act` and you have a valid challenge-track submission server.

Protocol (mirrors omnigibson/eval/utils/network_utils.py):
  1. Client connects over websocket. Server immediately sends a msgpack-encoded
     `metadata` dict (may be empty).
  2. Per environment step the client sends a msgpack-encoded `obs` dict.
       - If the dict contains key "reset" -> call policy.reset(), send nothing back.
       - Otherwise -> call policy.act(obs), send back {"action": <np.ndarray>, ...}.
  3. The `action` must be a 1-D array of length `robot.action_dim`
     (23 for the default R1Pro config in omnigibson/eval/r1pro.yaml).
  4. A GET /healthz must return 200 OK -- the client waits on this before connecting.

msgpack payloads use the numpy/torch codec vendored below (identical semantics to
network_utils.pack_data / unpack_data): arrays travel as
{b"__ndarray__": True, b"data": bytes, b"dtype": str, b"shape": tuple}.

--------------------------------------------------------------------------------
IN THE REAL `behavior` CONDA ENV, PREFER THE PROVIDED SERVER:

    from omnigibson.eval.utils.network_utils import WebsocketPolicyServer
    WebsocketPolicyServer(policy=MyPolicy(), host="0.0.0.0", port=8000).serve_forever()

  where `policy` only needs `.act(obs_dict) -> torch.Tensor` and `.reset()`.

The standalone server in this file exists ONLY so the protocol can be smoke-tested
on a machine that doesn't have OmniGibson installed (e.g. via mock_eval_client.py).
It targets the classic `websockets` API (tested with websockets==12.0).
--------------------------------------------------------------------------------

Run:
    python minimal_policy_server.py --host 127.0.0.1 --port 8000 --action-dim 23
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
import time
import traceback

import msgpack
import numpy as np
import torch as th

logging.basicConfig(level=logging.INFO, format="[server] %(message)s")
logger = logging.getLogger("minimal_policy_server")


# ======================================================================================
# The policy. THIS is the only part you replace for a real submission.
# ======================================================================================
class MyPolicy:
    """Reference policy. Replace `act` with your model.

    The default implementation emits zero actions (a no-op), matching
    omnigibson.eval.policies.LocalPolicy -- useful as an eval smoke test.

    Observation dict keys the evaluator sends (flattened with "::"), for the
    default R1Pro robot config with cameras head / left_wrist / right_wrist:

      "robot_r1::proprio"                                  float32 (61,)
      "robot_r1::robot_r1:zed_link:Camera:0::rgb"          uint8  (720,720,4)  [head]
      "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"  uint8 (480,480,4)
      "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb" uint8 (480,480,4)
      "robot_r1::...::depth_linear"                        float32(H,W)  [only w/ RGBDFullResWrapper]
      "robot_r1::cam_rel_poses"                            float32 (7*num_cams,)
      "task_id"                                            int64  (1,)

    Proprio layout (PROPRIOCEPTION_INDICES["R1Pro"], total 61):
      base_qvel[0:3] arm_left_qpos[3:10] arm_left_qvel[10:17] eef_left_pos[17:20]
      eef_left_quat[20:24] gripper_left_qpos[24:26] gripper_left_qvel[26:28]
      arm_right_qpos[28:35] arm_right_qvel[35:42] eef_right_pos[42:45]
      eef_right_quat[45:49] gripper_right_qpos[49:51] gripper_right_qvel[51:53]
      trunk_qpos[53:57] trunk_qvel[57:61]

    Action layout (default r1pro.yaml controllers, total action_dim = 23):
      base[0:3] (holonomic vel) torso[3:7] (trunk pos) left_arm[7:14] (pos)
      left_gripper[14:15] right_arm[15:22] (pos) right_gripper[22:23]

    IMPORTANT (challenge-track rule): only consume rgb / depth / proprio here.
    Do NOT read segmentation, object state, target poses, or global robot pose.
    """

    def __init__(self, action_dim: int = 23, action_horizon: int = 1) -> None:
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        # Buffer for action chunking: if your model predicts a chunk of future
        # actions, cache them here and serve one per call (receding horizon).
        self._chunk: list[th.Tensor] = []

    def reset(self) -> None:
        """Called at the start of every rollout. Clear any temporal state here."""
        self._chunk.clear()
        logger.info("policy.reset()")

    @th.no_grad()
    def act(self, obs: dict) -> th.Tensor:
        """Return a single action of shape (action_dim,) as a float32 torch tensor."""
        if not self._chunk:
            self._chunk = list(self._predict_chunk(obs))
        return self._chunk.pop(0)

    def _predict_chunk(self, obs: dict) -> th.Tensor:
        """Run the model once, produce `action_horizon` actions.

        Replace this body with a real forward pass. Example of reading inputs:

            proprio = obs["robot_r1::proprio"]                # np.ndarray (61,)
            head_rgb = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
            actions = self.model.infer(head_rgb, proprio)     # (H, action_dim)
            return th.as_tensor(actions, dtype=th.float32)
        """
        return th.zeros((self.action_horizon, self.action_dim), dtype=th.float32)


# ======================================================================================
# msgpack numpy/torch codec -- identical semantics to network_utils.py
# ======================================================================================
def pack_data(obj):
    if isinstance(obj, th.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data, strict_map_key=False)


# ======================================================================================
# Standalone websocket server (classic websockets API, e.g. v12).
# In the `behavior` env use omnigibson's WebsocketPolicyServer instead.
# ======================================================================================
async def _handler(websocket, policy: MyPolicy, metadata: dict):
    logger.info(f"connection opened from {websocket.remote_address}")
    await websocket.send(packb(metadata))
    try:
        async for message in websocket:
            result = unpackb(message)
            if isinstance(result, dict) and "reset" in result:
                policy.reset()
                continue
            t0 = time.monotonic()
            action = policy.act(result)
            infer_ms = (time.monotonic() - t0) * 1000.0
            payload = {"action": action.detach().cpu().numpy(), "server_timing": {"infer_ms": infer_ms}}
            await websocket.send(packb(payload))
    except Exception:
        logger.error(f"handler error:\n{traceback.format_exc()}")
        raise
    finally:
        logger.info("connection closed")


async def _process_request(path, request_headers):
    """Classic websockets (v12) health-check hook: return a response to short-circuit."""
    if path == "/healthz":
        return http.HTTPStatus.OK, [("Content-Type", "text/plain")], b"OK\n"
    return None


async def _serve(policy: MyPolicy, host: str, port: int, metadata: dict) -> None:
    import websockets

    handler = functools.partial(_handler, policy=policy, metadata=metadata)
    logger.info(f"starting websocket server on {host}:{port} (action_dim={policy.action_dim})")
    async with websockets.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=_process_request,
        ping_interval=60,
        ping_timeout=300,
    ):
        await asyncio.Future()  # run forever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-dim", type=int, default=23, help="R1Pro default config = 23.")
    parser.add_argument("--action-horizon", type=int, default=1)
    args = parser.parse_args()

    policy = MyPolicy(action_dim=args.action_dim, action_horizon=args.action_horizon)
    try:
        asyncio.run(_serve(policy, args.host, args.port, metadata={"policy": "minimal-zero"}))
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
