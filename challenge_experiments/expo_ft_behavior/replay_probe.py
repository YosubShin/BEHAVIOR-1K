"""Open-loop demo-action replay probe.

The decisive execution-layer test: reset the env to demo i's extracted start
state, then feed demo i's RECORDED actions open-loop. If the sim reaches the
toggle (success), then state-restore + controllers + sim are all validated and
any policy failure from the same starts is attributable to the policy/obs.
If replay fails, the execution layer itself (restore drift, controller
integration) is broken and policy work is premature.

Run in the behavior env against a server started with matching trims/starts:
    python replay_probe.py --starts /mnt/nvme/expoft_snapshots/demo_starts_navend \
        --trims /mnt/nvme/expoft_demos/turning_on_radio_navend --n 5
NOTE: pause/stop the learner first — the probe shares the singleton env.
"""

import argparse
import glob
import os
import time

import h5py
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
    p.add_argument("--starts", required=True)
    p.add_argument("--trims", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--extra-steps", type=int, default=60, help="grace steps beyond demo segment")
    args = p.parse_args()

    packer = Packer()
    conn = websockets.sync.client.connect(
        f"ws://{args.host}:{args.port}", compression=None, max_size=None, ping_interval=None
    )

    def call(op, **kw):
        conn.send(packer.pack({"operation": op, **kw}))
        r = unpackb(conn.recv())
        assert r.get("status") != "error", f"{op}: {r.get('message')}"
        return r

    env_id = call("create_env")["env_id"]
    starts = sorted(glob.glob(f"{args.starts}/*.pt"))
    results = []
    for k in range(args.n):
        snap = starts[k]
        ep_idx = int(os.path.basename(snap).split("_")[0][2:])
        # trims are renumbered 0..n in extraction order == sorted ep order
        with h5py.File(f"{args.trims}/{k}/traj.hdf5") as f:
            actions = np.asarray(f["action"]["joint_full"])
        call("reset", env_id=env_id, snapshot_path=snap)
        outcome, steps = "timeout", 0
        for t in range(len(actions) + args.extra_steps):
            a = actions[min(t, len(actions) - 1)]
            call("step", env_id=env_id, action=a.tolist())
            info = call("get_info_for_step", env_id=env_id)
            steps = t + 1
            if info["done"]:
                outcome = "SUCCESS" if info["success"] else "done-fail"
                break
        results.append(outcome == "SUCCESS")
        print(f"[probe] demo ep{ep_idx}: {outcome} at step {steps}/{len(actions)}")
    print(f"[probe] REPLAY RESULT: {sum(results)}/{len(results)} successes")
    conn.close()


if __name__ == "__main__":
    main()
