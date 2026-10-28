"""
Ceiling of the indoor endurance test under the MPC's own action menu
====================================================================
A privileged policy that reads the house's distance field directly (no vision, no world model)
and picks, every DECIDE_EVERY steps, the candidate whose straight-line future over 32 steps keeps
the largest clearance — using exactly the MPC's menu: forward, or a 0.5 s lateral veer (yaw is
fixed, as in WMPolicy). `--stop` adds "hover when every moving candidate ends within 0.3 m",
which the real MPC never chooses. If even this oracle survives rarely, the indoor number is a
property of the action space and the metric, not of the learned world model.

  python src/hd_oracle.py --houses 8 --house-first 16 --seeds 300
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HYPERLUX_BACKEND", "WEBGPU")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import ACTION_NAMES, ACTION_VECS, CTRL_HZ, FORWARD, H_MAX  # noqa: E402
from gen_wm_dataset_hyperdrone import CHUNK, IMG_RES, Flight, _jit_fast  # noqa: E402
from hd_closed_loop import RoomEpisode, fly_batch_rooms  # noqa: E402
from wm_closed_loop import DECIDE_EVERY, EVADE_HOLD  # noqa: E402
from wm_rooms import PILLAR_R  # noqa: E402


class OraclePolicy:
    def __init__(self, allow_stop: bool):
        self.allow_stop, self.hold, self.evade = allow_stop, 0, FORWARD
        self.ep = None

    def begin(self, pillars):
        self.hold, self.evade = 0, FORWARD

    def decide(self, frame, state):
        if self.hold > 0:
            self.hold -= DECIDE_EVERY
            return self.evade
        ep = self.ep
        rot = np.array([[math.cos(ep.yaw), -math.sin(ep.yaw)], [math.sin(ep.yaw), math.cos(ep.yaw)]])
        world = ep.start_xy + rot @ np.asarray(state[0:2])
        taus = np.arange(H_MAX + 1) / CTRL_HZ
        scores = {}
        for name in ("forward", "veer_left", "veer_right"):
            v = ep.speed * ACTION_VECS[ACTION_NAMES.index(name)][:2]
            path = world[None, :] + taus[:, None] * (rot @ v)[None, :]
            scores[name] = float(ep.field.clearance(path).min()) + PILLAR_R
        if scores["forward"] >= 0.7:  # nothing ahead within the warn ring: fly on
            self.evade = FORWARD
            return FORWARD
        best = max(("veer_left", "veer_right"), key=lambda n: scores[n])
        if self.allow_stop and max(scores.values()) < 0.3:
            self.evade, self.hold = ACTION_NAMES.index("hover"), EVADE_HOLD
            return self.evade
        self.evade, self.hold = ACTION_NAMES.index(best), EVADE_HOLD
        return self.evade


def main():
    import hyperlux
    from hyperlux.camera import RGB, CameraSpec
    from wm_rooms import DistanceField, house_references, load_house, probe_renderer

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--houses", type=int, default=8)
    ap.add_argument("--house-first", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()
    flight = Flight(CHUNK, "crazyflie", 0, False, "body")
    cams = (CameraSpec("front", (RGB("rgba8"),)),)
    make_renderer = lambda: _jit_fast(lambda: hyperlux.HyperLux(width=IMG_RES, height=IMG_RES, batch_size=CHUNK, cameras=cams, fidelity="medium"))
    refs = house_references(args.houses, first=args.house_first)
    rows = []
    t0 = time.perf_counter()
    for h, (house_id, ref) in enumerate(refs):
        house = load_house(ref, fidelity="medium")
        rig = _jit_fast(probe_renderer)
        rig.init(house)
        field = DistanceField(house, rig, rig.scene_bounds, verbose=False)
        del rig
        episodes = []
        for i in range(h, args.seeds, len(refs)):
            seed = args.seed0 + i
            a = RoomEpisode(seed, OraclePolicy(False), 1.0, field, house_id)
            b = RoomEpisode(seed, OraclePolicy(True), 1.0, field, house_id)
            b.start_xy, b.yaw = a.start_xy, a.yaw
            a.policy.ep, b.policy.ep = a, b
            rows.append((a, b))
            episodes += [a, b]
        for c0 in range(0, len(episodes), CHUNK):
            renderer = make_renderer()
            renderer.init(house)
            fly_batch_rooms(episodes[c0 : c0 + CHUNK], flight, renderer)
            del renderer
        print(f"[INFO] house {house_id}: {len(episodes) // 2} starts")
    for label, k in (("oracle, MPC menu (forward / lateral veers, fixed yaw)", 0), ("oracle + hover when boxed in", 1)):
        eps = [r[k] for r in rows]
        surv = np.mean([not e.crashed for e in eps])
        print(f"ORACLE OK [{label}]: starts={len(eps)} survival {surv:.0%}, mean flight time {np.mean([e.survived_s for e in eps]):.1f} s, "
              f"mean distance {np.mean([e.distance for e in eps]):.2f} m, evaded at least once {np.mean([e.trigger >= 0 for e in eps]):.0%}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
