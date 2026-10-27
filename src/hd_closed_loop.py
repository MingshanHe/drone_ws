"""
Closed-loop evaluation inside hyperdrone — the mirror of the pybullet scoreboards
================================================================================
`eval_world_model_policy.py` and `speed_sweep.py` fly a world-model checkpoint through
pybullet pillar courses. This script flies the SAME courses (same seed -> same pillar layout,
because `sample_pillars` makes the pybullet generator's random draws), with the SAME policy
objects (`ReactivePolicy`, `WMPolicy` from wm_closed_loop.py, deciding every DECIDE_EVERY
steps from the 64x64 frame), the same 360-step budget, 3 m finish line and 0.22 m crash
criterion — but the drone is hyperdrone's crazyflie under RAPTOR and the pictures come from
the hyperlux raytracer (`--arena-style hyperdrone` or `pybullet`). So a pybullet-trained
checkpoint can be tested in hyperdrone, and hyperdrone-trained ones in their own domain.

"Collision" is the same geometric test as in pybullet: planar distance to a pillar centre
below COLLISION_R counts as a crash (pillars have no contact physics in either simulator).
hyperlux's collision probes exist, but the shared criterion keeps the numbers comparable.

Courses are flown 32 at a time (one arena each, in one static scene), so a 100-course
scoreboard takes about a minute on this machine's CPU renderer.

`--scene rooms` is the indoor test, inside furnished ProcTHOR houses that no training set used
(catalog entries 16-19 by default). Pillar-style courses do not exist there (a free-standing
obstacle with a 3 m straight run behind it and room to pass on both sides is essentially never
found: furniture stands against walls, rooms are small), so the indoor test is a **clutter
endurance** run instead: 100 starts with >= 0.9 m clearance and >= 2 m free ahead, a random
heading, the policy flying "forward unless it decides to evade" exactly as on the courses, until
the first contact (the field's surface clearance + 0.18 m below COLLISION_R, the same
centre-equivalent test) or 360 steps (7.5 s). Reported per policy: survival rate (no contact),
time to first contact, distance flown before it, and mean clearance. The reactive baseline's
privileged evasion direction uses the nearest occupied cell.

Run (conda env `dreamer`, from drone_ws/):
  python src/hd_closed_loop.py --model src/output/world_model_pybullet_ref.pth --arena-style pybullet
  python src/hd_closed_loop.py --model ... --arena-style hyperdrone --seeds 100 --sweep-seeds 30
Prints the same `WORLD-POLICY OK` / `SPEED-SWEEP OK` blocks as the pybullet scripts.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HYPERLUX_BACKEND", os.environ.get("HYPERLUX_BACKEND", "WEBGPU"))

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import ACTION_VECS, COLLISION_R, CTRL_HZ, FORWARD, START  # noqa: E402
from gen_wm_dataset_hyperdrone import (  # noqa: E402
    ARENA_STYLES,
    CHUNK,
    DEFAULT_FOV,
    IMG_RES,
    Flight,
    _jit_fast,
    _rot,
    build_scene,
    room_offsets,
    sample_pillars,
)
from speed_sweep import BASE_V, SPEEDS  # noqa: E402
from train_world_model import MODEL, load_model  # noqa: E402
from wm_closed_loop import DECIDE_EVERY, GOAL_X, TMAX, ReactivePolicy, WMPolicy  # noqa: E402


class Episode:
    """One course: its policy object, pillar layout and running metrics."""

    def __init__(self, seed: int, policy, in_path: bool, speed: float, solo: bool):
        self.seed, self.policy, self.speed = seed, policy, float(speed)
        self.pillars = sample_pillars(np.random.default_rng(seed), in_path=in_path, solo=solo)
        self.centres = np.array([(p["x"], p["y"]) for p in self.pillars], dtype=np.float32)
        self.policy.begin([tuple(c) for c in self.centres])
        self.min_clear, self.trigger, self.steps, self.reached, self.active = 9.0, -1, TMAX, False, True
        self.a_id = FORWARD


def fly_batch(episodes: list, flight: Flight, renderer, style: str, fov_deg: float = DEFAULT_FOV, randomize: bool = False) -> None:
    """Fly up to CHUNK episodes together, one arena each, exactly as run_episode does one.
    `randomize` draws the dataset generator's appearance randomization (pillar radius/height/
    colour, sun and fill lights, wall and floor tints) per batch, seeded by the first course."""
    from hyperlux.camera import RGB

    n = len(episodes)
    offsets = room_offsets(CHUNK, style)
    rng = np.random.default_rng(10_000 + episodes[0].seed)
    if randomize:
        for ep in episodes:
            for q in ep.pillars:  # same draws as sample_pillars(randomize=True), geometry unchanged
                q["r"], q["h"] = float(rng.uniform(0.14, 0.22)), float(rng.uniform(1.0, 1.8))
                q["color"] = tuple(np.clip(np.array((0.80, 0.32, 0.22)) + rng.uniform(-0.3, 0.3, 3), 0.05, 1.0))
    scene = build_scene([ep.pillars for ep in episodes], offsets[:n], rng, randomize, style)
    renderer.init(scene)
    off3 = np.concatenate([offsets, np.zeros((CHUNK, 1), np.float32)], 1)
    starts = START.astype(np.float32) + off3
    starts[n:, 0] += 1e4  # spare drones parked far away
    flight.reset(starts)
    exec_ids = np.full(CHUNK, -3, dtype=np.int64)
    v = np.zeros((CHUNK, 3), dtype=np.float32)
    state = np.zeros(20)
    for t in range(TMAX):
        local = flight.position()[:n] - off3[:n]
        if t % DECIDE_EVERY == 0:
            renderer.set_cameras(flight.camera_bases(fov_deg))
            renderer.render()
            frames = renderer.outputs[RGB("rgba8")].read()[:n, :, :, :3]
            for i, ep in enumerate(episodes):
                if not ep.active:
                    continue
                state[0:3] = local[i]
                ep.a_id = ep.policy.decide(np.ascontiguousarray(frames[i]), state)
                if ep.a_id != FORWARD and ep.trigger < 0:
                    ep.trigger = t
        for i, ep in enumerate(episodes):
            if ep.active:
                v[i] = (ep.speed * ACTION_VECS[ep.a_id])[:3]
                exec_ids[i] = ep.a_id
            else:
                v[i], exec_ids[i] = 0.0, -2  # finished: hover in place
        flight.step(v, exec_ids)
        local = flight.position()[:n] - off3[:n]
        for i, ep in enumerate(episodes):
            if not ep.active:
                continue
            d = float(np.linalg.norm(ep.centres - local[i, :2], axis=1).min()) if len(ep.centres) else 9.0
            ep.min_clear = min(ep.min_clear, d)
            if local[i, 0] >= GOAL_X:
                ep.reached, ep.steps, ep.active = True, t + 1, False
        if not any(ep.active for ep in episodes):
            break
    for ep in episodes:
        ep.crashed = ep.min_clear < COLLISION_R


def run_all(episodes: list, style: str, flight: Flight, make_renderer, randomize: bool = False) -> None:
    for c0 in range(0, len(episodes), CHUNK):
        renderer = make_renderer()
        fly_batch(episodes[c0 : c0 + CHUNK], flight, renderer, style, randomize=randomize)
        del renderer


class RoomEpisode:
    """One clutter-endurance run inside a house: start and heading from the distance field."""

    def __init__(self, seed: int, policy, speed: float, field, house_id: str):
        from wm_rooms import PILLAR_R

        rng = np.random.default_rng(seed)
        found = field.sample_open_start(rng)
        if found is None:
            raise RuntimeError(f"house {house_id}: no flyable start for seed {seed}")
        self.seed, self.policy, self.speed, self.field, self.house_id = seed, policy, float(speed), field, house_id
        self.start_xy, self.yaw = np.asarray(found[0], dtype=np.float32), float(found[1])
        self.offset_r = PILLAR_R
        self.policy.begin([])
        self.min_clear, self.trigger, self.steps, self.active = 9.0, -1, TMAX, True
        self.contact_t, self.distance, self.clear_sum, self.a_id = -1, 0.0, 0.0, FORWARD
        self._last = None


def fly_batch_rooms(episodes: list, flight: Flight, renderer, fov_deg: float = DEFAULT_FOV) -> None:
    """Fly up to CHUNK room episodes (all in the renderer's house) together."""
    from hyperlux.camera import RGB

    n = len(episodes)
    starts = np.zeros((CHUNK, 3), dtype=np.float32)
    yaws = np.zeros(CHUNK, dtype=np.float32)
    for i, ep in enumerate(episodes):
        starts[i, :2], starts[i, 2], yaws[i] = ep.start_xy, START[2], ep.yaw
    starts[n:, 0] = 1e4 + np.arange(CHUNK - n) * 10.0
    flight.reset(starts, yaws)
    rot = _rot(yaws)  # local -> world
    exec_ids = np.full(CHUNK, -3, dtype=np.int64)
    v = np.zeros((CHUNK, 3), dtype=np.float32)
    state = np.zeros(20)

    def local_of(world):
        rel = world[:n] - starts[:n]
        return np.einsum("nji,nj->ni", rot[:n], rel[:, :2]), rel[:, 2] + START[2]

    for t in range(TMAX):
        world = flight.position()
        local_xy, local_z = local_of(world)
        if t % DECIDE_EVERY == 0:
            renderer.set_cameras(flight.camera_bases(fov_deg))
            renderer.render()
            frames = renderer.outputs[RGB("rgba8")].read()[:n, :, :, :3]
            for i, ep in enumerate(episodes):
                if not ep.active:
                    continue
                state[0:2], state[2] = local_xy[i], local_z[i]
                if isinstance(ep.policy, ReactivePolicy):  # privileged direction: nearest occupied cell
                    threat = ep.field.threat_xy(world[i, :2])
                    ep.policy.pillars = [rot[i].T @ (threat - ep.start_xy)]
                ep.a_id = ep.policy.decide(np.ascontiguousarray(frames[i]), state)
                if ep.a_id != FORWARD and ep.trigger < 0:
                    ep.trigger = t
        for i, ep in enumerate(episodes):
            if ep.active:
                vl = ep.speed * ACTION_VECS[ep.a_id]
                v[i, :2], v[i, 2] = rot[i] @ vl[:2], vl[2]
                exec_ids[i] = ep.a_id
            else:
                v[i], exec_ids[i] = 0.0, -2
        flight.step(v, exec_ids)
        world = flight.position()
        for i, ep in enumerate(episodes):
            if not ep.active:
                continue
            d = float(ep.field.clearance(world[i, :2])) + ep.offset_r
            ep.min_clear = min(ep.min_clear, d)
            ep.clear_sum += d
            if ep._last is not None:
                ep.distance += float(np.linalg.norm(world[i, :2] - ep._last))
            ep._last = world[i, :2].copy()
            if d < COLLISION_R:  # first contact ends the run
                ep.contact_t, ep.steps, ep.active = t + 1, t + 1, False
        if not any(ep.active for ep in episodes):
            break
    for ep in episodes:
        ep.crashed = ep.contact_t >= 0
        ep.survived_s = ep.steps / 48.0
        ep.mean_clear = ep.clear_sum / max(ep.steps, 1)


def room_endurance(n_seeds, seed0, houses, house_first, enc, pred, cheads, nhead, meta, flight, make_renderer, fidelity):
    """The clutter-endurance test inside `houses` catalog houses; returns one row per start."""
    from wm_rooms import DistanceField, house_references, load_house, probe_renderer

    refs = house_references(houses, first=house_first)
    rows = []
    for h, (house_id, ref) in enumerate(refs):
        t0 = time.perf_counter()
        house = load_house(ref, fidelity=fidelity)
        rig = _jit_fast(probe_renderer)
        rig.init(house)
        field = DistanceField(house, rig, rig.scene_bounds, verbose=False)
        del rig
        episodes = []
        for i in range(h, n_seeds, len(refs)):
            seed = seed0 + i
            r = RoomEpisode(seed, ReactivePolicy(enc, nhead), 1.0, field, house_id)
            w = RoomEpisode(seed, WMPolicy(enc, pred, cheads, meta), 1.0, field, house_id)
            w.start_xy, w.yaw = r.start_xy, r.yaw  # both policies fly the same start
            rows.append({"seed": seed, "reactive": r, "wm": w, "house": house_id})
            episodes += [r, w]
        for c0 in range(0, len(episodes), CHUNK):
            renderer = make_renderer()
            renderer.init(house)
            fly_batch_rooms(episodes[c0 : c0 + CHUNK], flight, renderer)
            del renderer
        print(f"[INFO] house {house_id}: {len(episodes) // 2} starts in {time.perf_counter() - t0:.0f}s")
    rows.sort(key=lambda x: x["seed"])
    return rows


def main() -> None:
    import hyperlux
    from hyperlux.camera import RGB, CameraSpec

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--arena-style", default="pybullet", choices=ARENA_STYLES)
    ap.add_argument("--seeds", type=int, default=100, help="cluttered-course scoreboard seeds (0 to skip)")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--sweep-seeds", type=int, default=30, help="single-pillar speed-sweep seeds per speed (0 to skip)")
    ap.add_argument("--sweep-seed0", type=int, default=3000)
    ap.add_argument("--fidelity", default="medium")
    ap.add_argument("--scene", default="pillars", choices=("pillars", "rooms"), help="pillar arenas or furnished ProcTHOR houses")
    ap.add_argument("--houses", type=int, default=4)
    ap.add_argument("--house-first", type=int, default=16, help="first catalog house; 16+ were never used for training")
    ap.add_argument("--randomize", action="store_true", help="pillar arenas with the generator's appearance randomization (unseen colours, sizes, lights)")
    args = ap.parse_args()
    torch.set_num_threads(2)

    enc, pred, cheads, nhead, meta = load_model(args.model, device="cpu")
    print(f"[INFO] checkpoint {args.model}: trained on {meta.get('source', 'pybullet')}, camera {meta.get('fov_deg', 60):g} deg; "
          f"hyperdrone arena style {args.arena_style}")
    flight = Flight(CHUNK, "crazyflie", 0, False, "body")
    cameras = (CameraSpec("front", (RGB("rgba8"),)),)

    def make_renderer():
        return _jit_fast(lambda: hyperlux.HyperLux(width=IMG_RES, height=IMG_RES, batch_size=CHUNK, cameras=cameras, fidelity=args.fidelity))

    where = (f"hyperdrone/{args.arena_style}" + (" randomized" if args.randomize else "")) if args.scene == "pillars" else f"hyperdrone/rooms {args.house_first}-{args.house_first + args.houses - 1}"
    if args.seeds and args.scene == "rooms":
        t0 = time.perf_counter()
        rows = room_endurance(args.seeds, args.seed0, args.houses, args.house_first, enc, pred, cheads, nhead, meta, flight, make_renderer, args.fidelity)
        for row in rows:
            r, w = row["reactive"], row["wm"]
            print(f"  seed {row['seed']} ({row['house']}): contact at {r.survived_s:4.1f}->{w.survived_s:4.1f} s "
                  f"({'none' if not r.crashed else 'hit'}/{'none' if not w.crashed else 'hit'}), distance {r.distance:.1f}->{w.distance:.1f} m, "
                  f"mean clear {r.mean_clear:.2f}->{w.mean_clear:.2f} m, trigger {r.trigger:>3}->{w.trigger:>3}")
        stat = lambda pol, key: float(np.mean([getattr(x[pol], key) for x in rows]))
        hit = lambda pol: [x for x in rows if x[pol].crashed]
        ttc = lambda pol: float(np.mean([x[pol].survived_s for x in hit(pol)])) if hit(pol) else float("nan")
        print(
            f"ROOMS OK: starts={args.seeds} [{where}, {time.perf_counter() - t0:.0f}s]\n"
            f"  survival (no contact in 7.5 s): reactive {1 - stat('reactive', 'crashed'):.0%} -> wm {1 - stat('wm', 'crashed'):.0%}\n"
            f"  time to first contact (contacted runs): reactive {ttc('reactive'):.1f} s -> wm {ttc('wm'):.1f} s\n"
            f"  mean flight time:    reactive {stat('reactive', 'survived_s'):.1f} s -> wm {stat('wm', 'survived_s'):.1f} s\n"
            f"  mean distance flown: reactive {stat('reactive', 'distance'):.2f} m -> wm {stat('wm', 'distance'):.2f} m\n"
            f"  mean clearance:      reactive {stat('reactive', 'mean_clear'):.2f} m -> wm {stat('wm', 'mean_clear'):.2f} m\n"
            f"  evaded at least once: reactive {float(np.mean([x['reactive'].trigger >= 0 for x in rows])):.0%} -> wm {float(np.mean([x['wm'].trigger >= 0 for x in rows])):.0%}"
        )
    if args.seeds and args.scene == "pillars":
        t0 = time.perf_counter()
        if True:
            rows = []
            episodes = []
            for i in range(args.seeds):
                seed, in_path = args.seed0 + i, (i % 10) < 7
                r = Episode(seed, ReactivePolicy(enc, nhead), in_path, 1.0, False)
                w = Episode(seed, WMPolicy(enc, pred, cheads, meta), in_path, 1.0, False)
                rows.append({"seed": seed, "in_path": in_path, "reactive": r, "wm": w})
                episodes += [r, w]
            run_all(episodes, args.arena_style, flight, make_renderer, args.randomize)
        for row in rows:
            r, w = row["reactive"], row["wm"]
            print(f"  seed {row['seed']} ({'in-path' if row['in_path'] else 'clear  '}): "
                  f"clear {r.min_clear:.2f}->{w.min_clear:.2f} m, trigger {r.trigger:>3}->{w.trigger:>3}, crash {int(r.crashed)}/{int(w.crashed)}")
        ip = [x for x in rows if x["in_path"]]
        cl = [x for x in rows if not x["in_path"]]
        rate = lambda rs, pol, key: float(np.mean([float(getattr(x[pol], key)) for x in rs])) if rs else float("nan")
        crash_r, crash_w = rate(ip, "reactive", "crashed"), rate(ip, "wm", "crashed")
        clr_r, clr_w = rate(ip, "reactive", "min_clear"), rate(ip, "wm", "min_clear")
        leads = [(x["reactive"].trigger - x["wm"].trigger) * 1000.0 / CTRL_HZ for x in ip if x["reactive"].trigger >= 0 and x["wm"].trigger >= 0]
        lead = float(np.mean(leads)) if leads else float("nan")
        fp_r = float(np.mean([x["reactive"].trigger >= 0 for x in cl])) if cl else 0.0
        fp_w = float(np.mean([x["wm"].trigger >= 0 for x in cl])) if cl else 0.0
        ok = [x for x in ip if x["reactive"].reached and x["wm"].reached and not x["reactive"].crashed and not x["wm"].crashed]
        goal_pct = 100.0 * (float(np.mean([x["wm"].steps for x in ok])) / float(np.mean([x["reactive"].steps for x in ok])) - 1.0) if ok else float("nan")
        print(
            f"WORLD-POLICY OK: seeds={args.seeds} ({len(ip)} in-path / {len(cl)} clear) [{where}, {time.perf_counter() - t0:.0f}s]\n"
            f"  crash_rate:        reactive {crash_r:.0%} -> wm {crash_w:.0%}\n"
            f"  mean_min_clearance: {clr_r:.2f} m -> {clr_w:.2f} m\n"
            f"  mean_trigger_lead: +{lead:.0f} ms (n={len(leads)} both triggered)\n"
            f"  false_positive:    reactive {fp_r:.0%} -> wm {fp_w:.0%} of clear runs\n"
            f"  goal_time:         wm {goal_pct:+.0f}% vs reactive (n={len(ok)} clean)"
        )

    if args.sweep_seeds and args.scene == "rooms":
        print("[INFO] no single-pillar sweep in rooms")
    if args.sweep_seeds and args.scene == "pillars":
        t0 = time.perf_counter()
        report = {}
        for s in SPEEDS:
            episodes = []
            for i in range(args.sweep_seeds):
                seed = args.sweep_seed0 + i
                episodes.append(Episode(seed, ReactivePolicy(enc, nhead), True, s, True))
                episodes.append(Episode(seed, WMPolicy(enc, pred, cheads, meta, speed=s), True, s, True))
            run_all(episodes, args.arena_style, flight, make_renderer, args.randomize)
            r = [ep for ep in episodes[0::2]]
            w = [ep for ep in episodes[1::2]]
            report[s] = (np.mean([ep.crashed for ep in r]), np.mean([ep.crashed for ep in w]), np.mean([ep.min_clear for ep in r]), np.mean([ep.min_clear for ep in w]))
            print(f"  v={s * BASE_V:.1f} m/s: crash reactive {report[s][0]:.0%} / wm {report[s][1]:.0%}  (min clear {report[s][2]:.2f} / {report[s][3]:.2f} m)")
        lo, hi = report[SPEEDS[0]], report[SPEEDS[-1]]
        print(
            f"SPEED-SWEEP OK: {args.sweep_seeds} single-pillar courses/speed [hyperdrone/{args.arena_style}, {time.perf_counter() - t0:.0f}s] — crash "
            f"(reactive/wm) at {SPEEDS[0] * BASE_V:.1f} m/s = {lo[0]:.0%}/{lo[1]:.0%}; at {SPEEDS[-1] * BASE_V:.1f} m/s = {hi[0]:.0%}/{hi[1]:.0%}"
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
