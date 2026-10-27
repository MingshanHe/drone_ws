"""
Lesson 29 (step 1, hyperdrone edition) — the same intervention dataset, flown in hyperdrone
=========================================================================================
Drop-in alternative to `gen_wm_dataset.py`: identical action vocabulary, held-segment
schedule, pillar layouts, labels and npz schema — so `train_world_model.py` (and every
`counterfactual_labels` / `window_valid` consumer) reads it unchanged — but the flight and
the pictures come from hyperdrone instead of gym-pybullet-drones:

  * dynamics  — `hyperdrone.dynamics.Sim`: L2F quadrotor dynamics (crazyflie preset) under
                the RAPTOR position controller, all rollouts of a chunk stepped as ONE batch.
                The velocity *intent* becomes a moving position reference exactly like
                `VelCommander` (re-anchored at every command switch), plus a 0.5 s velocity
                lead that cancels RAPTOR's measured steady-state lag (~0.5 s).
  * rendering — the hyperlux raytracer: 64x64 RGB from the drone's forward camera, plus
                optional metric depth.
  * scenes    — `--scene pillars`: one closed arena per rollout (checker floor, four walls,
                cylinder pillars) in a single static scene, the pybullet course.
                `--scene rooms`: furnished ProcTHOR houses from hyperlux's pinned catalog
                (downloaded through conta on first use), labels measured from the geometry
                (see `wm_rooms.py`). `--scene mixed`: both in one dataset.
  * camera    — square pinhole, FOV configurable, body-fixed (`--camera body`, as in
                pybullet: the view pitches and rolls with the airframe) or virtually
                stabilised (`--camera level`: yaw only, like a slow-flying Tello whose camera
                never tilts more than a degree or two). 60 deg reproduces the pybullet camera.
                `--fov-deg 40` matches what the ChArUco-calibrated Tello can actually deliver
                through SakanaDreamer's `virtual_maps(64, 40)`: its 960x720 lens (HFOV ~56 deg,
                VFOV ~43.5 deg) covers at most a ~42.5 deg square view, so the 60 deg training
                view can never be produced from real Tello frames.

Why bother, when gen_wm_dataset.py already works? (1) the camera can be matched to the real
Tello; (2) rollouts are batched, so 64 trials cost one; (3) raytraced shading and shadows are
a different (richer) appearance distribution than pybullet's flat render; (4) real furnished
rooms instead of pillars. Honest note: this is still a simulator — obstacles have no contact
physics (the drone flies through, as before) and no sim here is photorealistic; the
Tello-side validation remains the real test.

Extra npz keys, ignored by old consumers: fov_deg, fov_half_deg, camera ("body"/"level"),
quat (R, L, 4) body attitude wxyz, scene_type (0 pillars / 1 room), scene_id, yaw,
start_world, cf_labels / cf_visible (the counterfactual oracle, precomputed so room rollouts
need no pillar list), source, drone_model, sim_dt, kinematic, and `depth` (float16 metres,
with --save-depth). Positions `pos` are always in the rollout's own frame (x forward from
the start, as in the pybullet data); `yaw` and `start_world` map them into the scene.

Run (conda env `dreamer`, from drone_ws/):
  python src/gen_wm_dataset_hyperdrone.py --rollouts 64 --len 120               # 60 deg pillars
  python src/gen_wm_dataset_hyperdrone.py --rollouts 64 --len 120 --fov-deg 40 --camera level  # Tello
  python src/gen_wm_dataset_hyperdrone.py --scene mixed --rollouts 32 --rooms 32 --houses 2 \\
      --fov-deg 40 --camera level --out src/output/wm_dataset_mixed.npz
  python src/gen_wm_dataset_hyperdrone.py --selftest                            # asserts
Saves output/wm_dataset_hyperdrone.npz (+ a preview PNG next to it). Then:
  python src/train_world_model.py --data src/output/wm_dataset_hyperdrone.npz

The first run JIT-compiles a few native components (~2 min each on this machine); later
runs load them from ~/.cache/hyperlux. hyperlux re-runs a CMake staleness check on every
construction (~1 min per component here), so this script sets HYPERLUX_SKIP_BUILD for the
cached fast path and only falls back to a real build when an artifact is missing.
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # hyperdrone's native library asks for it

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import (  # noqa: E402  (constants + pure-numpy helpers only)
    ACTION_NAMES,
    ACTION_VECS,
    A_NORM,
    CTRL_HZ,
    DANGER_R,
    FORWARD,
    H_MAX,
    HORIZONS,
    IMG_RES,
    SPEED_RANGE,
    START,
    _schedule,
    as_pairs,
    counterfactual_labels,
)

OUT = os.path.join(HERE, "output", "wm_dataset_hyperdrone.npz")

# --- simulation -------------------------------------------------------------------------
CHUNK = 32  # rollouts flown + rendered per batch; fixed because it is part of the JIT keys
TICK = 1.0 / CTRL_HZ
SUBSTEPS = 2  # dynamics dt = TICK / SUBSTEPS (~96 Hz, close to RAPTOR's native 100 Hz)
LEAD_S = 0.5  # velocity feed-forward on the position reference (RAPTOR lag ~0.5 s, measured)
LEAD_RAMP = 12  # ticks over which the lead fades in after a switch (a step pitches the airframe hard)
ALT_KP, ALT_KI, ALT_IMAX = 1.0, 0.5, 0.25  # outer altitude loop: RAPTOR alone sags ~0.25 m at 1.6 m/s
CAMERA_Z = 0.04  # gym-pybullet-drones mounts the camera L = 0.0397 m above the body centre
DEFAULT_FOV = 60.0
FOV_MARGIN_DEG = 2.0  # gen_wm_dataset: FOV_HALF_DEG = 28 = 60/2 - 2 (label-visibility margin)
DEFAULT_MODEL = "crazyflie"
SCENES = ("pillars", "rooms", "mixed")

# --- arena: one closed room per rollout, so no rollout ever sees another's pillars -------
ROOM_X, ROOM_Y, ROOM_H = (-2.0, 10.0), (-5.0, 5.0), 3.0
ROOM_PITCH = (16.0, 14.0)  # grid spacing between rooms (m)
TILE = 0.5
PILLAR_R, PILLAR_H, PILLAR_RGB = 0.18, 1.4, (0.80, 0.32, 0.22)  # gen_wm_dataset defaults
FLOOR_RGB = ((0.55, 0.55, 0.55), (0.35, 0.35, 0.35))
WALL_RGB = ((0.85, 0.82, 0.75), (0.70, 0.75, 0.80), (0.80, 0.70, 0.70), (0.75, 0.75, 0.75))
AMBIENT, SKY = (0.35, 0.35, 0.35), (0.60, 0.70, 0.85)
# --arena-style pybullet: the look of gym-pybullet-drones' course (white background, its
# blue/white checker plane running to the horizon, no walls, saturated red pillars), so a
# model trained here transfers to the pybullet evaluation on content rather than on paint
ARENA_STYLES = ("hyperdrone", "pybullet")
PB_FLOOR_X, PB_FLOOR_Y = (-2.0, 40.0), (-7.0, 7.0)  # adjacent strips touch: one continuous plane
PB_FLOOR_RGB = ((0.93, 0.93, 0.95), (0.58, 0.66, 0.88))
PB_PILLAR_RGB = (0.85, 0.18, 0.10)
PB_AMBIENT, PB_SKY = (0.45, 0.45, 0.45), (1.0, 1.0, 1.0)


def sample_pillars(rng, in_path: bool, solo: bool = False, randomize: bool = False) -> list:
    """The pillar layout of one rollout — the SAME random draws, in the same order, as
    `gen_wm_dataset.spawn_pillars`, so a given seed gives both generators the same courses.
    Returns dicts with planar centre, radius, height and colour."""
    pillars = []
    n = 1 if solo else int(rng.integers(2, 4))
    for i in range(n):
        if in_path and i == 0:
            px, py = float(rng.uniform(1.3, 2.0)), float(rng.uniform(-0.2, 0.2))
        else:
            px, py = float(rng.uniform(0.9, 2.6)), float(rng.uniform(0.9, 1.5))
            py *= 1.0 if rng.random() < 0.5 else -1.0
        if randomize:
            radius = float(rng.uniform(0.14, 0.22))
            length = float(rng.uniform(1.0, 1.8))
            color = tuple(np.clip(np.array(PILLAR_RGB) + rng.uniform(-0.3, 0.3, 3), 0.05, 1.0))
        else:
            radius, length, color = PILLAR_R, PILLAR_H, PILLAR_RGB
        pillars.append(dict(x=px, y=py, r=radius, h=length, color=color))
    return pillars


# --- procedural geometry (hyperlux.Mesh = vertices, triangles, flat colour) ----------------
_CORE = {}


def _core(name):
    """hyperlux's bound classes, resolved once: every `hyperlux.<Class>` attribute access goes
    through the module's __getattr__ and re-probes the backend (~50 ms each — minutes over
    thousands of meshes)."""
    if name not in _CORE:
        import hyperlux

        _CORE[name] = getattr(hyperlux, name)
    return _CORE[name]


def _quads(obj, quads, color):
    """One two-sided mesh from a list of planar quads (each four corner points)."""
    quads = np.asarray(quads, dtype=np.float32).reshape(-1, 4, 3)
    verts = quads.reshape(-1, 3)
    base = (np.arange(len(quads), dtype=np.int32) * 4)[:, None, None]
    faces = (base + np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int32)).reshape(-1, 3)
    obj.add_mesh(_core("Mesh")(verts, faces, color=tuple(float(c) for c in color)))


def _quad(obj, v0, v1, v2, v3, color):
    _quads(obj, [(v0, v1, v2, v3)], color)


def _cylinder(obj, cx, cy, radius, height, color, segments=24):
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    base = np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang), np.zeros_like(ang)], 1)
    top = base + np.array([0.0, 0.0, height])
    verts = np.concatenate([base, top, [[cx, cy, height]]]).astype(np.float32)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[i, j, segments + j], [i, segments + j, segments + i]]  # wall (outward)
        faces += [[segments + i, segments + j, 2 * segments]]  # top cap
    obj.add_mesh(_core("Mesh")(verts, np.array(faces, dtype=np.int32), color=tuple(float(c) for c in color)))


def _room(obj, ox, oy, pillars, rng, randomize: bool, style: str = "hyperdrone"):
    """Checker floor, four walls (no ceiling: the sun must get in) and the pillars, all
    translated to the room's grid offset (ox, oy). The pybullet style drops the walls,
    stretches the floor to the horizon and uses pybullet's colours."""
    if style == "pybullet":
        x0, x1 = PB_FLOOR_X[0] + ox, PB_FLOOR_X[1] + ox
        y0, y1 = PB_FLOOR_Y[0] + oy, PB_FLOOR_Y[1] + oy
        tiles = ([], [])
        for i in range(int(round((x1 - x0) / TILE))):
            for j in range(int(round((y1 - y0) / TILE))):
                xa, ya = x0 + i * TILE, y0 + j * TILE
                tiles[(i + j) % 2].append(((xa, ya, 0), (xa + TILE, ya, 0), (xa + TILE, ya + TILE, 0), (xa, ya + TILE, 0)))
        for quads, c in zip(tiles, PB_FLOOR_RGB):
            _quads(obj, quads, c)
        for p in pillars:
            color = p["color"] if randomize else PB_PILLAR_RGB
            _cylinder(obj, ox + p["x"], oy + p["y"], p["r"], p["h"], color)
        return
    x0, x1 = ROOM_X[0] + ox, ROOM_X[1] + ox
    y0, y1 = ROOM_Y[0] + oy, ROOM_Y[1] + oy
    if randomize:
        g = float(rng.uniform(0.3, 0.7))
        floor = (
            tuple(np.clip(g + 0.12 + rng.uniform(-0.05, 0.05, 3), 0.05, 1.0)),
            tuple(np.clip(g - 0.12 + rng.uniform(-0.05, 0.05, 3), 0.05, 1.0)),
        )
        walls = tuple(tuple(rng.uniform(0.55, 0.9, 3)) for _ in range(4))
    else:
        floor, walls = FLOOR_RGB, WALL_RGB
    nx, ny = int(round((ROOM_X[1] - ROOM_X[0]) / TILE)), int(round((ROOM_Y[1] - ROOM_Y[0]) / TILE))
    tiles = ([], [])
    for i in range(nx):
        for j in range(ny):
            xa, ya = x0 + i * TILE, y0 + j * TILE
            tiles[(i + j) % 2].append(((xa, ya, 0), (xa + TILE, ya, 0), (xa + TILE, ya + TILE, 0), (xa, ya + TILE, 0)))
    for quads, c in zip(tiles, floor):
        _quads(obj, quads, c)
    h = ROOM_H
    _quad(obj, (x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h), walls[0])  # far wall (+x)
    _quad(obj, (x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h), walls[1])  # right wall (-y)
    _quad(obj, (x0, y1, 0), (x1, y1, 0), (x1, y1, h), (x0, y1, h), walls[2])  # left wall (+y)
    _quad(obj, (x0, y0, 0), (x0, y1, 0), (x0, y1, h), (x0, y0, h), walls[3])  # back wall (-x)
    for p in pillars:
        _cylinder(obj, ox + p["x"], oy + p["y"], p["r"], p["h"], p["color"])


def room_offsets(n: int, style: str = "hyperdrone") -> np.ndarray:
    """Grid offsets (n, 2) of the arenas of one chunk (one column along y for the wall-less
    pybullet style, so nothing of another arena lies ahead of a camera)."""
    k = np.arange(n)
    if style == "pybullet":
        return np.stack([np.zeros(n), k * (PB_FLOOR_Y[1] - PB_FLOOR_Y[0])], 1).astype(np.float32)
    cols = int(math.ceil(math.sqrt(n)))
    return np.stack([(k % cols) * ROOM_PITCH[0], (k // cols) * ROOM_PITCH[1]], 1).astype(np.float32)


def build_scene(layouts: list, offsets: np.ndarray, rng, randomize: bool, style: str = "hyperdrone"):
    """One static scene holding every arena of the chunk, lit by a sun and a fill light."""
    if style not in ARENA_STYLES:
        raise ValueError(f"arena style must be one of {ARENA_STYLES}")
    scene = _core("Scene")()
    world = _core("Object")(name="arenas")
    for pillars, (ox, oy) in zip(layouts, offsets):
        _room(world, float(ox), float(oy), pillars, rng, randomize, style)
    scene.add_object(world)
    light = _core("SceneLight")
    if style == "pybullet":
        scene.add_light(light.directional(direction=(0.25, 0.15, -1.0), color=(0.9, 0.9, 0.9)))
        scene.set_environment(ambient=PB_AMBIENT, background=PB_SKY)
        return scene
    if randomize:
        az, el = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(35, 75))
        sun = (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), -math.sin(el))
        sun_rgb = tuple(np.clip(rng.uniform(0.85, 1.1) * np.array([1.0, 0.97, 0.92]) + rng.uniform(-0.05, 0.05, 3), 0, 1.2))
        fill = (-sun[0] * 0.6, -sun[1] * 0.6, -0.3)
        fill_rgb = tuple(rng.uniform(0.25, 0.5, 3))
        ambient = tuple(np.full(3, rng.uniform(0.25, 0.45)))
        sky = tuple(np.clip(np.array(SKY) + rng.uniform(-0.15, 0.1, 3), 0, 1))
    else:
        sun, sun_rgb = (0.3, 0.2, -1.0), (1.0, 1.0, 1.0)
        fill, fill_rgb, ambient, sky = (-0.5, 0.4, -0.3), (0.4, 0.4, 0.5), AMBIENT, SKY
    scene.add_light(light.directional(direction=sun, color=sun_rgb))
    scene.add_light(light.directional(direction=fill, color=fill_rgb))
    scene.set_environment(ambient=ambient, background=sky)
    return scene


def _jit_fast(factory):
    """Construct a native hyperlux/hyperdrone component through the cached fast path
    (HYPERLUX_SKIP_BUILD skips the per-construction CMake staleness check, minutes on this
    tree); if the cached artifact turns out to be missing, build it once and retry."""
    os.environ["HYPERLUX_SKIP_BUILD"] = "1"
    try:
        return factory()
    except Exception as error:  # noqa: BLE001  (a missing artifact surfaces as many types)
        print(f"[INFO] native component not cached yet ({type(error).__name__}); building it once ...")
        os.environ.pop("HYPERLUX_SKIP_BUILD", None)
        try:
            return factory()
        finally:
            os.environ["HYPERLUX_SKIP_BUILD"] = "1"


def _rot(yaw: np.ndarray) -> np.ndarray:
    """(n, 2, 2) rollout-frame -> world rotation matrices."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(np.float32)


# --- flight: velocity intent -> position reference -> RAPTOR (or pure kinematics) ---------
class Flight:
    """`VelCommander` semantics on top of hyperdrone's position controller: the reference
    is re-anchored to the true planar position whenever the executed command changes, then
    integrates the commanded velocity — plus LEAD_S seconds of that velocity as
    feed-forward, because RAPTOR targets zero velocity at its setpoint and otherwise trails a
    moving reference by ~0.5 s. Altitude is never re-anchored (the reference stays absolute:
    START height plus integrated climb) and gets a small outer P+I loop, since RAPTOR by itself
    sags up to ~0.25 m at the fast end of the speed range. Each drone keeps a fixed heading
    (its rollout frame's +x, set as the body yaw at reset — the dynamics are the same at
    every heading). Velocities handed to `step` are world-frame. `kinematic=True` skips the
    dynamics: the camera rides the reference."""

    def __init__(self, n: int, model: str, seed: int, kinematic: bool, camera: str = "body", device: str = "cpu"):
        if camera not in ("body", "level"):
            raise ValueError("camera must be 'body' or 'level'")
        self.n, self.kinematic, self.camera = n, kinematic, camera
        self.sim = None
        if not kinematic:
            from hyperdrone import dynamics

            params = {"dynamics": {"dynamics": {"model": model}, "integration": {"dt": TICK / SUBSTEPS}}}
            self.sim = _jit_fast(
                lambda: dynamics.Sim(n, parameters=params, seed=seed, device=device, control="position")
            )
            self.dt = float(self.sim.dt)
        else:
            self.dt = TICK
        self.mount = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, CAMERA_Z]], dtype=np.float32)

    def reset(self, positions: np.ndarray, yaw=None) -> None:
        self.pos = np.array(positions, dtype=np.float32)
        self.yaw = np.zeros(self.n, dtype=np.float32) if yaw is None else np.asarray(yaw, dtype=np.float32)
        self.anchor = self.pos.copy()
        self.integral = np.zeros((self.n, 3), dtype=np.float32)
        self.alt_i = np.zeros(self.n, dtype=np.float32)
        self.since = np.zeros(self.n, dtype=np.int64)
        self.last_id = np.full(self.n, -1, dtype=np.int64)
        if self.sim is not None:
            self.sim.reset()
            self.sim.state["position"] = self.pos
            half = self.yaw / 2.0  # start AT the heading: a yaw step at reset topples RAPTOR
            self.sim.state["orientation"] = np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], 1)

    def position(self) -> np.ndarray:
        return self.sim.state["position"] if self.sim is not None else self.pos.copy()

    def attitude(self) -> np.ndarray:
        """(n, 4) body attitude quaternion wxyz (yaw-only when kinematic)."""
        if self.sim is None:
            half = self.yaw / 2.0
            return np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], 1).astype(np.float32)
        return self.sim.state["orientation"]

    def step(self, v_exec: np.ndarray, exec_id: np.ndarray) -> None:
        """Advance one control tick under the executed world-frame velocity command (n, 3)."""
        switched = exec_id != self.last_id
        if switched.any():
            self.anchor[switched, :2] = self.position()[switched, :2]  # planar only
            self.integral[switched, :2] = 0.0
            self.since[switched] = 0
            self.last_id[switched] = exec_id[switched]
        self.integral += v_exec * TICK
        self.since += 1
        ref = self.anchor + self.integral
        if self.sim is None:
            self.pos = ref
            return
        z = self.position()[:, 2]
        self.alt_i = np.clip(self.alt_i + ALT_KI * (ref[:, 2] - z) * TICK, -ALT_IMAX, ALT_IMAX)
        ramp = np.minimum(1.0, self.since / LEAD_RAMP).astype(np.float32)[:, None]
        target = np.zeros((self.n, 4), dtype=np.float32)
        target[:, :3] = ref + v_exec * LEAD_S * ramp
        target[:, 2] = ref[:, 2] + ALT_KP * (ref[:, 2] - z) + self.alt_i
        target[:, 3] = self.yaw
        for _ in range(SUBSTEPS):
            self.sim.step(target)

    def camera_bases(self, fov_deg: float) -> np.ndarray:
        """(n, 12) hyperlux camera bases: position, dir_00, dir_du, dir_dv (world FLU)."""
        if self.sim is not None and self.camera == "body":
            return self.sim.camera_bases(mount=self.mount, fov=fov_deg, aspect=1.0)
        # level: a virtual gimbal — the camera sits at the body's camera point but keeps
        # pitch and roll at zero and looks along the heading. Same layout hyperlux.make_camera
        # produces: dir_00 = forward - du/2 - dv/2, du = right * s, dv = -z * s, s = 2 tan(fov/2).
        s = 2.0 * math.tan(math.radians(fov_deg) / 2.0)
        c, sn = np.cos(self.yaw), np.sin(self.yaw)
        bases = np.zeros((self.n, 12), dtype=np.float32)
        bases[:, 0:3] = self.position() + np.array([0.0, 0.0, CAMERA_Z], dtype=np.float32)
        bases[:, 6:9] = np.stack([sn * s, -c * s, np.zeros_like(c)], 1)  # image right = clockwise
        bases[:, 9:12] = (0.0, 0.0, -s)
        bases[:, 3:6] = np.stack([c, sn, np.zeros_like(c)], 1) - bases[:, 6:9] / 2.0 - bases[:, 9:12] / 2.0
        return bases


# --- the dataset ---------------------------------------------------------------------------
def gen(
    n_rollouts: int,
    length: int,
    seed: int = 0,
    randomize: bool = False,
    fov_deg: float = DEFAULT_FOV,
    model: str = DEFAULT_MODEL,
    kinematic: bool = False,
    camera: str = "body",
    save_depth: bool = False,
    fidelity: str = "medium",
    scene: str = "pillars",
    n_rooms: int = None,
    houses: int = 2,
    house_first: int = 0,
    catalog: str = "procthor",
    arena_style: str = "hyperdrone",
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Fly intervention trials in hyperdrone and return the sequence-format dataset dict of
    `gen_wm_dataset.gen` (same keys, dtypes and semantics, plus the extras listed above).
    `n_rollouts` pillar courses (scene 'pillars'/'mixed') and `n_rooms` furnished-room
    trials spread over `houses` catalog houses (scene 'rooms'/'mixed')."""
    import hyperlux
    from hyperlux.camera import RGB, CameraSpec, Depth

    if scene not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}")
    if not 1.0 <= fov_deg < 179.0:
        raise ValueError("fov_deg must be in [1, 179)")
    rng = np.random.default_rng(seed)
    fov_half = fov_deg / 2.0 - FOV_MARGIN_DEG
    n_pillars = n_rollouts if scene in ("pillars", "mixed") else 0
    n_rooms = (n_rollouts if n_rooms is None else n_rooms) if scene in ("rooms", "mixed") else 0
    R, L = n_pillars + n_rooms, length
    if R < 1:
        raise ValueError("nothing to fly")
    frames = np.zeros((R, L, IMG_RES, IMG_RES, 3), dtype=np.uint8)
    depth = np.zeros((R, L, IMG_RES, IMG_RES), dtype=np.float16) if save_depth else None
    actions = np.zeros((R, L, 4), dtype=np.float32)
    act_id = np.zeros((R, L), dtype=np.int16)
    seg = np.zeros((R, L), dtype=np.int16)
    dists = np.zeros((R, L), dtype=np.float32)
    pos = np.zeros((R, L, 3), dtype=np.float32)
    quat = np.zeros((R, L, 4), dtype=np.float32)
    pillars_meta = np.full((R, 3, 2), np.nan, dtype=np.float32)
    in_path = np.zeros(R, dtype=bool)
    speed = np.zeros(R, dtype=np.float32)
    scene_type = np.zeros(R, dtype=np.int8)
    scene_id = ["pillars"] * R
    yaw = np.zeros(R, dtype=np.float32)
    start_world = np.tile(START.astype(np.float32), (R, 1))
    cf = np.zeros((R, L, len(ACTION_VECS), len(HORIZONS), 2), dtype=np.uint8)
    vis = np.ones((R, L, len(ACTION_VECS)), dtype=np.uint8)

    # per-rollout draws in gen_wm_dataset.gen's order (its env.reset seed draw included), so
    # the pillar courses and command schedules of a seed match the pybullet generator's
    layouts, lat = [], np.zeros(R, dtype=np.int64)
    for r in range(R):
        rng.integers(2**31 - 1)
        in_path[r] = r % 2 == 0
        speed[r] = rng.uniform(*SPEED_RANGE)
        if r < n_pillars:
            layouts.append(sample_pillars(rng, in_path=bool(in_path[r]), randomize=randomize))
            pillars_meta[r, : len(layouts[r])] = [(p["x"], p["y"]) for p in layouts[r]]
        else:
            scene_type[r] = 1
        act_id[r], seg[r] = _schedule(rng, L, passive=(r % 3 == 2))
        lat[r] = int(rng.integers(0, 3)) if randomize else 0

    flight = Flight(CHUNK, model, seed, kinematic, camera, device)
    outputs = (RGB("rgba8"), Depth()) if save_depth else (RGB("rgba8"),)
    cameras = (CameraSpec("front", outputs),)
    timing = {"render": 0.0, "dynamics": 0.0, "scenes": 0.0}

    def make_renderer():
        renderer = _jit_fast(
            lambda: hyperlux.HyperLux(width=IMG_RES, height=IMG_RES, batch_size=CHUNK, cameras=cameras, fidelity=fidelity)
        )
        _CORE["backend"] = renderer.backend
        return renderer

    def fly(rows, renderer, starts, yaws):
        """One chunk: rollouts `rows` from world `starts` (n, 3) with headings `yaws` (n,).
        Records frames/pos/quat/actions; returns world planar positions (n, L, 2)."""
        n = len(rows)
        full_start = np.tile(START.astype(np.float32), (CHUNK, 1))
        full_start[:n] = starts
        full_start[n:, 0] = 1e4 + np.arange(CHUNK - n) * 10.0  # spare drones parked far away
        full_yaw = np.zeros(CHUNK, dtype=np.float32)
        full_yaw[:n] = yaws
        flight.reset(full_start, full_yaw)
        rot = _rot(full_yaw)  # local -> world
        world_xy = np.zeros((n, L, 2), dtype=np.float32)
        last_v = np.zeros((CHUNK, 3), dtype=np.float32)
        last_id = np.full(CHUNK, -2, dtype=np.int64)
        for t in range(L):
            t0 = time.perf_counter()
            renderer.set_cameras(flight.camera_bases(fov_deg))
            renderer.render()
            frames[rows, t] = renderer.outputs[RGB("rgba8")].read()[:n, :, :, :3]
            if save_depth:
                depth[rows, t] = renderer.outputs[Depth()].read()[:n]
            t1 = time.perf_counter()
            p = flight.position()
            rel = p[:n] - full_start[:n]
            local = np.einsum("nji,nj->ni", rot[:n], rel[:, :2])  # world -> local (R^T)
            pos[rows, t, :2], pos[rows, t, 2] = local, rel[:, 2] + START[2]
            world_xy[:, t] = p[:n, :2]
            quat[rows, t] = flight.attitude()[:n]
            ids = act_id[rows, t].astype(np.int64)
            actions[rows, t] = speed[rows, None] * ACTION_VECS[ids]  # the intent, recorded
            exec_ids = act_id[rows, np.maximum(t - lat[rows], 0)].astype(np.int64)
            v_exec = speed[rows, None] * ACTION_VECS[exec_ids]
            if randomize:
                v_exec = v_exec * (1.0 + rng.normal(0.0, 0.08, size=v_exec.shape))
            last_v[:n, :2] = np.einsum("nij,nj->ni", rot[:n], v_exec[:, :2])  # local -> world
            last_v[:n, 2] = v_exec[:, 2]
            last_id[:n] = exec_ids
            flight.step(last_v, last_id)
            timing["render"] += t1 - t0
            timing["dynamics"] += time.perf_counter() - t1
        return world_xy

    def report(rows):
        if not verbose:
            return
        for r in rows:
            held = sorted({ACTION_NAMES[i] for i in act_id[r][seg[r] > 0]})
            print(
                f"  rollout {r + 1}/{R} ({scene_id[r]}, {'in-path' if in_path[r] else 'clear'}, {speed[r]:.2f}x, "
                f"{'passive' if seg[r].max() == 0 else '+'.join(held)})"
            )

    # --- pillar arenas: one static scene per chunk ---------------------------------------
    for c0 in range(0, n_pillars, CHUNK):
        rows = np.arange(c0, min(c0 + CHUNK, n_pillars))
        n = len(rows)
        t0 = time.perf_counter()
        offsets = room_offsets(CHUNK, arena_style)
        arena = build_scene([layouts[r] for r in rows], offsets[:n], rng, randomize, arena_style)
        renderer = make_renderer()
        renderer.init(arena)
        timing["scenes"] += time.perf_counter() - t0
        starts = np.concatenate([offsets[:n], np.zeros((n, 1), np.float32)], 1) + START.astype(np.float32)
        fly(rows, renderer, starts, np.zeros(n, dtype=np.float32))
        pil = pillars_meta[rows]
        d = np.linalg.norm(pos[rows][:, :, None, :2] - pil[:, None, :, :], axis=3)
        dists[rows] = np.where(np.isnan(d).all(axis=2), 9.0, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2))
        report(rows)
        del renderer
    if n_pillars:
        sub = {k: v[:n_pillars] for k, v in (("frames", frames), ("pillars", pillars_meta), ("pos", pos), ("speed", speed))}
        sub["fov_half_deg"] = np.float32(fov_half)
        cf[:n_pillars], vis[:n_pillars] = counterfactual_labels(sub)

    # --- furnished houses: labels from the geometry (wm_rooms) -----------------------------
    if n_rooms:
        from wm_rooms import FLIGHT_Z, DistanceField, house_references, load_house, probe_renderer
        from wm_rooms import PILLAR_R as ROOM_PILLAR_R

        refs = house_references(houses, catalog=catalog, first=house_first)
        per_house = int(math.ceil(n_rooms / houses))
        r_next = n_pillars
        for house_id, ref in refs:
            rows_house = np.arange(r_next, min(r_next + per_house, R))
            if not len(rows_house):
                break
            r_next = rows_house[-1] + 1
            t0 = time.perf_counter()
            house = load_house(ref, fidelity=fidelity)
            if randomize:
                house.set_environment(ambient=tuple(np.full(3, rng.uniform(0.4, 0.8))))
            rig = _jit_fast(probe_renderer)  # a hyperlux renderer binds to one scene for life
            rig.init(house)
            if verbose:
                print(f"[INFO] house {house_id}: probing geometry ...")
            field = DistanceField(house, rig, rig.scene_bounds, verbose=verbose)
            del rig
            renderer = make_renderer()
            renderer.init(house)
            timing["scenes"] += time.perf_counter() - t0
            starts_xy, yaws = [], []
            for r in rows_house:
                found = field.sample_start(rng, in_path=bool(in_path[r])) or field.sample_start(rng, in_path=not in_path[r])
                if found is None:
                    raise RuntimeError(f"house {house_id}: no flyable start found (try another --house-first)")
                starts_xy.append(found[0])
                yaws.append(found[1])
                scene_id[r] = house_id
            starts_xy, yaws = np.array(starts_xy, dtype=np.float32), np.array(yaws, dtype=np.float32)
            yaw[rows_house], start_world[rows_house, :2], start_world[rows_house, 2] = yaws, starts_xy, FLIGHT_Z
            for c0 in range(0, len(rows_house), CHUNK):
                rows = rows_house[c0 : c0 + CHUNK]
                sl = slice(c0, c0 + len(rows))
                starts = np.concatenate([starts_xy[sl], np.full((len(rows), 1), FLIGHT_Z, np.float32)], 1)
                world_xy = fly(rows, renderer, starts, yaws[sl])
                dists[rows] = field.clearance(world_xy) + ROOM_PILLAR_R
                for k, r in enumerate(rows):
                    cf[r], vis[r] = field.counterfactual(pos[r], starts_xy[c0 + k], float(yaws[c0 + k]), float(speed[r]), fov_half)
                report(rows)
            del renderer

    if verbose:
        n_frames = R * L
        print(
            f"[INFO] {n_frames} frames: render {timing['render']:.1f}s ({n_frames / max(timing['render'], 1e-9):.0f} fps), "
            f"dynamics {timing['dynamics']:.1f}s, scenes {timing['scenes']:.1f}s, "
            f"backend={'kinematic' if kinematic else model + '/raptor/' + str(flight.sim.device)}, renderer={_CORE.get('backend', '?')}"
        )
    data = {
        "frames": frames,
        "actions": actions,
        "act_id": act_id,
        "seg": seg,
        "dists": dists,
        "pos": pos,
        "pillars": pillars_meta,
        "in_path": in_path,
        "speed": speed,
        "randomized": np.uint8(randomize),
        "horizons": np.array(HORIZONS, dtype=np.int16),
        "a_norm": A_NORM,
        "danger_r": np.float32(DANGER_R),
        # hyperdrone extras (older consumers ignore them)
        "fov_deg": np.float32(fov_deg),
        "fov_half_deg": np.float32(fov_half),
        "camera": np.array(camera),
        "arena_style": np.array(arena_style),
        "quat": quat,
        "scene_type": scene_type,
        "scene_id": np.array(scene_id),
        "yaw": yaw,
        "start_world": start_world,
        "cf_labels": cf,
        "cf_visible": vis,
        "source": np.array("hyperdrone"),
        "drone_model": np.array("kinematic" if kinematic else model),
        "sim_dt": np.float32(flight.dt),
        "kinematic": np.uint8(kinematic),
    }
    if save_depth:
        data["depth"] = depth
    return data


def tracking_ratio(data: dict) -> float:
    """Mean of |measured planar speed| / |commanded planar speed| over steps where the same
    moving command had been held for >= 0.5 s — how faithfully the dynamics follow intent."""
    v = np.diff(data["pos"], axis=1) * CTRL_HZ  # (R, L-1, 3)
    cmd = data["actions"][:, :-1, :2]
    ids, held = data["act_id"], np.zeros(data["act_id"].shape, dtype=bool)
    for r in range(ids.shape[0]):
        run = 0
        for t in range(ids.shape[1]):
            run = run + 1 if t > 0 and ids[r, t] == ids[r, t - 1] else 1
            held[r, t] = run >= CTRL_HZ // 2
    mask = held[:, :-1] & (np.linalg.norm(cmd, axis=2) > 1e-6)
    if not mask.any():
        return float("nan")
    return float((np.linalg.norm(v[:, :, :2], axis=2)[mask] / np.linalg.norm(cmd, axis=2)[mask]).mean())


def tilt_deg(data: dict) -> tuple:
    """(mean, 95th percentile, max) of the airframe's tilt from vertical, degrees."""
    w, x, y, z = (data["quat"][..., i] for i in range(4))
    cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)  # body z . world z
    tilt = np.degrees(np.arccos(cos_tilt))
    return float(tilt.mean()), float(np.percentile(tilt, 95)), float(tilt.max())


def preview(data: dict, path: str, rollouts: int = 8, columns: int = 8, scale: int = 2) -> None:
    """A PNG mosaic: one row per rollout (spread over the dataset, so a mixed set shows both
    kinds), frames evenly spaced in time."""
    F = data["frames"]
    R, L = F.shape[:2]
    rows = np.unique(np.linspace(0, R - 1, min(rollouts, R)).astype(int))
    cols = np.linspace(0, L - 1, min(columns, L)).astype(int)
    mosaic = np.concatenate([np.concatenate([F[r, t] for t in cols], 1) for r in rows], 0)
    mosaic = np.kron(mosaic, np.ones((scale, scale, 1), dtype=np.uint8))
    try:
        import imageio

        imageio.imwrite(path, mosaic)
    except ImportError:
        import cv2

        cv2.imwrite(path, mosaic[:, :, ::-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rollouts", type=int, default=32, help="pillar-course rollouts (scene pillars/mixed)")
    ap.add_argument("--len", dest="length", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="pillar shape/colour, lights, walls, latency, actuation noise")
    ap.add_argument("--fov-deg", type=float, default=DEFAULT_FOV, help="square camera FOV; 60 = pybullet, 40 = calibrated Tello")
    ap.add_argument("--drone-model", default=DEFAULT_MODEL, help="hyperdrone.dynamics.MODELS entry")
    ap.add_argument("--kinematic", action="store_true", help="no dynamics: the camera rides the reference")
    ap.add_argument("--camera", default="body", choices=("body", "level"), help="body-fixed (pybullet-like) or virtually stabilised camera")
    ap.add_argument("--save-depth", action="store_true", help="also store float16 metric depth")
    ap.add_argument("--fidelity", default="medium", choices=("low", "medium", "high", "veryhigh"))
    ap.add_argument("--scene", default="pillars", choices=SCENES, help="pillar arenas, furnished catalog houses, or both")
    ap.add_argument("--rooms", type=int, default=None, help="room rollouts (scene rooms/mixed; default = --rollouts)")
    ap.add_argument("--houses", type=int, default=2, help="how many catalog houses the room rollouts are spread over")
    ap.add_argument("--house-first", type=int, default=0, help="first catalog index to use")
    ap.add_argument("--catalog", default="procthor", choices=("procthor", "hssd"))
    ap.add_argument("--arena-style", default="hyperdrone", choices=ARENA_STYLES, help="pillar arena look: hyperdrone (walls, grey floor) or pybullet (white sky, blue checker plane, red pillars)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"), help="hyperdrone dynamics device (the renderer follows HYPERLUX_BACKEND)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    n_roll, length = (10, 100) if args.selftest else (args.rollouts, args.length)
    out = args.out.replace(".npz", "_selftest.npz") if args.selftest else args.out

    tag = " (randomized)" if args.randomize else ""
    print(
        f"[INFO] flying {n_roll} {args.scene} rollouts x {length} steps{tag} in hyperdrone, "
        f"camera {args.fov_deg:g} deg {args.camera}, {'kinematic' if args.kinematic else args.drone_model} ..."
    )
    t0 = time.perf_counter()
    data = gen(
        n_roll,
        length,
        seed=args.seed,
        randomize=args.randomize,
        fov_deg=args.fov_deg,
        model=args.drone_model,
        kinematic=args.kinematic,
        camera=args.camera,
        save_depth=args.save_depth,
        fidelity=args.fidelity,
        scene=args.scene,
        n_rooms=args.rooms,
        houses=args.houses,
        house_first=args.house_first,
        catalog=args.catalog,
        arena_style=args.arena_style,
        device=args.device,
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, **data)
    png = out.replace(".npz", "_preview.png")
    preview(data, png)

    R = data["frames"].shape[0]
    rates = {}
    for k in HORIZONS:
        pairs = as_pairs(data, k)
        rates[k] = (len(pairs["c"]), float(pairs["c"].mean()))
    rate_str = ", ".join(f"k={k}: n={n} pos={p:.2f}" for k, (n, p) in rates.items())
    n_seg = int(sum(data["seg"][r].max() for r in range(R)))
    held = sorted({ACTION_NAMES[i] for r in range(R) for i in data["act_id"][r]})
    ratio = tracking_ratio(data)
    tilt = tilt_deg(data)
    kinds = {name: int((data["scene_type"] == code).sum()) for code, name in ((0, "pillars"), (1, "rooms"))}
    print(
        f"WM-DATA OK (hyperdrone): {R} rollouts x {length} steps @ {CTRL_HZ} Hz "
        f"({kinds['pillars']} pillars + {kinds['rooms']} rooms in {len(set(data['scene_id'].tolist()) - {'pillars'})} houses), "
        f"{n_seg} held intervention segments, labels [{rate_str}], "
        f"speed tracking {ratio:.2f}x of command, {time.perf_counter() - t0:.0f}s, saved {out}"
    )
    print(f"  commands held: {held}; camera {args.fov_deg:g} deg ({args.camera}) -> label FOV half-angle "
          f"{float(data['fov_half_deg']):g} deg; airframe tilt mean/p95/max {tilt[0]:.1f}/{tilt[1]:.1f}/{tilt[2]:.1f} deg; "
          f"preview {png}")

    if args.selftest:
        assert data["frames"].dtype == np.uint8, "frames must be uint8"
        assert data["frames"].shape[2:] == (IMG_RES, IMG_RES, 3), "bad frame shape"
        drift = np.abs(data["pos"][:, 0, :] - START).max()
        assert drift < 0.1, f"rollouts do not start at START (drift {drift:.2f} m)"
        env_max = np.abs(ACTION_VECS).max() * SPEED_RANGE[1]
        assert np.abs(data["actions"]).max() <= env_max + 1e-6
        assert data["speed"].max() - data["speed"].min() > 0.3, "speeds too uniform"
        assert n_seg >= R, "too few held segments for counterfactual contrast"
        assert len(held) >= 4, f"too little action diversity ({held})"
        for k, (n, p) in rates.items():
            assert n > 0, f"no valid windows at k={k}"
            assert 0.03 < p < 0.97, f"labels too imbalanced at k={k} ({p:.2f})"
        # hyperdrone-specific: the pictures move and the dynamics follow the intent
        assert float(data["frames"].std()) > 10, "frames look constant"
        first_diff = np.abs(data["frames"][:, H_MAX].astype(int) - data["frames"][:, 0]).mean()
        assert first_diff > 2, "no visual motion over the longest horizon"
        if not args.kinematic:
            assert 0.7 < ratio < 1.3, f"dynamics do not track the command ({ratio:.2f}x)"
        # the stored oracle must agree with the executed-future labels where both exist
        agree = []
        for r in range(R):
            for t in range(length - H_MAX):
                if data["seg"][r, t] != data["seg"][r, t + H_MAX]:
                    continue
                flown = float(data["dists"][r, t : t + H_MAX + 1].min() < DANGER_R)
                agree.append(flown == float(data["cf_labels"][r, t, data["act_id"][r, t], -1, 0]))
        assert np.mean(agree) > 0.8, f"counterfactual oracle disagrees with flown labels ({np.mean(agree):.2f})"
        print("SELFTEST OK")


if __name__ == "__main__":
    main()
    sys.exit(0)
