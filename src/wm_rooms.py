"""
Real rooms for the hyperdrone world-model dataset
=================================================
`gen_wm_dataset_hyperdrone.py --scene rooms|mixed` flies the same intervention trials inside
ProcTHOR houses (hyperlux ships 200 of them as content-addressed GLBs; HSSD works the same
way). Pillars had known centres, so "danger" and the counterfactual labels were planar
distances to a list of points. A furnished room has no such list, so this module measures
the geometry itself, in the renderer's own frame:

  1. **Occupancy at flight height.** Every 0.1 m cell of the house footprint is probed with a
     horizontal ring of depth rays (4 cameras x 90 deg, a +-3.6 deg vertical band around the
     flight altitude) plus one ray straight up. Cells whose rays hit surfaces mark those
     surface cells as occupied (a lidar map of walls and furniture at drone height); cells
     without a ceiling are outdoors and count as occupied too.
  2. **Planar distance field.** A Euclidean distance transform of the occupancy map gives the
     surface clearance everywhere, plus the nearest occupied cell (for "is the threat inside
     the camera FOV?"). `dists` = clearance + PILLAR_R, so DANGER_R / CRIT_R keep the same
     *surface* meaning they had around a 0.18 m pillar (0.52 m / 0.17 m).
  3. **Starts.** Indoor cells with >= 0.9 m clearance and a heading whose straight-ahead free
     distance is 1.3-2.6 m ("in-path", an obstacle where the pillar used to be) or >= 3.5 m
     ("clear"), with the first metre of the approach at least 0.5 m clear sideways.
  4. **Labels.** `dists` along the flown path, and `counterfactual_labels`-equivalent arrays
     (`cf_labels`, `cf_visible`) by rolling every candidate command forward through the field,
     exactly like gen_wm_dataset does through the pillar list. They are stored in the npz,
     and `gen_wm_dataset.counterfactual_labels` returns stored labels when present.

Honest notes: the band is thin (+-3.6 deg), so a low table the drone flies over is not an
obstacle, and a lamp shade above the band is not either; the drone flies through walls like
it flew through pillars (no contact physics); the field is at one altitude, so `climb` keeps
its "changes the view, not the planar label" role.
"""

import math
import os

import numpy as np

from gen_wm_dataset import ACTION_VECS, CTRL_HZ, H_MAX, HORIZONS, RADII

GRID = 0.1  # m
FLIGHT_Z = 1.0  # probing altitude (START[2])
PILLAR_R = 0.18  # the pillar radius the danger rings were defined around
RIG_W, RIG_H, RIG_FOV = 32, 2, 90.0  # per-camera depth probe: 32 columns, +-3.6 deg band
RIG_CAMERAS = 5  # 4 horizontal headings + 1 up
PROBE_BATCH = 512  # cells per probe render (fixed: part of the JIT key)
MAX_RAY = 8.0  # ignore hits farther than this (other rooms through doorways still count)
INDOOR_CEILING = (0.3, 4.0)  # ceiling distance range that says "inside the house"
CLEAR_START = 0.9  # m surface clearance required at a start cell
CLEAR_APPROACH = 0.5  # m sideways clearance along the first metre
IN_PATH_RANGE, CLEAR_MIN = (1.3, 2.6), 3.5  # straight-ahead free distance per rollout kind
COURSE_LEN, CLEAR_COURSE = 3.3, 0.6  # closed-loop courses: finish line 3 m ahead; clear courses stay 0.6 m from everything


def house_references(count: int, catalog: str = "procthor", first: int = 0) -> list:
    """(id, conta reference) of `count` houses from hyperlux's pinned catalogs."""
    import hyperlux

    if catalog == "procthor":
        dataset = hyperlux.datasets.procthor.Train200()
    elif catalog == "hssd":
        dataset = hyperlux.datasets.hssd.Train125()
    else:
        raise ValueError("catalog must be 'procthor' or 'hssd'")
    if not 0 <= first and first + count <= len(dataset):
        raise ValueError(f"{catalog} catalog has {len(dataset)} scenes")
    return [(dataset[i].id, dataset[i].reference) for i in range(first, first + count)]


def load_house(reference, fidelity: str = "medium"):
    """Resolve a conta reference (download on first use) and load it as a hyperlux scene."""
    import conta
    import hyperlux

    path = conta.resolve(reference)
    return hyperlux.load_scene(str(path), fidelity=fidelity)


def _rig_bases(origins: np.ndarray) -> np.ndarray:
    """(len(origins) * RIG_CAMERAS, 12) camera bases: 4 horizontal 90 deg cameras at yaw
    0/90/180/270 deg and one looking up, all RIG_W x RIG_H."""
    s = 2.0 * math.tan(math.radians(RIG_FOV) / 2.0)
    sv = s * RIG_H / RIG_W
    bases = np.zeros((len(origins), RIG_CAMERAS, 12), dtype=np.float32)
    for k in range(4):
        psi = k * math.pi / 2.0
        f = np.array([math.cos(psi), math.sin(psi), 0.0])
        du = np.array([math.sin(psi), -math.cos(psi), 0.0]) * s  # image right = clockwise
        dv = np.array([0.0, 0.0, -sv])
        bases[:, k, 0:3] = origins
        bases[:, k, 3:6] = f - du / 2.0 - dv / 2.0
        bases[:, k, 6:9] = du
        bases[:, k, 9:12] = dv
    up_du, up_dv = np.array([0.0, -s, 0.0]), np.array([-sv, 0.0, 0.0])
    bases[:, 4, 0:3] = origins
    bases[:, 4, 3:6] = np.array([0.0, 0.0, 1.0]) - up_du / 2.0 - up_dv / 2.0
    bases[:, 4, 6:9] = up_du
    bases[:, 4, 9:12] = up_dv
    return bases.reshape(-1, 12)


class DistanceField:
    """Occupancy and planar clearance of one house at FLIGHT_Z, in renderer coordinates."""

    def __init__(self, scene, renderer, bounds: dict, verbose: bool = True):
        from hyperlux.camera import Depth

        self.scene, self.renderer = scene, renderer
        c, h = np.asarray(bounds["center"], dtype=float), np.asarray(bounds["half_extent"], dtype=float)
        self.x0, self.y0 = c[0] - h[0] - 0.5, c[1] - h[1] - 0.5
        self.nx, self.ny = int(np.ceil((2 * h[0] + 1.0) / GRID)), int(np.ceil((2 * h[1] + 1.0) / GRID))
        xs = self.x0 + (np.arange(self.nx) + 0.5) * GRID
        ys = self.y0 + (np.arange(self.ny) + 0.5) * GRID
        cells = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        origins = np.concatenate([cells, np.full((len(cells), 1), FLIGHT_Z)], 1).astype(np.float32)
        occupied = np.zeros((self.nx, self.ny), dtype=bool)
        ceiling = np.full(len(cells), np.inf, dtype=np.float32)
        clearance = np.full(len(cells), np.inf, dtype=np.float32)
        # column azimuths of a horizontal rig camera (image right = clockwise)
        u = (np.arange(RIG_W) + 0.5) / RIG_W * 2.0 - 1.0
        az0 = -np.arctan(u * math.tan(math.radians(RIG_FOV) / 2.0))  # (RIG_W,)
        planar_scale = np.sqrt(1.0 + (u * math.tan(math.radians(RIG_FOV) / 2.0)) ** 2)
        for start in range(0, len(origins), PROBE_BATCH):
            chunk = origins[start : start + PROBE_BATCH]
            padded = np.concatenate([chunk, np.repeat(chunk[-1:], PROBE_BATCH - len(chunk), 0)])
            renderer.set_cameras(_rig_bases(padded))
            renderer.render()
            depth = renderer.outputs[Depth()].read().reshape(PROBE_BATCH, RIG_CAMERAS, RIG_H, RIG_W)[: len(chunk)]
            horiz = depth[:, :4].min(axis=2)  # (n, 4, W) z-depth per heading/column
            planar = horiz * planar_scale[None, None, :]  # ray length projected on the plane
            clearance[start : start + len(chunk)] = planar.min(axis=(1, 2))
            ceiling[start : start + len(chunk)] = depth[:, 4, :, RIG_W // 2 - 1 : RIG_W // 2 + 1].min(axis=(1, 2))
            hit = np.isfinite(planar) & (planar < MAX_RAY)
            k = np.arange(4)[None, :, None] * (math.pi / 2.0)
            az = k + az0[None, None, :]
            hx = chunk[:, 0, None, None] + planar * np.cos(az)
            hy = chunk[:, 1, None, None] + planar * np.sin(az)
            ix = np.floor((hx[hit] - self.x0) / GRID).astype(int)
            iy = np.floor((hy[hit] - self.y0) / GRID).astype(int)
            ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            occupied[ix[ok], iy[ok]] = True
        self.indoor = ((ceiling > INDOOR_CEILING[0]) & (ceiling < INDOOR_CEILING[1])).reshape(self.nx, self.ny)
        inside_geometry = (clearance < 0.06).reshape(self.nx, self.ny)
        self.occupied = occupied | inside_geometry | ~self.indoor
        from scipy import ndimage

        dist, (ii, jj) = ndimage.distance_transform_edt(~self.occupied, return_indices=True)
        self.field = (dist * GRID).astype(np.float32)  # surface clearance (m) per cell
        self.nearest = np.stack([ii, jj], -1)  # nearest occupied cell per cell
        if verbose:
            frac = float(self.indoor.mean())
            print(
                f"  field {self.nx}x{self.ny} cells @ {GRID} m: indoor {frac:.0%}, occupied {float(self.occupied.mean()):.0%}, "
                f"free clearance median {np.median(self.field[~self.occupied]):.2f} m"
            )

    # --- lookups -------------------------------------------------------------------------
    def _cell(self, xy: np.ndarray):
        ix = np.clip(np.floor((xy[..., 0] - self.x0) / GRID).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor((xy[..., 1] - self.y0) / GRID).astype(int), 0, self.ny - 1)
        return ix, iy

    def clearance(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear surface clearance (m) at world planar positions (..., 2)."""
        gx = np.clip((xy[..., 0] - self.x0) / GRID - 0.5, 0, self.nx - 1.001)
        gy = np.clip((xy[..., 1] - self.y0) / GRID - 0.5, 0, self.ny - 1.001)
        i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
        fx, fy = gx - i0, gy - j0
        f = self.field
        return (
            f[i0, j0] * (1 - fx) * (1 - fy)
            + f[i0 + 1, j0] * fx * (1 - fy)
            + f[i0, j0 + 1] * (1 - fx) * fy
            + f[i0 + 1, j0 + 1] * fx * fy
        )

    def threat_xy(self, xy: np.ndarray) -> np.ndarray:
        """World position of the nearest occupied cell for each planar position."""
        ix, iy = self._cell(xy)
        n = self.nearest[ix, iy]
        return np.stack([self.x0 + (n[..., 0] + 0.5) * GRID, self.y0 + (n[..., 1] + 0.5) * GRID], -1)

    def free_ahead(self, xy: np.ndarray, yaw: float, limit: float = 8.0) -> float:
        """Straight-ahead distance to the first occupied cell."""
        steps = np.arange(0, limit, GRID / 2)
        pts = xy[None, :] + steps[:, None] * np.array([math.cos(yaw), math.sin(yaw)])
        ix, iy = self._cell(pts)
        hits = np.flatnonzero(self.occupied[ix, iy])
        return float(steps[hits[0]]) if len(hits) else limit

    def sample_start(self, rng, in_path: bool, tries: int = 20000, course: bool = False):
        """A (world xy, yaw) start for one rollout, or None when the house has no such spot.

        `course=True` is the closed-loop test's stricter definition, mirroring the pillar
        courses where the 3 m finish line is always reachable: an in-path obstacle must be
        passable (free space behind it and on at least one side, i.e. furniture rather than a
        wall), and a clear course must stay >= CLEAR_COURSE from everything along the whole
        straight line, so an evasion on it really is a false trigger."""
        cand = np.argwhere(~self.occupied & self.indoor & (self.field >= CLEAR_START))
        if not len(cand):
            return None
        for _ in range(tries):
            ix, iy = cand[rng.integers(len(cand))]
            xy = np.array([self.x0 + (ix + 0.5) * GRID, self.y0 + (iy + 0.5) * GRID])
            yaw = float(rng.uniform(-math.pi, math.pi))
            fwd = np.array([math.cos(yaw), math.sin(yaw)])
            left = np.array([-fwd[1], fwd[0]])
            ahead = self.free_ahead(xy, yaw)
            if in_path and not (IN_PATH_RANGE[0] <= ahead <= IN_PATH_RANGE[1]):
                continue
            if not in_path and ahead < CLEAR_MIN:
                continue
            approach = xy[None, :] + np.linspace(0.0, 1.0, 11)[:, None] * fwd
            if self.clearance(approach).min() < CLEAR_APPROACH:
                continue
            if course:
                if in_path:
                    behind = xy[None, :] + np.linspace(ahead + 0.7, COURSE_LEN, 12)[:, None] * fwd
                    if self.clearance(behind).min() < 0.3:
                        continue  # a wall, not a passable piece of furniture
                    # a free-standing obstacle, like a pillar: a detour must exist on BOTH sides
                    # (a wall with one doorway passes a one-sided test but is a navigation problem,
                    # not an avoidance one)
                    sides_ok = True
                    for sign in (1.0, -1.0):
                        detour = xy[None, :] + np.linspace(0.6, COURSE_LEN, 14)[:, None] * fwd + sign * 0.9 * left[None, :]
                        if self.clearance(detour).min() < 0.3:
                            sides_ok = False
                            break
                    if not sides_ok:
                        continue
                else:
                    straight = xy[None, :] + np.linspace(0.0, COURSE_LEN, 17)[:, None] * fwd
                    if self.clearance(straight).min() < CLEAR_COURSE:
                        continue
            return xy, yaw
        return None

    def sample_open_start(self, rng, tries: int = 4000, ahead_min: float = 2.0):
        """A (world xy, yaw) for the clutter-endurance test: >= CLEAR_START clearance, indoors,
        >= `ahead_min` free straight ahead and 0.5 m of side clearance over the first metre."""
        cand = np.argwhere(~self.occupied & self.indoor & (self.field >= CLEAR_START))
        if not len(cand):
            return None
        for _ in range(tries):
            ix, iy = cand[rng.integers(len(cand))]
            xy = np.array([self.x0 + (ix + 0.5) * GRID, self.y0 + (iy + 0.5) * GRID])
            yaw = float(rng.uniform(-math.pi, math.pi))
            if self.free_ahead(xy, yaw) < ahead_min:
                continue
            approach = xy[None, :] + np.linspace(0.0, 1.0, 11)[:, None] * np.array([math.cos(yaw), math.sin(yaw)])
            if self.clearance(approach).min() < CLEAR_APPROACH:
                continue
            return xy, yaw
        return None

    # --- labels ----------------------------------------------------------------------------
    def counterfactual(self, local_pos: np.ndarray, start_xy: np.ndarray, yaw: float, speed: float, fov_half_deg: float):
        """gen_wm_dataset.counterfactual_labels for one rollout, through the field instead of
        the pillar list: labels (L, n_actions, n_horizons, n_radii) uint8, visible (L, n_actions)."""
        L = local_pos.shape[0]
        taus = np.arange(H_MAX + 1) / CTRL_HZ
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        cf = np.zeros((L, len(ACTION_VECS), len(HORIZONS), len(RADII)), dtype=np.uint8)
        vis = np.ones((L, len(ACTION_VECS)), dtype=np.uint8)
        cos_fov = math.cos(math.radians(fov_half_deg))
        p0 = local_pos[:, :2]
        for i, v in enumerate(speed * ACTION_VECS):
            path = p0[:, None, :] + taus[None, :, None] * v[:2]  # (L, K+1, 2) local
            world = path @ rot.T + start_xy
            d = self.clearance(world) + PILLAR_R  # (L, K+1) centre-equivalent distance
            for j, k in enumerate(HORIZONS):
                dmin = d[:, : k + 1].min(axis=1)
                for q, rad in enumerate(RADII):
                    cf[:, i, j, q] = (dmin < rad).astype(np.uint8)
            worst = d.argmin(axis=1)  # the path point of closest approach
            threat = self.threat_xy(world[np.arange(L), worst])
            rel = (threat - start_xy) @ rot - p0  # back to the local (camera) frame
            in_fov = rel[:, 0] > np.linalg.norm(rel, axis=1) * cos_fov
            vis[:, i] = np.where(cf[:, i, :, 0].any(axis=1) & ~in_fov, 0, 1)
        return cf, vis

    def occupancy_png(self, path: str, starts=None) -> None:
        """Debug picture: occupied = black, outdoors = grey, free shaded by clearance, starts red."""
        img = np.full((self.ny, self.nx, 3), 255, dtype=np.uint8)
        free = ~self.occupied
        shade = (255 - np.clip(self.field / 2.0, 0, 1) * 160).astype(np.uint8)
        for ch in range(3):
            img[..., ch] = np.where(free.T, shade.T, 0)
        img[~self.indoor.T & ~free.T] = 128
        for (xy, yaw) in starts or []:
            ix, iy = self._cell(np.asarray(xy))
            img[max(iy - 1, 0) : iy + 2, max(ix - 1, 0) : ix + 2] = (220, 30, 30)
            tip = np.asarray(xy) + 0.6 * np.array([math.cos(yaw), math.sin(yaw)])
            jx, jy = self._cell(tip)
            img[jy, jx] = (30, 120, 220)
        try:
            import imageio

            imageio.imwrite(path, np.kron(img[::-1], np.ones((3, 3, 1), dtype=np.uint8)))
        except ImportError:
            pass


def probe_renderer(batch: int = PROBE_BATCH):
    """The depth-only probe rig renderer (one JIT config, shared by every house)."""
    import hyperlux
    from hyperlux.camera import CameraSpec, Depth

    return hyperlux.HyperLux(
        width=RIG_W, height=RIG_H, batch_size=batch * RIG_CAMERAS, cameras=(CameraSpec("probe", (Depth(),)),), fidelity="low"
    )


__all__ = ["DistanceField", "FLIGHT_Z", "PILLAR_R", "house_references", "load_house", "probe_renderer"]

if __name__ == "__main__":  # quick look at one house's field: python src/wm_rooms.py [index]
    import sys
    import time

    index = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    house_id, ref = house_references(1, first=index)[0]
    t0 = time.perf_counter()
    scene = load_house(ref)
    rig = probe_renderer()
    rig.init(scene)
    field = DistanceField(scene, rig, rig.scene_bounds)
    rng = np.random.default_rng(0)
    starts = [s for s in (field.sample_start(rng, in_path=(k % 2 == 0)) for k in range(16)) if s is not None]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", f"field_{house_id}.png")
    field.occupancy_png(out, starts)
    print(f"{house_id}: {len(starts)}/16 starts found in {time.perf_counter() - t0:.1f}s -> {out}")
