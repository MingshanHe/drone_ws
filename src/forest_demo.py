"""
Forest demo — fly the trained policies through a pillar "forest" and render it
==============================================================================
Not part of the original lesson: a visual test of the step-6 learned policy (and
the step-4 hand MPC / reactive baseline for comparison) on a longer, denser
course than anything they trained on (training courses have 1-3 pillars over
3 m; this one has a staggered grid of pillars over FOREST_LEN m).

The drone gets one mission — fly from START to the goal point on the centre
line at the far end — and every policy shares that goal (`goal_seek`). The
avoidance policy decides every 1/12 s from the on-board 64x64 camera (vision
only, exactly like `run_episode`): when it says `forward` the drone flies toward
the goal, any other choice overrides the mission until the threat is passed. Pillar positions are used to stage the course, score clearance
and draw the picture, never to fly (except the reactive baseline's documented
privileged evasion direction).

Outputs (in src/output/):
  forest_<policy>.mp4   top-down third-person video: trajectory so far, drone
                        marker, crash marks, and the drone's own camera inset
  forest_compare.png    one top-down frame with every policy's full trajectory

Run (from ~/drone, drone_ws env):
  python src/forest_demo.py                        # video of the learned policy
  python src/forest_demo.py --video wm-mpc         # video of the hand MPC instead
  python src/forest_demo.py --seed 3 --rows 8      # another / longer forest
  python src/forest_demo.py --rows 4 --row-dx 1.2 --col-dy 1.8   # sparser forest
  python src/forest_demo.py --onboard              # + first-person video (forest_<policy>_onboard.mp4)
Needs output/world_model.pth and output/ppo_wm_policy.zip (steps 2 and 6).
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import (  # noqa: E402
    ACTION_NAMES,
    ACTION_VECS,
    COLLISION_R,
    FORWARD,
    START,
    VelCommander,
    grab_frame,
    make_ctrl,
    make_env,
    nearest_planar,
)
from learn_policy import LearnedPolicy, _load_policy, zip_path  # noqa: E402
from train_world_model import load_model  # noqa: E402
from wm_closed_loop import DECIDE_EVERY, ReactivePolicy, WMPolicy, reactive_policy, wm_policy  # noqa: E402

OUT_DIR = os.path.join(HERE, "output")
ROW_DX = 0.75  # distance between pillar rows along x (m)
COL_DY = 1.15  # distance between pillars inside a row (m)
JITTER = 0.15  # random offset of each pillar (m)
Y_HALF = 2.0  # forest half-width (the policies were trained within |y| < 2.4)
FIRST_ROW_X = 1.2
PILLAR_R, PILLAR_H = 0.18, 1.4  # same look as training (randomize=False)
PILLAR_RGBA = [0.80, 0.32, 0.22, 1]
VIEW_W, VIEW_H = 1000, 600  # top-down render size (px)
VIEW_HALF_Y = 3.4  # half-height of the overhead view in metres (wider than the forest)
CAM_HEIGHT = 80.0  # far above + narrow FOV ~ orthographic: pillars stay round
OB_W, OB_H, OB_FOV = 800, 600, 60.0  # onboard render: same 60 deg vertical FOV as the drone camera
OB_FPS = 24  # onboard video frame rate (render every 2 control steps)
CRUISE_V = 0.8  # mission cruise speed (m/s), the same as the policies' forward
GOAL_GAIN = 1.0  # slow-down gain near the goal: speed = min(CRUISE_V, gain * distance)
GOAL_TOL = 0.10  # arrived when within this planar distance of the goal (m)
COLORS_BGR = {  # OpenCV draws in BGR
    "learned": (220, 120, 30),
    "wm-mpc": (60, 170, 60),
    "reactive": (30, 140, 245),
}


def forest_layout(rows: int, seed: int, row_dx: float = ROW_DX, col_dy: float = COL_DY) -> list:
    """Staggered grid of pillar centres (x, y). The first row always has a
    pillar right on the start line (y ~ 0), so the drone must dodge at once."""
    rng = np.random.default_rng(seed)
    pillars = []
    for i in range(rows):
        x0 = FIRST_ROW_X + i * row_dx
        offset = 0.0 if i % 2 == 0 else col_dy / 2  # staggered like a forest
        y = -Y_HALF + offset
        while y <= Y_HALF + 1e-6:
            pillars.append(
                (
                    float(x0 + rng.uniform(-JITTER, JITTER)),
                    float(y + rng.uniform(-JITTER, JITTER)),
                )
            )
            y += col_dy
    # guarantee a blocker in front of the start
    pillars.append((FIRST_ROW_X - 0.1, float(rng.uniform(-0.1, 0.1))))
    return pillars


def place_pillars(env, pillars) -> None:
    import pybullet as p

    for px, py in pillars:
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=PILLAR_R,
            length=PILLAR_H,
            rgbaColor=PILLAR_RGBA,
            physicsClientId=env.CLIENT,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=[px, py, PILLAR_H / 2],
            physicsClientId=env.CLIENT,
        )


class TopCamera:
    """Overhead third-person camera looking straight down at the course, plus
    the world->pixel projection used to draw trajectories on its images."""

    def __init__(self, goal_x: float):
        import pybullet as p

        cx = goal_x / 2
        height = CAM_HEIGHT
        fov = 2 * np.degrees(np.arctan(VIEW_HALF_Y / height))
        # looking down with up = +y: image right = +x (flight direction), image top = +y (left)
        self.view = p.computeViewMatrix([cx, 0, height], [cx, 0, 0], [0, 1, 0])
        self.proj = p.computeProjectionMatrixFOV(
            fov, VIEW_W / VIEW_H, height - 5.0, height + 1.0
        )
        V = np.array(self.view).reshape(4, 4).T  # pybullet is column-major
        P = np.array(self.proj).reshape(4, 4).T
        self.PV = P @ V

    def render(self, env) -> np.ndarray:
        import pybullet as p

        _, _, rgba, _, _ = p.getCameraImage(
            VIEW_W,
            VIEW_H,
            self.view,
            self.proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=env.CLIENT,
        )
        img = np.reshape(np.asarray(rgba, dtype=np.uint8), (VIEW_H, VIEW_W, 4))
        return np.ascontiguousarray(img[:, :, 2::-1])  # RGBA -> BGR for OpenCV

    def px(self, xyz) -> tuple:
        c = self.PV @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
        ndc = c[:3] / c[3]
        return (
            int(round((ndc[0] + 1) / 2 * VIEW_W)),
            int(round((1 - ndc[1]) / 2 * VIEW_H)),
        )


class OnboardCamera:
    """High-resolution first-person view rendered from the drone's own pose,
    exactly the way BaseAviary renders the 64x64 policy camera (look direction
    follows the drone's attitude, up vector stays world-up), so this video and
    the policy's input show the same view. Purely for viewing: the policy
    still sees only the 64x64 frame."""

    def render(self, env, state) -> np.ndarray:
        import pybullet as p

        rot = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        eye = state[0:3] + rot @ np.array([0.06, 0.0, 0.0])  # just ahead of the body
        target = eye + rot @ np.array([1.0, 0.0, 0.0])
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(OB_FOV, OB_W / OB_H, 0.03, 30.0)
        _, _, rgba, _, _ = p.getCameraImage(
            OB_W, OB_H, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=env.CLIENT
        )
        img = np.reshape(np.asarray(rgba, dtype=np.uint8), (OB_H, OB_W, 4))
        return np.ascontiguousarray(img[:, :, 2::-1])


def goal_seek(state, goal) -> np.ndarray:
    """The mission layer: fly toward the goal point at cruise speed, slowing
    down in the last metre so the drone stops on it, and hold START's altitude.
    Uses only the drone's own odometry and the commanded goal."""
    d = np.asarray(goal[:2]) - state[0:2]
    dist = float(np.linalg.norm(d))
    speed = min(CRUISE_V, GOAL_GAIN * dist)
    vxy = d / max(dist, 1e-6) * speed
    vz = 1.5 * (START[2] - float(state[2]))
    return np.array([vxy[0], vxy[1], vz, 0.0])


def fly(env, policy, pillars, goal_x: float, tmax: int, cam=None, writer=None, name="",
        ob_writer=None):
    """One flight from START to the goal point (goal_x, 0) through the forest.

    Two layers, like a real autonomy stack:
      * mission layer — `goal_seek` steers toward the one shared goal point;
      * avoidance layer — the vision-only policy decides at 12 Hz exactly as in
        `run_episode`. When it says `forward` (nothing threatening) the mission
        command is flown; any other choice (veer / slow / hover) overrides it.
    So every policy ends at the same point, and the policies differ only in how
    they dodge on the way. 48 Hz PID underneath, rendering optional: `writer`
    gets the top-down video (12 fps), `ob_writer` the onboard one (24 fps)."""
    ob_cam = OnboardCamera() if ob_writer is not None else None
    obs, _ = env.reset(seed=0)
    place_pillars(env, pillars)
    cmd = VelCommander(make_ctrl(), env.CTRL_TIMESTEP)
    cmd.reset(obs[0][0:3])
    policy.begin(pillars)
    goal = np.array([goal_x, 0.0])
    state, a_id, v_cmd, label = obs[0], FORWARD, ACTION_VECS[FORWARD], "goal"
    path, min_clear, crashes, in_crash = [state[0:3].copy()], 9.0, [], False
    reached, n_avoid, n_dec = False, 0, 0
    for t in range(tmax):
        if t % DECIDE_EVERY == 0:
            frame = grab_frame(env)
            arrived = float(np.linalg.norm(goal - state[0:2])) < GOAL_TOL
            if arrived:  # mission done: hold position on the goal
                reached, v_cmd, label = True, goal_seek(state, goal), "arrived"
            else:
                a_id = policy.decide(frame, state)
                n_dec += 1
                if a_id == FORWARD:  # no threat: follow the mission
                    v_cmd, label = goal_seek(state, goal), "goal"
                else:  # the avoidance policy takes over
                    v_cmd, label = ACTION_VECS[a_id], f"AVOID {ACTION_NAMES[a_id]}"
                    n_avoid += 1
            if writer is not None:
                writer.write(_overlay(cam, cam.render(env), path, crashes, frame,
                                      name, label, t, min_clear, goal))
        if ob_writer is not None and t % (48 // OB_FPS) == 0:
            ob_writer.write(_onboard_overlay(ob_cam.render(env, state), frame, path, crashes,
                                             pillars, goal, name, label, t, min_clear))
        if t % DECIDE_EVERY == 0 and arrived:
            break
        obs, _, _, _, _ = env.step(cmd.rpm(state, v_cmd).reshape(1, 4))
        state = obs[0]
        path.append(state[0:3].copy())
        d = nearest_planar(state[0:2], pillars)
        min_clear = min(min_clear, d)
        if d < COLLISION_R and not in_crash:  # count each contact once
            crashes.append(state[0:3].copy())
        in_crash = d < COLLISION_R
    if writer is not None:  # hold the last frame for one second
        last = _overlay(cam, cam.render(env), path, crashes, grab_frame(env), name,
                        "arrived" if reached else "timeout", t, min_clear, goal)
        for _ in range(12):
            writer.write(last)
    if ob_writer is not None:
        last = _onboard_overlay(ob_cam.render(env, state), grab_frame(env), path, crashes,
                                pillars, goal, name, "arrived" if reached else "timeout",
                                t, min_clear)
        for _ in range(OB_FPS):
            ob_writer.write(last)
    return {
        "path": np.array(path),
        "min_clear": min_clear,
        "crashes": crashes,
        "reached": reached,
        "end_err": float(np.linalg.norm(goal - state[0:2])),
        "avoid_share": n_avoid / max(n_dec, 1),
        "time_s": len(path) / 48.0,
    }


def _minimap(pillars, path, crashes, goal, color, w=260):
    """Small 2D top-down map: pillars (collision circles), path so far, goal."""
    import cv2

    x0, x1 = -0.3, goal[0] + 0.5
    y0, y1 = -Y_HALF - 0.6, Y_HALF + 0.6
    s = w / (x1 - x0)
    h = int(round((y1 - y0) * s))
    img = np.full((h, w, 3), 245, np.uint8)

    def uv(x, y):  # image right = +x, image top = +y
        return int(round((x - x0) * s)), int(round((y1 - y) * s))

    for px, py in pillars:
        cv2.circle(img, uv(px, py), max(2, int(PILLAR_R * s)), (40, 80, 200), -1, cv2.LINE_AA)
    if len(path) > 1:
        pts = np.array([uv(q[0], q[1]) for q in path[::2]], np.int32)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
    for q in crashes:
        cv2.drawMarker(img, uv(q[0], q[1]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    cv2.drawMarker(img, uv(*goal), (0, 0, 0), cv2.MARKER_STAR, 16, 2)
    cv2.circle(img, uv(path[-1][0], path[-1][1]), 5, color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 1)
    return img


def _text(img, s, org, scale=0.6, color=(255, 255, 255)):
    import cv2

    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale >= 0.6 else 1,
                cv2.LINE_AA)


def _onboard_overlay(img, fpv, path, crashes, pillars, goal, name, label, t, min_clear):
    import cv2

    color = COLORS_BGR.get(name, (255, 255, 255))
    # the policy's own field of view: the central OB_H x OB_H square (60 x 60 deg)
    m = (OB_W - OB_H) // 2
    cv2.rectangle(img, (m, 0), (m + OB_H - 1, OB_H - 1), (255, 255, 255), 1)
    cv2.line(img, (OB_W // 2 - 10, OB_H // 2), (OB_W // 2 + 10, OB_H // 2), (255, 255, 255), 1)
    cv2.line(img, (OB_W // 2, OB_H // 2 - 10), (OB_W // 2, OB_H // 2 + 10), (255, 255, 255), 1)
    # control-layer banner: green = mission, red = avoidance override
    avoiding = label.startswith("AVOID")
    banner = (40, 40, 220) if avoiding else (60, 160, 60)
    cv2.rectangle(img, (0, 0), (330, 36), banner, -1)
    _text(img, label if avoiding else f"MISSION: {label}", (10, 25), 0.7)
    _text(img, f"policy: {name}   t = {t / 48.0:4.1f} s", (10, 60))
    _text(img, f"min clearance {min_clear:.2f} m   crashes {len(crashes)}", (10, 84))
    mm = _minimap(pillars, path, crashes, goal, color)
    img[10:10 + mm.shape[0], OB_W - 10 - mm.shape[1]:OB_W - 10] = mm
    # what the policy actually receives: the 64x64 camera frame
    inset = cv2.resize(fpv[:, :, ::-1], (160, 160), interpolation=cv2.INTER_NEAREST)
    img[OB_H - 170:OB_H - 10, 10:170] = inset
    cv2.rectangle(img, (10, OB_H - 170), (170, OB_H - 10), (255, 255, 255), 2)
    _text(img, "policy input 64x64", (10, OB_H - 176), 0.5)
    return img


def _draw_path(img, cam, path, color, thick=3):
    import cv2

    if len(path) > 1:
        pts = np.array([cam.px(q) for q in path[::2]], dtype=np.int32)
        cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)


def _draw_crashes(img, cam, crashes):
    import cv2

    for q in crashes:
        u, v = cam.px(q)
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 4)


def _draw_goal(img, cam, goal):
    import cv2

    u, v = cam.px((goal[0], goal[1], START[2]))
    cv2.drawMarker(img, (u, v), (0, 0, 0), cv2.MARKER_STAR, 30, 3)
    cv2.putText(img, "goal", (u + 12, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                cv2.LINE_AA)


def _overlay(cam, img, path, crashes, fpv, name, action, t, min_clear, goal):
    import cv2

    color = COLORS_BGR.get(name, (255, 255, 255))
    _draw_goal(img, cam, goal)
    _draw_path(img, cam, path, color)
    _draw_crashes(img, cam, crashes)
    u, v = cam.px(path[-1])
    cv2.circle(img, (u, v), 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (u, v), 9, color, 3, cv2.LINE_AA)
    # the drone's own 64x64 camera, upscaled, top-left
    inset = cv2.resize(fpv[:, :, ::-1], (192, 192), interpolation=cv2.INTER_NEAREST)
    img[10:202, 10:202] = inset
    cv2.rectangle(img, (10, 10), (202, 202), (255, 255, 255), 2)
    lines = [
        f"policy: {name}",
        f"t = {t / 48.0:4.1f} s   cmd: {action}",
        f"min clearance {min_clear:.2f} m   crashes {len(crashes)}",
    ]
    for i, s in enumerate(lines):
        y = VIEW_H - 70 + 24 * i
        cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "drone camera", (14, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--row-dx", type=float, default=ROW_DX, help="row spacing along x (m)")
    ap.add_argument("--col-dy", type=float, default=COL_DY, help="pillar spacing inside a row (m)")
    ap.add_argument("--video", default="learned", choices=list(COLORS_BGR) + ["none"])
    ap.add_argument("--onboard", action="store_true",
                    help="also render a first-person video of the --video policy")
    ap.add_argument("--model", default=None, help="world-model checkpoint (default output/world_model.pth)")
    ap.add_argument("--no-learned", action="store_true",
                    help="skip the PPO policy (it was trained on the pybullet world model's latents)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="fly this many forests (seed, seed+1, ...); video/picture only for the first")
    ap.add_argument("--tag", default="", help="output name tag: forest_<tag>_<policy>.mp4 / forest_<tag>_compare.png")
    args = ap.parse_args()
    torch.set_num_threads(1)  # tiny nets: threads only add overhead
    tag = f"{args.tag}_" if args.tag else ""

    enc, pred, cheads, nhead, meta = load_model(args.model) if args.model else load_model()
    print(f"[INFO] checkpoint {args.model or 'output/world_model.pth'}: trained on "
          f"{meta.get('source', 'pybullet')}, camera {meta.get('fov_deg', 60):g} deg")
    policies = {
        "reactive": lambda: reactive_policy(enc, nhead, meta),
        "wm-mpc": lambda: wm_policy(enc, pred, cheads, meta),
    }
    if os.path.exists(zip_path()) and not args.no_learned:
        model = _load_policy(zip_path())
        policies["learned"] = lambda: LearnedPolicy(model, enc, pred, cheads, meta)
    elif args.video == "learned":
        raise SystemExit(f"no {zip_path()} (or --no-learned) — pick --video wm-mpc/reactive/none")

    env = make_env()
    os.makedirs(OUT_DIR, exist_ok=True)
    totals = {name: {"crashes": [], "min_clear": [], "reached": [], "time_s": []} for name in policies}
    results, pillars, goal_x, cam = {}, None, None, None
    for k in range(args.seeds):
        seed = args.seed + k
        pillars = forest_layout(args.rows, seed, args.row_dx, args.col_dy)
        goal_x = FIRST_ROW_X + (args.rows - 1) * args.row_dx + 1.0
        tmax = int(goal_x / 0.25 * 48)  # generous: even a slow detour fits
        cam = TopCamera(goal_x)
        if k == 0:
            print(f"[INFO] forest: {len(pillars)} pillars, {args.rows} rows, goal x={goal_x:.1f} m, "
                  f"start {START[:2].tolist()}, {args.seeds} seed(s) from {args.seed}")
        for name in ["reactive", "wm-mpc", "learned"]:
            if name not in policies:
                continue
            writer = None
            if k == 0 and name == args.video:
                path = os.path.join(OUT_DIR, f"forest_{tag}{name}.mp4")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12,
                                         (VIEW_W, VIEW_H))
            ob_writer = None
            if k == 0 and name == args.video and args.onboard:
                ob_path = os.path.join(OUT_DIR, f"forest_{tag}{name}_onboard.mp4")
                ob_writer = cv2.VideoWriter(ob_path, cv2.VideoWriter_fourcc(*"mp4v"), OB_FPS,
                                            (OB_W, OB_H))
            res = fly(env, policies[name](), pillars, goal_x, tmax, cam, writer, name, ob_writer)
            if writer is not None:
                writer.release()
                print(f"[INFO] saved {path}")
            if ob_writer is not None:
                ob_writer.release()
                print(f"[INFO] saved {ob_path}")
            if k == 0:
                results[name] = res
            t = totals[name]
            t["crashes"].append(len(res["crashes"]))
            t["min_clear"].append(res["min_clear"])
            t["reached"].append(bool(res["reached"]))
            t["time_s"].append(res["time_s"])
            print(
                f"  seed {seed} {name:9s}: reached={res['reached']}  end error={res['end_err']:.2f} m  "
                f"crashes={len(res['crashes'])}  min clearance={res['min_clear']:.2f} m  "
                f"avoiding {res['avoid_share']:.0%} of decisions  time={res['time_s']:.1f} s"
            )
    if args.seeds > 1:
        parts = []
        for name, t in totals.items():
            parts.append(
                f"{name}: crashes/forest {np.mean(t['crashes']):.2f}, crash-free {np.mean([c == 0 for c in t['crashes']]):.0%}, "
                f"reached {np.mean(t['reached']):.0%}, min clear {np.mean(t['min_clear']):.2f} m, time {np.mean(t['time_s']):.1f} s"
            )
        print(f"FOREST OK ({args.seeds} forests x {args.rows} rows): " + " | ".join(parts))

    # comparison picture: empty forest from above + every trajectory (first forest)
    pillars = forest_layout(args.rows, args.seed, args.row_dx, args.col_dy)
    cam = TopCamera(goal_x)
    env.reset(seed=0)
    place_pillars(env, pillars)
    img = cam.render(env)
    for name, res in results.items():
        _draw_path(img, cam, res["path"], COLORS_BGR[name], 3)
    for name, res in results.items():
        _draw_crashes(img, cam, res["crashes"])
    u0, v0 = cam.px(START)
    cv2.circle(img, (u0, v0), 8, (0, 0, 0), -1, cv2.LINE_AA)
    _draw_goal(img, cam, (goal_x, 0.0))
    for i, (name, res) in enumerate(results.items()):
        s = f"{name}: crashes {len(res['crashes'])}, min clear {res['min_clear']:.2f} m"
        y = 28 + 26 * i
        cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS_BGR[name], 2, cv2.LINE_AA)
    out = os.path.join(OUT_DIR, f"forest_{tag}compare.png")
    cv2.imwrite(out, img)
    env.close()
    print(f"[INFO] saved {out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
