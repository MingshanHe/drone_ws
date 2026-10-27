"""
Short closed-loop videos inside hyperdrone: one course, one policy, top-down + onboard
======================================================================================
Flies one `hd_closed_loop` course (same seed -> same pillars as the pybullet scoreboard) with
the reactive or wm policy and writes an MP4 (12 fps, one frame per policy decision, ~7 s) and a
GIF: left, a hyperlux top-down render of the arena with the flown path, crash marks and the
finish line; right, the 64x64 frame the policy actually saw, upscaled, with the command it chose.

  python src/hd_video.py --model src/output/world_model_pybullet_ref.pth --arena-style pybullet --seed 1001 --policy wm --tag ref_pb
Outputs src/output/report/hd_<tag>.mp4 and .gif.
"""

import argparse
import math
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import ACTION_NAMES, ACTION_VECS, COLLISION_R, FORWARD, START  # noqa: E402
from gen_wm_dataset_hyperdrone import ARENA_STYLES, CHUNK, DEFAULT_FOV, IMG_RES, Flight, _jit_fast, _rot, build_scene, room_offsets  # noqa: E402
from hd_closed_loop import Episode, RoomEpisode  # noqa: E402
from train_world_model import MODEL, load_model  # noqa: E402
from wm_closed_loop import DECIDE_EVERY, GOAL_X, TMAX, ReactivePolicy, WMPolicy  # noqa: E402

OUT_DIR = os.path.join(HERE, "output", "report")
TOP = 320  # top-down render size (square)
TOP_FOV = 60.0
TOP_CENTER = (1.6, 0.0)  # course centre (x, y)
TOP_HEIGHT = 6.5  # camera height: half-width = H tan(fov/2) = 3.75 m


def top_camera(renderer):
    cx, cy = TOP_CENTER
    # straight down, image "up" = +x (forward). right = forward x up = (0,0,-1) x (1,0,0) = (0,-1,0)
    return renderer.camera(position=(cx, cy, TOP_HEIGHT), look_at=(cx, cy, 0.0), up=(1.0, 0.0, 0.0), fov=TOP_FOV)


def to_px(x: float, y: float):
    half = TOP_HEIGHT * math.tan(math.radians(TOP_FOV) / 2.0)
    u = int(round(TOP / 2 - (y - TOP_CENTER[1]) / half * (TOP / 2)))
    v = int(round(TOP / 2 - (x - TOP_CENTER[0]) / half * (TOP / 2)))
    return u, v


def main() -> None:
    import cv2
    import hyperlux
    from hyperlux.camera import RGB, CameraSpec

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--arena-style", default="pybullet", choices=ARENA_STYLES)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--clear", action="store_true", help="a clear course instead of an in-path one")
    ap.add_argument("--policy", default="wm", choices=("wm", "reactive"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scene", default="pillars", choices=("pillars", "rooms"))
    ap.add_argument("--house", type=int, default=16, help="catalog house index for --scene rooms")
    args = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.scene == "rooms":
        return main_rooms(args)

    enc, pred, cheads, nhead, meta = load_model(args.model, device="cpu")
    policy = WMPolicy(enc, pred, cheads, meta, speed=args.speed) if args.policy == "wm" else ReactivePolicy(enc, nhead)
    ep = Episode(args.seed, policy, not args.clear, args.speed, False)
    flight = Flight(CHUNK, "crazyflie", 0, False, "body")
    cams = (CameraSpec("front", (RGB("rgba8"),)),)
    onboard = _jit_fast(lambda: hyperlux.HyperLux(width=IMG_RES, height=IMG_RES, batch_size=CHUNK, cameras=cams, fidelity="medium"))
    topdown = _jit_fast(lambda: hyperlux.HyperLux(width=TOP, height=TOP, batch_size=1, cameras=cams, fidelity="medium"))
    offsets = room_offsets(CHUNK, args.arena_style)
    scene = build_scene([ep.pillars], offsets[:1], np.random.default_rng(0), False, args.arena_style)
    onboard.init(scene)
    topdown.init(scene)
    topdown.set_cameras(top_camera(topdown)[None])
    topdown.render()
    background = np.ascontiguousarray(topdown.outputs[RGB("rgba8")].read()[0, :, :, :3][:, :, ::-1])  # BGR for cv2
    for p in ep.pillars:  # sanity ring: drawn circles must sit on the rendered pillars
        cv2.circle(background, to_px(p["x"], p["y"]), 4, (255, 255, 255), 1, cv2.LINE_AA)
    gx = to_px(GOAL_X, 0.0)[1]
    cv2.line(background, (0, gx), (TOP - 1, gx), (40, 40, 40), 1, cv2.LINE_AA)

    starts = START.astype(np.float32) + np.concatenate([offsets, np.zeros((CHUNK, 1), np.float32)], 1)
    starts[1:, 0] += 1e4
    flight.reset(starts)
    exec_ids = np.full(CHUNK, -3, dtype=np.int64)
    v = np.zeros((CHUNK, 3), dtype=np.float32)
    state = np.zeros(20)
    path, crashes, frames_out = [], [], []
    label, in_crash = "forward", False
    for t in range(TMAX):
        pos = flight.position()[0]
        if t % DECIDE_EVERY == 0:
            onboard.set_cameras(flight.camera_bases(DEFAULT_FOV))
            onboard.render()
            frame = np.ascontiguousarray(onboard.outputs[RGB("rgba8")].read()[0, :, :, :3])
            if ep.active:
                state[0:3] = pos
                ep.a_id = ep.policy.decide(frame, state)
                if ep.a_id != FORWARD and ep.trigger < 0:
                    ep.trigger = t
                label = ACTION_NAMES[ep.a_id]
            # compose a video frame
            top = background.copy()
            for i in range(1, len(path)):
                cv2.line(top, to_px(*path[i - 1][:2]), to_px(*path[i][:2]), (60, 170, 60) if args.policy == "wm" else (30, 140, 245), 2, cv2.LINE_AA)
            for c in crashes:
                u, w = to_px(*c[:2])
                cv2.drawMarker(top, (u, w), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            u, w = to_px(pos[0], pos[1])
            cv2.circle(top, (u, w), 5, (0, 0, 0), -1, cv2.LINE_AA)
            right = cv2.resize(frame[:, :, ::-1], (TOP, TOP), interpolation=cv2.INTER_NEAREST)
            cv2.putText(right, f"t={t:3d}  {label}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(right, f"t={t:3d}  {label}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(right, f"min clear {ep.min_clear:.2f} m  crashes {len(crashes)}", (8, TOP - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(right, f"min clear {ep.min_clear:.2f} m  crashes {len(crashes)}", (8, TOP - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            head = f"{os.path.basename(args.model)} | {args.policy} | hyperdrone/{args.arena_style} | seed {args.seed}"
            canvas = np.concatenate([top, right], 1)
            cv2.putText(canvas, head, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, head, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            frames_out.append(canvas)
        if ep.active:
            v[0] = (ep.speed * ACTION_VECS[ep.a_id])[:3]
            exec_ids[0] = ep.a_id
        else:
            v[0], exec_ids[0] = 0.0, -2
        flight.step(v, exec_ids)
        pos = flight.position()[0]
        path.append(pos.copy())
        if ep.active:
            d = float(np.linalg.norm(ep.centres - pos[:2], axis=1).min())
            ep.min_clear = min(ep.min_clear, d)
            if d < COLLISION_R and not in_crash:
                crashes.append(pos.copy())
            in_crash = d < COLLISION_R
            if pos[0] >= GOAL_X:
                ep.reached, ep.steps, ep.active = True, t + 1, False
        if not ep.active and t % DECIDE_EVERY == DECIDE_EVERY - 1:
            break
    for _ in range(12):  # hold the last frame one second
        frames_out.append(frames_out[-1])
    mp4 = os.path.join(OUT_DIR, f"hd_{args.tag}.mp4")
    writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 12, (2 * TOP, TOP))
    for f in frames_out:
        writer.write(f)
    writer.release()
    try:
        import imageio

        gif = os.path.join(OUT_DIR, f"hd_{args.tag}.gif")
        small = [cv2.resize(f[:, :, ::-1], (TOP, TOP // 2), interpolation=cv2.INTER_AREA) for f in frames_out[::2]]
        imageio.mimsave(gif, small, duration=1 / 6, loop=0)
    except Exception as error:  # noqa: BLE001
        gif = f"(gif skipped: {error})"
    print(f"HD-VIDEO OK: {args.tag}: reached={ep.reached} steps={ep.steps} min_clear={ep.min_clear:.2f} m crashes={len(crashes)} "
          f"trigger={ep.trigger} -> {mp4} {gif}")


def main_rooms(args) -> None:
    """One course inside a ProcTHOR house: occupancy map + path on the left, onboard view right."""
    import cv2
    import hyperlux
    from hyperlux.camera import RGB, CameraSpec
    from wm_rooms import GRID, DistanceField, house_references, load_house, probe_renderer

    enc, pred, cheads, nhead, meta = load_model(args.model, device="cpu")
    policy = WMPolicy(enc, pred, cheads, meta, speed=args.speed) if args.policy == "wm" else ReactivePolicy(enc, nhead)
    house_id, ref = house_references(1, first=args.house)[0]
    house = load_house(ref, fidelity="medium")
    rig = _jit_fast(probe_renderer)
    rig.init(house)
    field = DistanceField(house, rig, rig.scene_bounds, verbose=False)
    del rig
    ep = RoomEpisode(args.seed, policy, args.speed, field, house_id)
    flight = Flight(CHUNK, "crazyflie", 0, False, "body")
    cams = (CameraSpec("front", (RGB("rgba8"),)),)
    onboard = _jit_fast(lambda: hyperlux.HyperLux(width=IMG_RES, height=IMG_RES, batch_size=CHUNK, cameras=cams, fidelity="medium"))
    onboard.init(house)

    # map window: 8 m x 8 m around the start; occupied black, outdoors grey, free shaded by clearance
    half = 4.0
    px_per_m = TOP / (2 * half)

    def to_map(x, y):
        return int(round((x - (ep.start_xy[0] - half)) * px_per_m)), int(round((ep.start_xy[1] + half - y) * px_per_m))

    cell_img = np.full((field.nx, field.ny, 3), 255, dtype=np.uint8)  # indexed [ix, iy]
    free = ~field.occupied
    shade = (255 - np.clip(field.field / 2.0, 0, 1) * 150).astype(np.uint8)
    for ch in range(3):
        cell_img[..., ch] = np.where(free, shade, 0)
    cell_img[~field.indoor & ~free] = 128
    us, vs = np.meshgrid(np.arange(TOP), np.arange(TOP))
    xs = ep.start_xy[0] - half + us / px_per_m
    ys = ep.start_xy[1] + half - vs / px_per_m
    ix = np.floor((xs - field.x0) / GRID).astype(int)
    iy = np.floor((ys - field.y0) / GRID).astype(int)
    ok = (ix >= 0) & (ix < field.nx) & (iy >= 0) & (iy < field.ny)
    background = np.full((TOP, TOP, 3), 128, dtype=np.uint8)
    background[ok] = cell_img[ix[ok], iy[ok]]
    background = np.ascontiguousarray(background[:, :, ::-1])  # BGR for cv2
    fwd = np.array([math.cos(ep.yaw), math.sin(ep.yaw)])
    cv2.circle(background, to_map(*ep.start_xy), 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.arrowedLine(background, to_map(*ep.start_xy), to_map(*(ep.start_xy + 0.6 * fwd)), (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.4)

    starts = np.zeros((CHUNK, 3), dtype=np.float32)
    starts[:, 0] = 1e4 + np.arange(CHUNK) * 10.0
    starts[0, :2], starts[0, 2] = ep.start_xy, START[2]
    yaws = np.zeros(CHUNK, dtype=np.float32)
    yaws[0] = ep.yaw
    flight.reset(starts, yaws)
    rot = _rot(yaws)[0]
    exec_ids = np.full(CHUNK, -3, dtype=np.int64)
    v = np.zeros((CHUNK, 3), dtype=np.float32)
    state = np.zeros(20)
    path, crashes, frames_out = [], [], []
    label, in_crash = "forward", False
    for t in range(TMAX):
        pos = flight.position()[0]
        local = rot.T @ (pos[:2] - ep.start_xy)
        if t % DECIDE_EVERY == 0:
            onboard.set_cameras(flight.camera_bases(DEFAULT_FOV))
            onboard.render()
            frame = np.ascontiguousarray(onboard.outputs[RGB("rgba8")].read()[0, :, :, :3])
            if ep.active:
                state[0:2], state[2] = local, pos[2]
                if isinstance(ep.policy, ReactivePolicy):
                    ep.policy.pillars = [rot.T @ (field.threat_xy(pos[:2]) - ep.start_xy)]
                ep.a_id = ep.policy.decide(frame, state)
                if ep.a_id != FORWARD and ep.trigger < 0:
                    ep.trigger = t
                label = ACTION_NAMES[ep.a_id]
            top = background.copy()
            for i in range(1, len(path)):
                cv2.line(top, to_map(*path[i - 1][:2]), to_map(*path[i][:2]), (60, 170, 60) if args.policy == "wm" else (30, 140, 245), 2, cv2.LINE_AA)
            for c in crashes:
                cv2.drawMarker(top, to_map(*c[:2]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.circle(top, to_map(pos[0], pos[1]), 5, (0, 0, 0), -1, cv2.LINE_AA)
            right = cv2.resize(frame[:, :, ::-1], (TOP, TOP), interpolation=cv2.INTER_NEAREST)
            for text, y, size in ((f"t={t:3d}  {label}", 46, 0.6), (f"{t / 48:.1f} s  min clear {ep.min_clear:.2f} m" + ("  CONTACT" if crashes else ""), TOP - 10, 0.5)):
                cv2.putText(right, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(right, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
            head = f"{os.path.basename(args.model)} | {args.policy} | hyperdrone/{house_id} | seed {args.seed}"
            canvas = np.concatenate([top, right], 1)
            cv2.putText(canvas, head, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, head, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            frames_out.append(canvas)
        if ep.active:
            vl = ep.speed * ACTION_VECS[ep.a_id]
            v[0, :2], v[0, 2] = rot @ vl[:2], vl[2]
            exec_ids[0] = ep.a_id
        else:
            v[0], exec_ids[0] = 0.0, -2
        flight.step(v, exec_ids)
        pos = flight.position()[0]
        path.append(pos.copy())
        if ep.active:
            d = float(field.clearance(pos[:2])) + ep.offset_r
            ep.min_clear = min(ep.min_clear, d)
            if d < COLLISION_R:  # first contact ends the run, as in the endurance test
                crashes.append(pos.copy())
                ep.contact_t, ep.steps, ep.active = t + 1, t + 1, False
        if not ep.active and t % DECIDE_EVERY == DECIDE_EVERY - 1:
            break
    for _ in range(12):
        frames_out.append(frames_out[-1])
    mp4 = os.path.join(OUT_DIR, f"hd_{args.tag}.mp4")
    writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 12, (2 * TOP, TOP))
    for f in frames_out:
        writer.write(f)
    writer.release()
    try:
        import imageio

        gif = os.path.join(OUT_DIR, f"hd_{args.tag}.gif")
        small = [cv2.resize(f[:, :, ::-1], (TOP, TOP // 2), interpolation=cv2.INTER_AREA) for f in frames_out[::2]]
        imageio.mimsave(gif, small, duration=1 / 6, loop=0)
    except Exception as error:  # noqa: BLE001
        gif = f"(gif skipped: {error})"
    print(f"HD-VIDEO OK: {args.tag}: house {house_id} survived={ep.steps / 48:.1f} s contact={'yes' if crashes else 'no'} min_clear={ep.min_clear:.2f} m "
          f"trigger={ep.trigger} -> {mp4} {gif}")


if __name__ == "__main__":
    main()
