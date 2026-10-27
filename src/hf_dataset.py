"""
Package the world-model datasets for the Hugging Face Hub, and push them
========================================================================
The training code reads sequence-format npz files, so the Hub repo keeps exactly those
(plus previews and a dataset card) — no conversion that the trainer would have to undo:

    hf_dataset/
      README.md                 dataset card (YAML header + schema + how it was generated)
      data/<name>.npz           one file per split/variant (pillars, rooms, mixed, ...)
      previews/<name>.png       the 8x8 frame mosaics
      manifest.json             per-file shapes, label rates, generator arguments, sha256

Build the folder from npz files, then push it (needs `huggingface-cli login` or HF_TOKEN):

  python src/hf_dataset.py build src/output/wm_dataset_mixed.npz src/output/wm_dataset_hyperdrone_tello40.npz \\
      --out hf_dataset --metrics src/output/hf_metrics.json
  python src/hf_dataset.py push --folder hf_dataset --repo <user>/drone-world-model-sim --private

Load it back anywhere:

  from huggingface_hub import hf_hub_download
  path = hf_hub_download("<user>/drone-world-model-sim", "data/wm_dataset_mixed.npz", repo_type="dataset")
  python src/train_world_model.py --data $path
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import ACTION_NAMES, DANGER_R, HORIZONS, as_pairs  # noqa: E402


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(path: str) -> dict:
    """Shapes, provenance keys and label statistics of one npz."""
    blob = np.load(path)
    data = {k: blob[k] for k in blob.files}
    R, L = data["frames"].shape[:2]
    rates = {int(k): round(float(as_pairs(data, k)["c"].mean()), 3) for k in HORIZONS}
    scene_type = data.get("scene_type", np.zeros(R, dtype=np.int8))
    houses = sorted(set(data["scene_id"].tolist()) - {"pillars"}) if "scene_id" in data else []
    return {
        "file": os.path.basename(path),
        "sha256": _sha256(path),
        "bytes": os.path.getsize(path),
        "rollouts": int(R),
        "steps": int(L),
        "frames": int(R * L),
        "image": list(data["frames"].shape[2:]),
        "pillar_rollouts": int((scene_type == 0).sum()),
        "room_rollouts": int((scene_type == 1).sum()),
        "houses": houses,
        "source": str(data.get("source", "pybullet")),
        "camera": str(data.get("camera", "body")),
        "fov_deg": float(data.get("fov_deg", 60.0)),
        "randomized": bool(int(data.get("randomized", 0))),
        "drone_model": str(data.get("drone_model", "cf2x-pybullet")),
        "positive_rate_by_horizon": rates,
        "danger_r": float(data.get("danger_r", DANGER_R)),
        "actions": list(ACTION_NAMES),
        "keys": {k: [list(v.shape), str(v.dtype)] for k, v in data.items()},
    }


CARD = """---
license: mit
pretty_name: Drone world-model intervention dataset (hyperdrone)
tags:
  - robotics
  - drone
  - world-model
  - simulation
  - hyperdrone
  - obstacle-avoidance
size_categories:
  - 10K<n<100K
---

# Drone world-model intervention dataset

Sequence-format training data for the nano V-JEPA-style world model of
[nanodrone-ai lesson 29](https://github.com/csinghans/nanodrone-ai) (`drone_ws/src/train_world_model.py`),
generated with the [hyperdrone](https://pypi.org/project/hyperdrone/) simulator and the hyperlux
raytracer instead of gym-pybullet-drones. Each rollout is a short controlled experiment: a crazyflie-class
quadrotor (L2F dynamics, RAPTOR position controller) cruises forward, then holds a chain of ~1 s
commands drawn from `forward / slow / veer_left / veer_right / climb / hover`, while a 64x64 forward
camera records what it sees and the geometry provides the "dangerously close within k steps" labels.

Two kinds of scene share one schema:

* **pillars** — the original course: a checker-floored arena with 2-3 cylinder pillars (0.18 m radius),
  one of them in the flight path on even rollouts. Same random layouts as the pybullet generator for a
  given seed.
* **rooms** — furnished ProcTHOR houses from hyperlux's pinned catalog. Obstacle distances come from a
  0.1 m occupancy map probed at flight height (walls and furniture), with `dists = clearance + 0.18`
  so the 0.7 m / 0.35 m danger rings keep their surface meaning.

The camera is a square pinhole; the `*_tello40*` files use a 40 deg, virtually stabilised view that a
ChArUco-calibrated DJI Tello (960x720, HFOV ~56 deg) can reproduce exactly through an undistort-remap,
while 60 deg files reproduce the pybullet camera.

## Files

{files_table}

## Schema (every npz)

| key | shape | meaning |
|---|---|---|
| `frames` | (R, L, 64, 64, 3) uint8 | RGB from the forward camera |
| `actions` | (R, L, 4) float32 | commanded (vx, vy, vz, yaw-rate) m/s in the rollout frame (x forward, y left), scaled by the rollout's cruise factor |
| `act_id`, `seg` | (R, L) int16 | command index and held-segment id (0 = approach) |
| `dists` | (R, L) float32 | centre-equivalent distance to the nearest obstacle (m) |
| `pos`, `quat` | (R, L, 3) / (R, L, 4) | drone position in the rollout frame and body attitude (wxyz) |
| `pillars` | (R, 3, 2) float32 | pillar centres (NaN padded; all NaN for room rollouts) |
| `in_path`, `speed` | (R,) | obstacle-in-path flag, cruise-speed factor 0.75-2.0 |
| `cf_labels`, `cf_visible` | (R, L, 6, 4, 2) / (R, L, 6) uint8 | counterfactual oracle: would holding command i get within ring q inside horizon j; whether that threat is inside the camera FOV |
| `scene_type`, `scene_id`, `yaw`, `start_world` | (R,) | 0 pillars / 1 room, house id, heading and world start of the rollout frame |
| `fov_deg`, `fov_half_deg`, `camera`, `horizons`, `a_norm`, `danger_r`, `source`, `drone_model`, `sim_dt` | scalars | camera and provenance |

Horizons are 4/8/16/32 control steps at 48 Hz (83-667 ms). Labels are planar: `climb` changes the view, not
the danger label; obstacles have no contact physics (the drone flies through them, as in the original course).

## How it was made

```bash
# conda env with hyperdrone==0.1.8 (WebGPU/llvmpipe backend is enough: 64x64 renders at ~2000 fps)
{commands}
```

{metrics}

## Honest limits

Simulation only: no sim here is photorealistic, the crazyflie dynamics are not a Tello's, the room
occupancy is a thin band at one altitude (low tables are flown over, high lamps ignored), and the
pybullet-era closed-loop evaluation scripts still need the pybullet environment. Validate on the
real vehicle before trusting any number here.
"""


def build(paths: list, out: str, metrics_path: str = None, commands: list = None) -> None:
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    os.makedirs(os.path.join(out, "previews"), exist_ok=True)
    manifest = []
    for path in paths:
        info = describe(path)
        shutil.copy2(path, os.path.join(out, "data", info["file"]))
        preview = path.replace(".npz", "_preview.png")
        if os.path.exists(preview):
            shutil.copy2(preview, os.path.join(out, "previews", os.path.basename(preview)))
            info["preview"] = f"previews/{os.path.basename(preview)}"
        manifest.append(info)
        print(f"  {info['file']}: {info['rollouts']} rollouts ({info['pillar_rollouts']} pillars + {info['room_rollouts']} rooms), "
              f"{info['fov_deg']:g} deg {info['camera']}, {info['bytes'] / 1e6:.1f} MB")
    rows = ["| file | rollouts | scenes | camera | positive rate k=4/8/16/32 | size |", "|---|---|---|---|---|---|"]
    for m in manifest:
        scenes = " + ".join(s for s, n in (("pillars", m["pillar_rollouts"]), (f"rooms ({len(m['houses'])} houses)", m["room_rollouts"])) if n)
        pr = "/".join(f"{m['positive_rate_by_horizon'][k]:.2f}" for k in sorted(m["positive_rate_by_horizon"]))
        cam = f"{m['fov_deg']:g} deg {m['camera']}" + (" randomized" if m["randomized"] else "")
        rows.append(f"| `data/{m['file']}` | {m['rollouts']} x {m['steps']} | {scenes} | {cam} | {pr} | {m['bytes'] / 1e6:.1f} MB |")
    metrics = ""
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = f.read()
    card = CARD.format(files_table="\n".join(rows), commands="\n".join(commands or ["# see manifest.json"]), metrics=metrics)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"files": manifest, "commands": commands or []}, f, indent=2)
    print(f"built {out}: {len(manifest)} files, card README.md, manifest.json")


def push(folder: str, repo: str, private: bool, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()["name"]
    url = api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo, repo_type="dataset", commit_message=message)
    print(f"pushed {folder} as {who} -> {url}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="collect npz files + previews + card into a folder")
    b.add_argument("npz", nargs="+")
    b.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "hf_dataset"))
    b.add_argument("--metrics", default=None, help="markdown file appended to the card (training results)")
    b.add_argument("--command", action="append", default=[], help="generation command(s) recorded in the card")
    p = sub.add_parser("push", help="upload the folder to a Hub dataset repo")
    p.add_argument("--folder", default=os.path.join(os.path.dirname(HERE), "hf_dataset"))
    p.add_argument("--repo", required=True, help="<user-or-org>/<name>")
    p.add_argument("--private", action="store_true")
    p.add_argument("--message", default="Add hyperdrone world-model datasets")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.npz, args.out, args.metrics, args.command)
    else:
        push(args.folder, args.repo, args.private, args.message)


if __name__ == "__main__":
    main()
