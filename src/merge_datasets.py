"""
Concatenate world-model datasets (rollout axis) into one training set
=====================================================================
Lets the trainer see pybullet courses and hyperdrone pillars/rooms together, e.g. the original
`wm_dataset.npz` plus a hyperdrone mixed set as extra data. Parts must share the frame size,
rollout length and camera (fov/body-vs-level); older files without the hyperdrone extras get
them filled in (counterfactual labels from their pillar list, identity attitude, zero yaw).

  python src/merge_datasets.py src/output/wm_dataset.npz src/output/wm_dataset_mixed_big_60body.npz \\
      --out src/output/wm_dataset_pybullet_plus_mixed_big.npz
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import FOV_HALF_DEG, START, counterfactual_labels  # noqa: E402

PER_ROLLOUT = ("frames", "actions", "act_id", "seg", "dists", "pos", "pillars", "in_path", "speed", "quat", "scene_type", "scene_id", "yaw", "start_world", "cf_labels", "cf_visible")
OPTIONAL_PER_ROLLOUT = ("depth",)


def complete(data: dict, name: str) -> dict:
    """Fill the hyperdrone extras into a dataset that lacks them (a pybullet file)."""
    R, L = data["frames"].shape[:2]
    out = dict(data)
    out.setdefault("fov_deg", np.float32(2 * FOV_HALF_DEG + 4))
    out.setdefault("fov_half_deg", np.float32(FOV_HALF_DEG))
    out.setdefault("camera", np.array("body"))
    out.setdefault("source", np.array("pybullet"))
    if "quat" not in out:
        out["quat"] = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (R, L, 1))
    out.setdefault("scene_type", np.zeros(R, dtype=np.int8))
    if "scene_id" not in out:
        out["scene_id"] = np.array([f"{name}-pillars"] * R)
    out.setdefault("yaw", np.zeros(R, dtype=np.float32))
    out.setdefault("start_world", np.tile(START.astype(np.float32), (R, 1)))
    if "cf_labels" not in out:
        sub = {k: out[k] for k in ("frames", "pillars", "pos", "speed")}
        sub["fov_half_deg"] = out["fov_half_deg"]
        out["cf_labels"], out["cf_visible"] = counterfactual_labels(sub)
    return out


def merge(paths: list) -> dict:
    parts = []
    for path in paths:
        blob = np.load(path)
        parts.append(complete({k: blob[k] for k in blob.files}, os.path.splitext(os.path.basename(path))[0]))
    first = parts[0]
    for part, path in zip(parts[1:], paths[1:]):
        for key in ("frames", "actions"):
            if part[key].shape[1:] != first[key].shape[1:]:
                raise ValueError(f"{path}: {key} shape {part[key].shape[1:]} != {first[key].shape[1:]}")
        for key in ("fov_deg", "camera", "danger_r"):
            if str(part[key]) != str(first[key]):
                raise ValueError(f"{path}: {key} {part[key]} != {first[key]} — parts must share the camera and rings")
    merged = {}
    for key in PER_ROLLOUT + OPTIONAL_PER_ROLLOUT:
        if key in OPTIONAL_PER_ROLLOUT and not all(key in p for p in parts):
            continue
        merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    for key, value in first.items():
        if key not in merged:
            merged[key] = value
    merged["source"] = np.array("+".join(sorted({str(p["source"]) for p in parts})))
    merged["randomized"] = np.uint8(any(int(p.get("randomized", 0)) for p in parts))
    merged["parts"] = np.array([os.path.basename(p) for p in paths])
    merged["part_rollouts"] = np.array([int(p["frames"].shape[0]) for p in parts], dtype=np.int32)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = merge(args.npz)
    np.savez_compressed(args.out, **data)
    kinds = {name: int((data["scene_type"] == code).sum()) for code, name in ((0, "pillars"), (1, "rooms"))}
    print(
        f"MERGED OK: {data['frames'].shape[0]} rollouts from {len(args.npz)} files "
        f"({' + '.join(f'{n}:{r}' for n, r in zip(data['parts'], data['part_rollouts']))}); "
        f"{kinds['pillars']} pillar + {kinds['rooms']} room rollouts, camera {float(data['fov_deg']):g} deg {data['camera']}, saved {args.out}"
    )


if __name__ == "__main__":
    main()
