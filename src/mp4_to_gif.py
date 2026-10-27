"""Turn short mp4 clips into README-embeddable GIFs: python src/mp4_to_gif.py file.mp4 [...] (--width 480 --fps 6)."""
import argparse, os
import cv2, imageio

ap = argparse.ArgumentParser()
ap.add_argument("mp4", nargs="+")
ap.add_argument("--width", type=int, default=480)
ap.add_argument("--fps", type=float, default=6.0)
ap.add_argument("--max-seconds", type=float, default=12.0)
args = ap.parse_args()
for path in args.mp4:
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    stride = max(1, int(round(src_fps / args.fps)))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or i / src_fps > args.max_seconds:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (args.width, int(h * args.width / w)), interpolation=cv2.INTER_AREA)
            frames.append(frame[:, :, ::-1])
        i += 1
    cap.release()
    out = os.path.splitext(path)[0] + ".gif"
    imageio.mimsave(out, frames, duration=1.0 / args.fps, loop=0)
    print(f"{out}: {len(frames)} frames, {os.path.getsize(out) / 1e6:.1f} MB")
