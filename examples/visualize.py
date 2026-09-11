"""Draw a YOLO-format bounding box on an image, with a zoomed crop.

Tiny targets (a bird is ~40 px in a 4K frame) are unreadable at full scale,
so this saves BOTH a downscaled full-frame (box drawn) and a zoomed crop
around the box.

Usage:
    python examples/visualize.py <image.jpg> <annot.txt> <out.jpg>
"""
import argparse
import os

import numpy as np
import cv2


def imread(path):
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite(path, img):
    cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)


def read_box(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        parts = f.readline().strip().split()
    cls = parts[0]
    cx, cy, w, h = (float(x) for x in parts[1:5])
    return cls, cx, cy, w, h


def main():
    ap = argparse.ArgumentParser(description="Draw a YOLO box on an image (+ zoomed crop)")
    ap.add_argument("image")
    ap.add_argument("txt")
    ap.add_argument("out")
    ap.add_argument("--zoom", type=int, default=6, help="crop half-size = box * zoom")
    ap.add_argument("--color", default="0,255,0", help="BGR, e.g. 0,255,0")
    args = ap.parse_args()

    img = imread(args.image)
    H, W = img.shape[:2]
    _, cx, cy, w, h = read_box(args.txt)
    x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
    color = tuple(int(v) for v in args.color.split(","))

    # downscaled full frame with the box drawn
    scale = max(1.0, W / 1280.0)
    small = cv2.resize(img, (int(W / scale), int(H / scale)))
    cv2.rectangle(small, (int(x1 / scale), int(y1 / scale)),
                  (int(x2 / scale), int(y2 / scale)), color, 2)
    imwrite(args.out, small)

    # zoomed crop around the box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = max(40, bw * args.zoom // 2)
    pad_y = max(40, bh * args.zoom // 2)
    cx_px, cy_px = (x1 + x2) // 2, (y1 + y2) // 2
    zx1, zy1 = max(0, cx_px - pad_x), max(0, cy_px - pad_y)
    zx2, zy2 = min(W, cx_px + pad_x), min(H, cy_px + pad_y)
    crop = img[zy1:zy2, zx1:zx2]
    cv2.rectangle(crop, (x1 - zx1, y1 - zy1), (x2 - zx1, y2 - zy1), color, 3)

    root, ext = os.path.splitext(args.out)
    imwrite(root + "_crop" + ext, crop)
    print(f"wrote {args.out} and {root}_crop{ext}")


if __name__ == "__main__":
    main()
