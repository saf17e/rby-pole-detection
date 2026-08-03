"""
Wall-based metric depth pipeline using Depth Anything V2 (ViT-L).

Edit the CONFIG block below, then just run:
    python wall_depth_pipeline.py

Install once:
    pip install torch torchvision transformers pillow opencv-python scipy numpy
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth.depth_anything_v2.dpt import DepthAnythingV2


# ==========================================================================
# CONFIG -- edit these and run
# ==========================================================================
IMAGE_PATH = "auv-front-2025-2/train/Front_dataset3_Blue_0074_jpg.rf.3a866acbcce0a593b43215facc9e4431.jpg"          # input image
WALL_HEIGHT_M = 2.0               # real-world wall height in meters
VFOV_DEG = 55.0                   # camera VERTICAL field of view, degrees
MODEL_TYPE = "vitl"
CHECKPOINT_PATH = "checkpoints/depth_anything_v2_vitl.pth"
INPUT_SIZE = 518

MANUAL_CLICK = False              # True -> click wall top/bottom yourself
                                   # instead of auto-detecting
RESCALE_FULL_MAP = False          # True -> also save a metric-rescaled
                                   # depth map for the whole frame

OUT_PATH = "wall_depth_result1.png"

# strip used for auto-detection (fraction of image width, left edge by
# default -- move this if that part of your frame has ropes/buckets/etc.)
STRIP_X_FRAC = (0.05, 0.25)
TOP_MARGIN_FRAC = 0.03            # ignore this much from the very top (glare)
BOTTOM_MARGIN_FRAC = 0.10         # ignore this much from the very bottom (props)
PEAK_PROMINENCE_FRAC = 0.01       # threshold fraction for peak prominence
MIN_PIXEL_HEIGHT = 50           # min pixel height between peaks (None = dynamic)
# ==========================================================================


def run_depth_anything(image_bgr, model_type, checkpoint_path, input_size):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")

    depth_anything = DepthAnythingV2(**model_configs[model_type])
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    depth_anything = depth_anything.to(device).eval()

    depth = depth_anything.infer_image(image_bgr, input_size)
    return depth.astype(np.float32)


def detect_wall_boundaries(depth_map):
    h, w = depth_map.shape

    x0, x1 = int(w * STRIP_X_FRAC[0]), int(w * STRIP_X_FRAC[1])
    strip = depth_map[:, x0:x1]

    sobel_y = cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=5)
    row_energy = np.mean(np.abs(sobel_y), axis=1)
    row_energy = np.convolve(row_energy, np.ones(9) / 9, mode="same")

    lo = int(h * TOP_MARGIN_FRAC)
    hi = int(h * (1 - BOTTOM_MARGIN_FRAC))
    mask = np.zeros_like(row_energy)
    mask[lo:hi] = row_energy[lo:hi]

    min_dist = MIN_PIXEL_HEIGHT if MIN_PIXEL_HEIGHT is not None else max(20, h // 20)
    peaks, _ = find_peaks(mask, distance=min_dist,
                           prominence=mask.max() * PEAK_PROMINENCE_FRAC)

    if len(peaks) < 2:
        raise RuntimeError("couldn't confidently find two boundaries")

    top2 = sorted(peaks[np.argsort(row_energy[peaks])[-2:]])
    return int(top2[0]), int(top2[1])


def manual_wall_boundaries(image_bgr):
    import matplotlib.pyplot as plt

    clicked = []
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Click WALL TOP, then WALL BOTTOM, then close window")

    def onclick(event):
        if event.ydata is not None:
            clicked.append(int(event.ydata))
            ax.axhline(event.ydata, color="red")
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(clicked) < 2:
        raise RuntimeError("need two clicks: wall top and wall bottom")
    return min(clicked[:2]), max(clicked[:2])


def depth_from_wall_height(pixel_height, image_height_px, real_wall_height_m, vfov_deg):
    vfov_rad = np.deg2rad(vfov_deg)
    return (real_wall_height_m * image_height_px) / (
        2.0 * pixel_height * np.tan(vfov_rad / 2.0)
    )


def rescale_depth_map(depth_map, wall_top, wall_bottom, metric_depth_m):
    wall_relative_depth = np.median(depth_map[wall_top:wall_bottom, :])
    if wall_relative_depth == 0:
        raise RuntimeError("wall region has zero relative depth, can't rescale")
    scale = metric_depth_m / wall_relative_depth
    return depth_map * scale


def main():
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        sys.exit(f"Could not read image: {IMAGE_PATH}")
    h, w = image_bgr.shape[:2]

    print("[1/4] Running Depth Anything V2 ViT-L ...")
    depth_map = run_depth_anything(image_bgr, MODEL_TYPE, CHECKPOINT_PATH, INPUT_SIZE)

    print("[2/4] Finding wall boundaries ...")
    if MANUAL_CLICK:
        wall_top, wall_bottom = manual_wall_boundaries(image_bgr)
    else:
        try:
            wall_top, wall_bottom = detect_wall_boundaries(depth_map)
        except RuntimeError as e:
            print(f"  auto-detect failed ({e}); falling back to manual clicking.")
            wall_top, wall_bottom = manual_wall_boundaries(image_bgr)

    pixel_height = wall_bottom - wall_top
    print(f"  wall_top={wall_top}px  wall_bottom={wall_bottom}px  "
          f"pixel_height={pixel_height}px")

    print("[3/4] Computing metric depth ...")
    metric_depth = depth_from_wall_height(pixel_height, h, WALL_HEIGHT_M, VFOV_DEG)
    print(f"  --> estimated distance to wall: {metric_depth:.3f} m")

    print("[4/4] Saving visualization ...")
    vis = image_bgr.copy()
    cv2.line(vis, (0, wall_top), (w, wall_top), (0, 255, 0), 2)
    cv2.line(vis, (0, wall_bottom), (w, wall_bottom), (0, 0, 255), 2)
    cv2.putText(vis, f"depth ~ {metric_depth:.2f} m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('Spectral_r')
    normalized_depth = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_vis = (cmap(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
    combined = np.hstack([vis, depth_vis])
    cv2.imwrite(OUT_PATH, combined)
    print(f"  saved to {OUT_PATH}")

    if RESCALE_FULL_MAP:
        metric_map = rescale_depth_map(depth_map, wall_top, wall_bottom, metric_depth)
        out_npy = OUT_PATH.rsplit(".", 1)[0] + "_metric_depth.npy"
        np.save(out_npy, metric_map)
        print(f"  saved metric depth map to {out_npy}")


if __name__ == "__main__":
    main()