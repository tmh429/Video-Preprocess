"""
可视化对比：原图 vs 单个 alpha 编码结果，逐一对比。
每张图：左侧原图中心裁剪，右侧编码结果中心裁剪，附带 CLIP-IQA 分数。
用法：python verify_visual.py
"""
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 配置
FRAMES_DIR = "frames"
ENCODED_DIR = "encoded_frames"
RESULTS_JSON = "frame_usm_clip_iqa_results.json"
OUTPUT_DIR = "verify_plots"
WIDTH = 1920
HEIGHT = 1080
ALPHA_CANDIDATES = np.linspace(-2.0, 3.0, 11).tolist()
BITRATES = [1000, 2000, 3000, 4000]
CROP_W, CROP_H = 640, 360  # 中心裁剪尺寸


def read_yuv_frame(yuv_path, width, height):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    with open(yuv_path, "rb") as f:
        raw = f.read(y_size + uv_size * 2)
    y = np.frombuffer(raw[:y_size], dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(raw[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
    v = np.frombuffer(raw[y_size + uv_size:], dtype=np.uint8).reshape((height // 2, width // 2))
    u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
    v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
    yuv = cv2.merge([y, v_up, u_up])
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)


def decode_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def crop_center(frame):
    h, w = frame.shape[:2]
    x = (w - CROP_W) // 2
    y = (h - CROP_H) // 2
    return frame[y:y + CROP_H, x:x + CROP_W]


def make_comparison(frame_name, bitrate, results):
    """为指定帧和码率，每对 alpha 生成一张原图vs编码的对比图"""
    frame_key = str(Path(FRAMES_DIR) / f"{frame_name}.yuv")
    if frame_key not in results:
        print(f"帧 {frame_name} 不在结果中")
        return

    alpha_scores = results[frame_key]["alpha_scores"]
    best_per_bitrate = results[frame_key].get("best_alpha_per_bitrate", {})
    best_alpha = best_per_bitrate.get(str(bitrate), None)

    original = read_yuv_frame(str(Path(FRAMES_DIR) / f"{frame_name}.yuv"), WIDTH, HEIGHT)
    original_crop = crop_center(original)

    out_dir = os.path.join(OUTPUT_DIR, frame_name)
    os.makedirs(out_dir, exist_ok=True)

    for alpha in ALPHA_CANDIDATES:
        alpha_tag = f"{alpha:.2f}".replace("-", "n").replace(".", "p")
        encoded_path = os.path.join(ENCODED_DIR, f"{frame_name}_usm_{alpha_tag}_{bitrate}k.mp4")
        frame = decode_video_frame(encoded_path)
        if frame is None:
            continue

        score = alpha_scores.get(f"{alpha:.2f}", {}).get(str(bitrate), {}).get("quality", 0)
        frame_crop = crop_center(frame)
        is_best = (best_alpha is not None and abs(alpha - best_alpha) < 0.01)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"Frame: {frame_name} | Bitrate: {bitrate}k", fontsize=12)

        ax1.imshow(cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original (center crop)", fontsize=10)
        ax1.axis("off")

        ax2.imshow(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB))
        title = f"alpha={alpha:.1f}  Quality={score:.3f}"
        if is_best:
            title += "  [BEST]"
        ax2.set_title(title, fontsize=10,
                       fontweight="bold" if is_best else "normal",
                       color="red" if is_best else "black")
        ax2.axis("off")

        plt.tight_layout()
        fname = f"br{bitrate}k_alpha_{alpha_tag}{'_BEST' if is_best else ''}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
        plt.close()

    print(f"对比图已保存到: {out_dir}/")


def main(frame_idx):
    if not os.path.isfile(RESULTS_JSON):
        print(f"结果文件不存在: {RESULTS_JSON}，请先运行 evaluate_frames.py")
        return

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame_names = sorted(set(Path(key).stem for key in results.keys()))
    if not frame_names:
        print("没有可用的结果")
        return

    frame_name = frame_names[frame_idx]
    print(f"帧: {frame_name}")
    for br in BITRATES:
        make_comparison(frame_name, br, results)


if __name__ == "__main__":
    for frame_idx in range(0, 81):
        main(frame_idx)
