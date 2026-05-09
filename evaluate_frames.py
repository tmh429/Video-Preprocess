import json
import os
from pathlib import Path

import cv2
import numpy as np

from utils.clip_iqa_score import evaluate_clip_iqa_batch
from utils.plot_clip_scores import plot_score_lines

# 配置参数
width = 1920
height = 1080
frames_dir = "frames"
encoded_frames_dir = "encoded_frames"
bitrates = [1000, 2000, 3000, 4000]
alpha_candidates = np.linspace(-2.0, 3.0, 11).tolist()
results_json = "frame_usm_clip_iqa_results.json"
plots_dir = "frame_plots"


def save_results(all_results):
    """保存结果到 JSON"""
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def evaluate_all_frames():
    """评估所有帧的编码结果，每批处理完立即保存"""
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    frame_files = list(Path(frames_dir).glob("*.yuv"))
    if not frame_files:
        raise FileNotFoundError(f"未找到帧文件：{frames_dir}/*.yuv")

    all_results = {}
    if os.path.isfile(results_json):
        with open(results_json, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"已加载已有结果: {len(all_results)} 条")

    # 按帧分组收集需要评估的任务
    frame_jobs = {}  # frame_name -> [(encoded_path, alpha, bitrate)]
    for frame_file in frame_files:
        frame_name = frame_file.stem
        frame_key = str(frame_file)
        if frame_key in all_results:
            continue
        jobs = []
        for alpha in alpha_candidates:
            alpha_tag = f"{alpha:.2f}".replace("-", "n").replace(".", "p")
            for bitrate in bitrates:
                encoded_path = os.path.join(encoded_frames_dir, f"{frame_name}_usm_{alpha_tag}_{bitrate}k.mp4")
                if os.path.isfile(encoded_path):
                    jobs.append((encoded_path, alpha, bitrate))
        if jobs:
            frame_jobs[frame_name] = jobs

    if not frame_jobs:
        print("没有需要评估的帧")
        return all_results

    total_frames = len(frame_jobs)
    print(f"待评估帧数: {total_frames}")

    # 逐帧评估，每帧处理完立即保存
    for frame_idx, (frame_name, jobs) in enumerate(frame_jobs.items()):
        frame_key = str(Path(frames_dir) / f"{frame_name}.yuv")
        alpha_scores = {a: {} for a in alpha_candidates}

        for encoded_path, alpha, bitrate in jobs:
            try:
                video_paths = [(encoded_path, 1)]
                score_dicts = evaluate_clip_iqa_batch(video_paths, width, height, batch_size=1)
                alpha_scores[alpha][bitrate] = score_dicts[0]
            except Exception as e:
                print(f"  跳过损坏文件: {encoded_path} ({e})")
                alpha_scores[alpha][bitrate] = {"quality": 0.0}

        # 各码率下的最优 alpha
        best_per_bitrate = {}
        for br in bitrates:
            best_a = max(
                alpha_candidates,
                key=lambda a: alpha_scores.get(a, {}).get(br, {}).get("quality", 0)
            )
            best_per_bitrate[br] = float(best_a)

        if all(v == 0.0 for v in best_per_bitrate.values()):
            print(f"  警告: 帧 {frame_name} 没有有效结果，跳过")
            continue

        all_results[frame_key] = {
            "alpha_scores": {
                f"{alpha:.2f}": {
                    str(bitrate): score for bitrate, score in score_map.items()
                }
                for alpha, score_map in alpha_scores.items()
            },
            "best_alpha_per_bitrate": {str(br): a for br, a in best_per_bitrate.items()},
        }

        # 画图
        quality_scores = {
            alpha: {br: s["quality"] if isinstance(s, dict) else s
                    for br, s in score_map.items()}
            for alpha, score_map in alpha_scores.items()
        }
        plot_path = os.path.join(plots_dir, f"{frame_name}_clip_iqa.png")
        plot_score_lines(frame_name, quality_scores, plot_path)

        # 每帧处理完立即保存
        save_results(all_results)
        br_str = ", ".join(f"{br}k={best_per_bitrate[br]:.1f}" for br in bitrates)
        print(f"[{frame_idx + 1}/{total_frames}] 帧 {frame_name}: 各码率最优 alpha -> {br_str}")

    print(f"全部完成，结果已保存到 {results_json}")
    return all_results


if __name__ == "__main__":
    evaluate_all_frames()
