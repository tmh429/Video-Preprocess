import os

import cv2
import numpy as np
import pyiqa
import torch

# 全局缓存
_metric = None
_device = None


def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_metric():
    global _metric
    if _metric is None:
        _metric = pyiqa.create_metric('clipiqa', device=get_device())
    return _metric


def sample_compressed_frames(video_path, sample_every=1, max_samples=1):
    """从压缩视频中采样帧，返回 BGR 帧列表和 fps"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开压缩视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    samples = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            samples.append(frame)
            if len(samples) >= max_samples:
                break
        idx += 1
    cap.release()
    return samples, fps


def frames_to_tensor(frames_bgr):
    """BGR 帧列表 → (N, 3, H, W) float32 tensor, RGB, 0~1"""
    tensors = []
    for frame in frames_bgr:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    return torch.stack(tensors)


def compute_clipiqa_batch(frames_bgr, batch_size=16):
    """批量评估多帧的 CLIP-IQA 得分，返回分数列表"""
    metric = get_metric()
    device = get_device()
    tensor = frames_to_tensor(frames_bgr)

    scores = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i + batch_size].to(device)
            batch_scores = metric(batch)
            scores.extend(batch_scores.cpu().tolist())

    return scores


def evaluate_clip_iqa_batch(video_frame_pairs, width, height, device=None, batch_size=16,
                             prompts=None):
    """
    批量评估多个编码视频的 CLIP-IQA 得分。

    参数:
        video_frame_pairs: list of (encoded_video_path, max_samples) 元组
        batch_size: 推理批大小

    返回:
        list[dict[str, float]]，每个视频一个 {"quality": 均分} 字典
    """
    all_frames = []
    video_indices = []

    for idx, (video_path, max_samples) in enumerate(video_frame_pairs):
        frames, _ = sample_compressed_frames(video_path, sample_every=1, max_samples=max_samples)
        for f in frames:
            all_frames.append(f)
            video_indices.append(idx)

    if not all_frames:
        return [{"quality": 0.0}] * len(video_frame_pairs)

    all_scores = compute_clipiqa_batch(all_frames, batch_size=batch_size)

    # 按视频聚合
    n_videos = len(video_frame_pairs)
    video_scores = [[] for _ in range(n_videos)]
    for frame_idx, vid_idx in enumerate(video_indices):
        video_scores[vid_idx].append(all_scores[frame_idx])

    return [{"quality": float(np.mean(s)) if s else 0.0} for s in video_scores]
