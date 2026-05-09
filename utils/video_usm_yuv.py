import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# 直接写死或从环境变量加载 ffmpeg 路径
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"D:\develop\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe")


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace("-", "n").replace(".", "p")


def find_ffmpeg(ffmpeg_cmd: str = FFMPEG_PATH) -> str:
    if os.path.isfile(ffmpeg_cmd):
        return ffmpeg_cmd

    resolved = shutil.which(ffmpeg_cmd)
    if resolved:
        return resolved

    if os.name == "nt":
        alt = ffmpeg_cmd
        if not alt.lower().endswith(".exe"):
            alt = f"{alt}.exe"
        if os.path.isfile(alt):
            return alt
        resolved = shutil.which(alt)
        if resolved:
            return resolved

    raise RuntimeError(
        f"找不到 ffmpeg，可通过设置环境变量 FFMPEG_PATH 指定 ffmpeg.exe 的完整路径。\n"
        f"当前值: {ffmpeg_cmd}"
    )


def read_yuv420_frame_raw(f, width: int, height: int):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    raw = f.read(y_size + uv_size * 2)
    if len(raw) < y_size + uv_size * 2:
        return None

    y = np.frombuffer(raw[0:y_size], dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(raw[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
    v = np.frombuffer(raw[y_size + uv_size:], dtype=np.uint8).reshape((height // 2, width // 2))
    return y, u, v


def yuv420_to_bgr(y, u, v, width: int, height: int):
    u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
    v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
    yuv = cv2.merge([y, v_up, u_up])
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)


def unsharp_mask_y(y, alpha: float, kernel_size: int = 5, sigma: float = 1.0):
    blurred_y = cv2.GaussianBlur(y, (kernel_size, kernel_size), sigma)
    mask = cv2.subtract(y, blurred_y)
    sharpened_y = cv2.addWeighted(y, 1.0, mask, alpha, 0)
    return np.clip(sharpened_y, 0, 255).astype(np.uint8)


def encode_h265_from_yuv(yuv_file: str, alpha: float, width: int, height: int, fps: int, bitrate_kbps: int, output_path: str, ffmpeg_cmd: str = FFMPEG_PATH):
    ffmpeg_path = find_ffmpeg(ffmpeg_cmd)
    cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-b:v",
        f"{bitrate_kbps}k",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(yuv_file, "rb") as f:
        while True:
            frame = read_yuv420_frame_raw(f, width, height)
            if frame is None:
                break
            y, u, v = frame
            sharpened_y = unsharp_mask_y(y, alpha)
            process.stdin.write(sharpened_y.tobytes())
            process.stdin.write(u.tobytes())
            process.stdin.write(v.tobytes())

    process.stdin.close()
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 压缩失败: {output_path}\n" f"stderr:\n{stderr.decode(errors='ignore')}"
        )

    return output_path


def encode_all_alphas_bitrates(yuv_file: str, alphas, bitrates, output_dir: str, width: int, height: int, fps: int, ffmpeg_cmd: str = FFMPEG_PATH):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    video_name = Path(yuv_file).stem
    encoded = {}

    for alpha in alphas:
        alpha_label = alpha_tag(alpha)
        encoded[alpha] = {}
        for bitrate in bitrates:
            output_path = os.path.join(output_dir, f"{video_name}_usm_{alpha_label}_{bitrate}k.mp4")
            if os.path.isfile(output_path):
                print(f"文件已存在，跳过编码: {output_path}")
                encoded[alpha][bitrate] = output_path
            else:
                print(f"编码 {video_name}: alpha={alpha:.2f}, bitrate={bitrate}k -> {output_path}")
                encode_h265_from_yuv(yuv_file, alpha, width, height, fps, bitrate, output_path, ffmpeg_cmd=ffmpeg_cmd)
                encoded[alpha][bitrate] = output_path

    return encoded
