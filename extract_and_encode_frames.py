import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# 配置参数
width = 1920
height = 1080
yuv_pattern = "datasets/*.yuv"
frames_dir = "frames"
output_dir = "encoded_frames"
bitrates = [1000, 2000, 3000, 4000]
alpha_candidates = np.linspace(-2.0, 3.0, 11).tolist()
# ffmpeg 可执行文件路径
ffmpeg_cmd = r"D:\develop\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"


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


def write_yuv420_frame_raw(f, y, u, v):
    f.write(y.tobytes())
    f.write(u.tobytes())
    f.write(v.tobytes())


def extract_frames_from_yuv(yuv_file: str, frames_dir: str, width: int, height: int):
    """将YUV文件的每一帧提取为单独的YUV文件"""
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    video_name = Path(yuv_file).stem

    frame_count = 0
    with open(yuv_file, "rb") as f:
        while True:
            frame = read_yuv420_frame_raw(f, width, height)
            if frame is None:
                break

            frame_path = os.path.join(frames_dir, f"{video_name}_frame_{frame_count:04d}.yuv")
            with open(frame_path, "wb") as frame_f:
                write_yuv420_frame_raw(frame_f, *frame)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"已提取 {frame_count} 帧...")

    print(f"共提取 {frame_count} 帧到 {frames_dir}")
    return frame_count


def unsharp_mask_y(y, alpha: float, kernel_size: int = 5, sigma: float = 1.0):
    blurred_y = cv2.GaussianBlur(y, (kernel_size, kernel_size), sigma)
    mask = cv2.subtract(y, blurred_y)
    sharpened_y = cv2.addWeighted(y, 1.0, mask, alpha, 0)
    return np.clip(sharpened_y, 0, 255).astype(np.uint8)


def find_ffmpeg(ffmpeg_cmd: str = ffmpeg_cmd):
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
    raise RuntimeError(f"找不到 ffmpeg，可通过设置环境变量 FFMPEG_PATH 指定 ffmpeg.exe 的完整路径。\n当前值: {ffmpeg_cmd}")


def encode_single_frame_yuv(yuv_frame_file: str, alpha: float, bitrate_kbps: int, output_path: str):
    """对单帧YUV进行USM锐化后编码为H.265"""
    ffmpeg_path = find_ffmpeg()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 读取原始帧
    with open(yuv_frame_file, "rb") as f:
        y, u, v = read_yuv420_frame_raw(f, width, height)

    # 应用USM
    sharpened_y = unsharp_mask_y(y, alpha)

    # 临时文件存储处理后的帧（使用临时目录）
    temp_yuv = os.path.join(os.path.dirname(output_path), f"temp_{os.path.basename(output_path)}.yuv")
    with open(temp_yuv, "wb") as f:
        write_yuv420_frame_raw(f, sharpened_y, u, v)

    # ffmpeg编码
    cmd = [
        ffmpeg_path,
        "-y",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "yuv420p",
        "-i",
        temp_yuv,
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-b:v",
        f"{bitrate_kbps}k",
        "-maxrate",
        f"{bitrate_kbps}k",
        "-bufsize",
        f"{bitrate_kbps}k",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 编码失败: {output_path}\n{result.stderr}")

    # 清理临时文件
    os.remove(temp_yuv)


def process_all_frames():
    """处理所有帧：提取 -> 锐化 -> 编码"""
    yuv_files = list(Path(".").glob(yuv_pattern))
    if not yuv_files:
        raise FileNotFoundError(f"未找到匹配的 YUV 文件：{yuv_pattern}")

    for yuv_file in yuv_files:
        print(f"处理视频: {yuv_file}")

        # 1. 提取所有帧
        frame_count = extract_frames_from_yuv(str(yuv_file), frames_dir, width, height)

        # 2. 对每帧进行所有alpha和码率的编码
        for frame_idx in range(frame_count):
            frame_name = f"{yuv_file.stem}_frame_{frame_idx:04d}"
            frame_yuv = os.path.join(frames_dir, f"{frame_name}.yuv")

            print(f"处理帧 {frame_idx + 1}/{frame_count}: {frame_name}")

            for alpha in alpha_candidates:
                alpha_tag = f"{alpha:.2f}".replace("-", "n").replace(".", "p")

                for bitrate in bitrates:
                    output_path = os.path.join(output_dir, f"{frame_name}_usm_{alpha_tag}_{bitrate}k.mp4")

                    if os.path.isfile(output_path):
                        print(f"  文件已存在，跳过: {output_path}")
                        continue

                    print(f"  编码 alpha={alpha:.2f}, bitrate={bitrate}k")
                    encode_single_frame_yuv(frame_yuv, alpha, bitrate, output_path)

    print("所有帧编码完成")


if __name__ == "__main__":
    process_all_frames()
