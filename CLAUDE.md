# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video preprocessing pipeline for finding the optimal Unsharp Mask (USM) sharpening alpha using CLIP-IQA quality assessment. Input: raw YUV420p videos. Output: H.265 encoded videos at multiple sharpening levels and bitrates, with CLIP-IQA quality scores.

## Running the Pipeline

**Full pipeline (recommended entry point):**
```
python main_frame_processing.py
```
Runs the frame-level workflow end-to-end: extract frames → USM sharpen → encode → CLIP-IQA evaluate → select best alpha per frame.

**Alternative: multi-bitrate full-video workflow:**
```
python evaluate_usm_clip_iqa.py
```
Encodes entire YUV videos at all alpha/bitrate combinations, evaluates with CLIP-IQA, and generates plots.

**Individual steps:**
```
python extract_and_encode_frames.py   # Step 1: extract frames + encode
python evaluate_frames.py             # Step 2: CLIP-IQA evaluation only
```

**Install dependencies:**
```
pip install -r requirements.txt
```

## Prerequisites

- **ffmpeg** must be installed. The path is hardcoded in scripts as `D:\develop\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe`. Override via `FFMPEG_PATH` environment variable (used by `video_usm_yuv.py` and `load_yuv_sharp_y.py`).
- **CUDA GPU** recommended for CLIP inference (falls back to CPU).
- Place raw YUV420p files in `datasets/`.

## Architecture

Two parallel evaluation approaches share common utilities:

### Frame-level workflow (`main_frame_processing.py`)
1. `extract_and_encode_frames.py` — extracts individual frames from YUV, applies USM at 11 alpha values, encodes each at 4 bitrates via ffmpeg
2. `evaluate_frames.py` — runs CLIP-IQA on each encoded frame, selects best alpha per frame
3. Results: `frame_usm_clip_iqa_results.json`, plots in `frame_plots/`

### Multi-bitrate full-video workflow (`evaluate_usm_clip_iqa.py`)
1. `video_usm_yuv.py` — reads raw YUV, pipes USM-sharpened frames to ffmpeg via stdin, encodes at all alpha/bitrate combos
2. `clip_iqa_score.py` — uses torchmetrics CLIP-IQA with multi-prompt scoring (quality, sharpness, noisiness)
3. `plot_clip_scores.py` — generates per-video bitrate vs. quality score line charts
4. Results: `usm_clip_iqa_results.json`, plots in `plots/`

### Shared utilities
- `video_usm_yuv.py` — USM sharpening (`unsharp_mask_y`), H.265 encoding, YUV I/O, ffmpeg discovery
- `clip_iqa_score.py` — CLIP model loading, YUV-to-BGR conversion, quality scoring
- `load_yuv_sharp_y.py` — standalone script for YUV loading with USM sharpening (similar to `video_usm_yuv.py`)

## Key Parameters

- Resolution: 1920x1080 (hardcoded in scripts)
- USM alpha range: -2.0 to 3.0, 11 candidates (via `np.linspace`)
- Bitrates: [1000, 2000, 3000, 4000] kbps
- Encoder: H.265 (libx265, preset medium)
- CLIP-IQA: via torchmetrics `clip_image_quality_assessment`, prompts: quality, sharpness, noisiness
- USM kernel: 5x5 Gaussian, sigma=1.0

## Output Directories

- `frames/` — extracted individual YUV frames
- `encoded_frames/` — encoded single-frame videos
- `encoded_h265/` — encoded full-video outputs
- `plots/` / `frame_plots/` — quality score charts
