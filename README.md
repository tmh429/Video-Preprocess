# Video Preprocessing Pipeline

基于 CLIP-IQA 的视频 USM 锐化参数自动寻优工具。输入原始 YUV420p 视频，输出不同锐化水平和码率下的 H.265 编码视频及质量评分。

## 环境配置

```bash
conda create -n py312 python=3.12
conda activate py312
pip install -r requirements.txt
```

依赖：Python 3.10+, ffmpeg（需自行安装并配置路径）

ffmpeg 路径默认为 `D:\develop\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe`，可通过环境变量覆盖：

```bash
set FFMPEG_PATH=/path/to/ffmpeg
```

## 运行

**帧级流程（分步执行）：**

```bash
python extract_and_encode_frames.py   # 提取帧 + USM锐化 + 编码
python evaluate_frames.py             # CLIP-IQA 评估 + 选最优alpha
```

**Jupyter 演示：**

```bash
jupyter notebook example.ipynb
```

## 项目结构

```
├── utils/
│   ├── video_usm_yuv.py      # USM锐化、YUV读写、H.265编码
│   ├── clip_iqa_score.py     # CLIP-IQA 模型加载与评分
│   └── plot_clip_scores.py   # 评分曲线绘图
├── extract_and_encode_frames.py  # 帧级流程：提取+编码
├── evaluate_frames.py            # 帧级流程：评估+选最优
├── example.ipynb                 # 完整流程交互式演示
├── datasets/                     # 放置原始 YUV 文件
└── requirements.txt
```

## 关键参数

| 参数 | 值 |
|------|-----|
| 分辨率 | 1920x1080 |
| USM alpha 范围 | -2.0 ~ 3.0（11档） |
| 码率 | 1000/2000/3000/4000 kbps |
| 编码器 | H.265 (libx265) |
| USM 核 | 5x5 Gaussian, sigma=1.0 |
