def plot_score_lines(video_name: str, data: dict, output_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "缺少 matplotlib，请先安装：python -m pip install matplotlib"
        ) from exc

    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "<", ">"]
    bitrates = sorted(next(iter(data.values())).keys())
    plt.figure(figsize=(10, 6))

    for idx, (alpha, score_map) in enumerate(sorted(data.items(), key=lambda x: x[0])):
        scores = [score_map[bitrate] for bitrate in bitrates]
        marker = markers[idx % len(markers)]
        plt.plot(bitrates, scores, marker=marker, label=f"sharp_level={alpha:.1f}")

    plt.title(f"CLIP-based quality score by bitrate for {video_name}")
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel("Quality score")
    plt.xticks(bitrates)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
