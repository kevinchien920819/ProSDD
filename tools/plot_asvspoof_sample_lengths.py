#!/usr/bin/env python3
"""Plot ASVspoof2019 LA and ASVspoof5 audio-duration distributions.

The default directories cover the train, development, and evaluation splits
referenced by this repository's training scripts.  Supply one or more custom
directories with --asvspoof2019-dir and --asvspoof5-dir to override them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm


plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_2019_DIRS = (
    Path("dataset/ASVspoof2019/ASVspoof2019_LA_train/flac"),
    Path("dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac"),
    Path("dataset/ASVspoof2019/ASVspoof2019_LA_eval/flac"),
)
DEFAULT_5_DIRS = (
    Path("dataset/ASVspoof5/flac_T"),
    Path("dataset/ASVspoof5/flac_D"),
    Path("dataset/ASVspoof5/flac_E_eval"),
)
AUDIO_SUFFIXES = {".flac", ".wav"}
COLORS = {"ASVspoof2019 LA": "#0072B2", "ASVspoof5": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="統計 ASVspoof 音檔時長（秒）並繪製含四分位數的 Matplotlib 長條圖。"
    )
    parser.add_argument(
        "--asvspoof2019-dir",
        type=Path,
        action="append",
        dest="asvspoof2019_dirs",
        help="ASVspoof2019 LA 音檔目錄；可重複指定，指定後會覆蓋預設目錄。",
    )
    parser.add_argument(
        "--asvspoof5-dir",
        type=Path,
        action="append",
        dest="asvspoof5_dirs",
        help="ASVspoof5 音檔目錄；可重複指定，指定後會覆蓋預設目錄。",
    )
    parser.add_argument("--bins", type=int, default=100, help="直方圖長條數量（預設：100）。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/audio-duration/asvspoof_durations.png"),
        help="輸出的 PNG 圖檔。",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("reports/audio-duration/asvspoof_duration_stats.json"),
        help="輸出的統計 JSON 檔案。",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG 解析度（預設：200）。")
    return parser.parse_args()


def iter_audio_files(directories: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"找不到音檔目錄：{directory}")
        files.extend(path for path in directory.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
    return sorted(set(files))


def read_durations(files: list[Path], dataset_name: str) -> np.ndarray:
    if not files:
        raise ValueError(f"{dataset_name} 沒有找到 .flac 或 .wav 音檔。")

    durations: list[float] = []
    for path in tqdm(files, desc=f"讀取 {dataset_name}", unit="file"):
        try:
            info = sf.info(path)
            durations.append(info.frames / info.samplerate)
        except RuntimeError as exc:
            raise RuntimeError(f"無法讀取音檔資訊：{path}") from exc
    return np.asarray(durations, dtype=np.float64)


def summary(lengths: np.ndarray) -> dict[str, float | int]:
    q1, median, q3 = np.percentile(lengths, [25, 50, 75])
    return {
        "count": int(lengths.size),
        "min": float(lengths.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(lengths.max()),
        "mean": float(lengths.mean()),
    }


def add_histogram(ax: plt.Axes, lengths: np.ndarray, name: str, bins: int) -> dict[str, float | int]:
    stats = summary(lengths)
    ax.hist(lengths, bins=bins, color=COLORS[name], edgecolor="white", linewidth=0.25)
    for label, value, linestyle in (("Q1", stats["q1"], "--"), ("Q2（中位數）", stats["median"], "-"), ("Q3", stats["q3"], "--")):
        ax.axvline(value, color="#202124", linestyle=linestyle, linewidth=1.5, label=f"{label}: {value:.2f} 秒")
    ax.axvline(stats["mean"], color="#D62728", linewidth=2, label=f"平均值: {stats['mean']:.2f} 秒")
    ax.set_title(name)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.legend(fontsize=9)
    return stats


def main() -> None:
    args = parse_args()
    if args.bins < 1 or args.dpi < 1:
        raise ValueError("--bins 與 --dpi 必須大於或等於 1。")

    directories = {
        "ASVspoof2019 LA": args.asvspoof2019_dirs or DEFAULT_2019_DIRS,
        "ASVspoof5": args.asvspoof5_dirs or DEFAULT_5_DIRS,
    }
    lengths_by_dataset = {
        name: read_durations(iter_audio_files(paths), name)
        for name, paths in directories.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    stats = {
        name: add_histogram(ax, lengths, name, args.bins)
        for ax, (name, lengths) in zip(axes, lengths_by_dataset.items(), strict=True)
    }
    fig.suptitle("ASVspoof 音檔時長分布", fontsize=16)
    fig.supxlabel("音檔時長（秒）")
    fig.supylabel("統計數量")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    args.stats_output.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, values in stats.items():
        print(
            f"{name}: count={values['count']:,}, Q1={values['q1']:.2f} 秒, "
            f"median={values['median']:.2f} 秒, Q3={values['q3']:.2f} 秒, "
            f"mean={values['mean']:.2f} 秒"
        )
    print(f"Matplotlib 圖表：{args.output}")
    print(f"統計資料：{args.stats_output}")


if __name__ == "__main__":
    main()
