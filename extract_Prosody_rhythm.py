"""對已決定範圍的音訊抽取 prosody，依實際時間對齊 CNN frames。"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from extract_full_prosody import extract_aligned_targets, load_prosody_teacher


CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)


def extract_prosody_rhythm(model, wav, *, sr=16000, layer=7, conv_kernel=CONV_KERNEL,
                          conv_stride=CONV_STRIDE):
    """從乾淨片段抽取 prosody，不重新選取或縮放音訊範圍。

    Args:
        model: 已載入的 MPM／VAD teacher，提供 process_audio 與 vad_measure。
        wav: 單聲道音訊 Tensor [L]，範圍由呼叫端決定。
        sr: wav 的取樣率。
        layer: teacher 的輸出層。
        conv_kernel, conv_stride: 學生模型的 CNN 幾何。
    Returns:
        float32 Tensor [T, D]，每列對應片段內一個 CNN frame 的時間中心。
    """
    return torch.from_numpy(extract_aligned_targets(
        model, wav, conv_kernel, conv_stride, layer=layer, sr=sr,
    ))


def main(argv=None):
    """讀取已指定的 crop，抽取 targets；可將 float32 [T, D] 存成 .pt。"""
    from data_utils_rhythm import load_audio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--out_pt")
    parser.add_argument("--teacher_kind", choices=("mpm", "vad"), default="mpm")
    parser.add_argument("--teacher_checkpoint")
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--model_name", default="facebook/wav2vec2-xls-r-300m")
    args = parser.parse_args(argv)
    if args.teacher_kind == "vad" and not args.teacher_checkpoint:
        parser.error("VAD teacher requires --teacher_checkpoint")

    wav = load_audio(args.audio_path, args.start, args.end)
    model = load_prosody_teacher(args.teacher_kind, args.teacher_checkpoint)
    config = AutoConfig.from_pretrained(args.model_name)
    targets = extract_prosody_rhythm(
        model, wav, layer=args.layer,
        conv_kernel=config.conv_kernel, conv_stride=config.conv_stride,
    )
    if args.out_pt:
        with Path(args.out_pt).open("xb") as file:
            torch.save(targets, file)
    print(f"音訊 samples={wav.numel()}，prosody={tuple(targets.shape)}")
    return targets


if __name__ == "__main__":
    main()
