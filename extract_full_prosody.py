"""Extract full-utterance targets aligned to XLS-R receptive-field centers.

The frozen MPM teacher processes consecutive windows of at most six seconds,
its native process_audio window size. Every window, including the final tail,
contributes targets. Stage II still receives the entire utterance at once.
Interpolation uses physical timestamps, never a fixed 200-frame rescaling.
"""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoConfig

from data_utils_stage2realfake import _load_audio_with_ffmpeg
from full_utterance import cnn_geometry, frame_count


def load_prosody_teacher(kind="mpm", checkpoint=None):
    """載入 CPU 推論用 teacher；kind 選 mpm／vad，checkpoint 為模型位置。"""
    if kind == "vad":
        if checkpoint is None:
            raise ValueError("VAD teacher requires a checkpoint")
        from mask_prosody_model.masked_prosody_model_vad import MaskedProsodyModelVAD
        model = MaskedProsodyModelVAD.from_pretrained(checkpoint)
    elif kind == "mpm":
        from masked_prosody_model import MaskedProsodyModel
        model = MaskedProsodyModel.from_pretrained(checkpoint or "cdminix/masked_prosody_model")
    else:
        raise ValueError("teacher kind must be mpm or vad")
    return model.cpu().eval()


def load_full_audio(path):
    try:
        wav, sr = torchaudio.load(str(path))
    except (OSError, RuntimeError):
        wav, sr = _load_audio_with_ffmpeg(str(path), 16000)
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if not wav.numel() or not torch.isfinite(wav).all():
        raise ValueError("Empty or nonfinite waveform")
    return wav


@torch.inference_mode()
def extract_aligned_targets(model, wav, conv_kernel, conv_stride, *, layer=7, sr=16000):
    """以 teacher 抽取 wav [L]，回傳依 CNN 時間中心對齊的 float32 [T, D]。

    sr 是 wav 取樣率，layer 是 teacher 輸出層；conv_kernel／conv_stride
    定義學生的 frame 格點。六秒分窗涵蓋整段輸入，不再裁選片段。
    """
    stride, receptive = cnn_geometry(conv_kernel, conv_stride)
    count = frame_count(wav.numel(), stride, receptive)
    if not count:
        raise ValueError("Audio too short for a valid backbone frame")
    measure = model.vad_measure
    teacher_step = measure.hop_length / measure.sampling_rate
    times, representations = [], []
    window_samples = 6 * sr
    for start in range(0, wav.numel(), window_samples):
        window = wav[start:start + window_samples].cpu().numpy()
        # Pad only extremely short final teacher windows for the pitch estimator.
        # No fabricated samples are retained in the timestamp-aligned targets.
        teacher_window = np.pad(window, (0, max(0, round(0.1 * sr) - len(window))))
        with tempfile.NamedTemporaryFile(suffix=".wav") as file:
            sf.write(file.name, teacher_window, sr, subtype="FLOAT")
            rep = torch.as_tensor(model.process_audio(file.name, layer=layer)).detach().cpu().numpy()
        if rep.ndim != 2 or not len(rep) or not np.isfinite(rep).all():
            raise ValueError("Teacher returned invalid full-utterance representations")
        local_times = np.arange(len(rep), dtype=np.float64) * teacher_step
        keep = local_times <= len(window) / sr
        times.append(local_times[keep] + start / sr)
        representations.append(rep[keep])
    times = np.concatenate(times)
    representations = np.concatenate(representations)
    unique = np.r_[True, np.diff(times) > 1e-9]
    times, representations = times[unique], representations[unique]
    target_times = (np.arange(count) * stride + (receptive - 1) / 2) / sr
    aligned = np.stack([
        np.interp(target_times, times, representations[:, column])
        for column in range(representations.shape[1])
    ], axis=1).astype(np.float32)
    return aligned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol_txt", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--out_txt", required=True)
    parser.add_argument("--utt_col", type=int, default=1)
    parser.add_argument("--ext", default=".flac")
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--model_name", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--teacher_kind", choices=["mpm", "vad"], default="mpm")
    parser.add_argument("--teacher_checkpoint", help="Required for VAD; defaults to cdminix/masked_prosody_model for MPM")
    parser.add_argument("--skip_bad_samples", action="store_true")
    args = parser.parse_args(argv)
    if args.utt_col < 0 or (args.teacher_kind == "vad" and not args.teacher_checkpoint):
        parser.error("utt_col must be nonnegative; VAD requires --teacher_checkpoint")
    output = Path(args.out_txt)
    metadata_path = Path(str(output) + ".meta.json")
    if output.exists() or metadata_path.exists():
        parser.error("Output or metadata already exists; choose a new full-utterance cache path")
    checkpoint = args.teacher_checkpoint or "cdminix/masked_prosody_model"
    model = load_prosody_teacher(args.teacher_kind, checkpoint)
    config = AutoConfig.from_pretrained(args.model_name)
    ids = []
    with open(args.protocol_txt, encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                ids.append(line.split()[args.utt_col])
    if not ids or len(ids) != len(set(ids)):
        parser.error("Protocol must contain nonempty unique utterance IDs")
    metadata = {
        "schema_version": 1, "audio_mode": "full_utterance", "sample_rate": 16000,
        "alignment": "cnn_receptive_field_centers", "conv_kernel": list(config.conv_kernel),
        "conv_stride": list(config.conv_stride), "teacher_kind": args.teacher_kind,
        "teacher_checkpoint": checkpoint, "teacher_layer": args.layer,
        "teacher_window_seconds": 6, "utterances": {}, "skipped": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Leave no seemingly usable partial cache on failure; existing caches are
    # protected by exclusive creation and never modified.
    created = False
    try:
        with output.open("x", encoding="utf-8") as file:
            created = True
            for utt in tqdm(ids, desc="Full-utterance prosody"):
                try:
                    name = utt if utt.endswith(args.ext) else utt + args.ext
                    wav = load_full_audio(Path(args.audio_dir) / name)
                    target = extract_aligned_targets(
                        model, wav, config.conv_kernel, config.conv_stride, layer=args.layer,
                    )
                    if target.shape[1] not in (128, 256):
                        raise ValueError(f"Unsupported teacher dimension: {target.shape[1]}")
                except (MemoryError, torch.OutOfMemoryError):
                    raise
                except (OSError, ValueError, RuntimeError) as exc:
                    if not args.skip_bad_samples:
                        raise RuntimeError(f"{utt}: {exc}") from exc
                    metadata["skipped"][utt] = str(exc)
                    continue
                file.write(utt + "\t" + "|".join(
                    ",".join(f"{value:.6f}" for value in row) for row in target
                ) + "\n")
                metadata["utterances"][utt] = {"num_samples": wav.numel(), "frames": len(target)}
                metadata["prosody_dim"] = target.shape[1]
        if not metadata["utterances"]:
            raise ValueError("No full-utterance targets were extracted")
        metadata["cache_size_bytes"] = output.stat().st_size
        with metadata_path.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    print(f"Saved {len(metadata['utterances'])}/{len(ids)} full utterances to {output}; metadata: {metadata_path}")


if __name__ == "__main__":
    main()
