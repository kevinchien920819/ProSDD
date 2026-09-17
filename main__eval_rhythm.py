"""Evaluate Rhythm checkpoints with the same audio/duration alignment as training."""

import argparse
import json
from pathlib import Path
import subprocess
import math

import soundfile as sf

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils_stage2realfake import _load_audio_with_ffmpeg, pad
from data_utils_stage2realfake_rhythm import (
    SkippedSample, duration_features_for_window, duration_features_for_utterance, load_duration_csv,
)
from evaluation_metric.prosdd import evaluate_score_file, load_protocol_labels, print_metrics
from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from multi_gpu import place_model, resolve_devices
from prosody_utils import infer_checkpoint_prosody_dim
from full_utterance import cnn_geometry, frame_count, LengthBatchSampler


def load_model(model_path, config_path=None):
    config_path = Path(config_path) if config_path else Path(model_path).with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_class") != "ProSDDStage2Rhythm":
        raise ValueError("Expected a ProSDDStage2Rhythm training config")
    full_utterance = config.get("audio_mode") == "full_utterance"
    if full_utterance:
        if config.get("sample_rate") != 16000 or config.get("T_target") is not None or config.get("target_samples") is not None:
            raise ValueError("Full-utterance checkpoint must use dynamic audio/frame lengths")
    elif (config.get("sample_rate"), config.get("target_samples")) != (16000, 64000):
        raise ValueError("Expected the training audio policy: 16 kHz, center crop/pad to 4 s")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.get("state_dict", state).items()}
    if infer_checkpoint_prosody_dim(state) != config["prosody_dim"]:
        raise ValueError("Checkpoint and training config disagree on prosody_dim")
    options = (
        "model_name", "prosody_dim", "T_target", "d_model", "rhythm_sources", "nhead",
        "n_rhythm_encoder_layers", "n_cls_encoder_layers", "dropout", "max_position_embeddings",
    )
    model = ProSDDStage2Rhythm(**{key: config[key] for key in options})
    # A partial load could silently leave an untrained classifier in place.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


class RhythmEvalDataset(Dataset):
    """Audio and duration inputs only; inference does not require SSL targets."""

    def __init__(self, utt_ids, wav_dir, duration_csv, *, rhythm_sources, T_target,
                 conv_kernel, conv_stride, audio_ext=".flac", skip_bad_samples=False):
        self.wav_dir = Path(wav_dir)
        self.audio_ext = "." + audio_ext.lstrip(".")
        self.skip_bad_samples = skip_bad_samples
        self.T_target = T_target
        self.skipped = {}
        self.durations = load_duration_csv(
            duration_csv, utt_ids, rhythm_sources,
            skip_bad_samples=skip_bad_samples, errors=self.skipped,
        )
        self.utt_ids = [utt for utt in utt_ids if utt in self.durations]
        if not self.utt_ids:
            raise ValueError("No utterances with usable durations remain")
        stride, receptive_field = cnn_geometry(conv_kernel, conv_stride)
        self.stride, self.receptive_field = stride, receptive_field
        if T_target is None:
            # Header-only reads for length-aware batches. Decoder fallbacks are
            # still handled by __getitem__, without excluding valid audio here.
            self.audio_lengths = []
            for utt in self.utt_ids:
                name = utt if utt.endswith(self.audio_ext) else utt + self.audio_ext
                try:
                    info = sf.info(self.wav_dir / name)
                    samples = math.ceil(info.frames * 16000 / info.samplerate)
                except (OSError, RuntimeError):
                    samples = 640000  # conservative bucket estimate, not a crop
                self.audio_lengths.append(max(1, samples))
            return
        if T_target < 1:
            raise ValueError("Invalid CNN geometry or T_target")
        actual_frames = max(0, (64000 - receptive_field) // stride + 1)
        if actual_frames > T_target:
            raise ValueError("T_target truncates acoustic frames; duration alignment would differ")
        self.frame_start = torch.arange(T_target) * stride
        self.frame_end = self.frame_start + receptive_field
        self.actual_frame_mask = torch.arange(T_target) < actual_frames

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        utt = self.utt_ids[index]
        name = utt if utt.endswith(self.audio_ext) else utt + self.audio_ext
        path = str(self.wav_dir / name)
        try:
            try:
                wav, sr = torchaudio.load(path)
            except (MemoryError, torch.OutOfMemoryError):
                raise
            except (OSError, RuntimeError):
                wav, sr = _load_audio_with_ffmpeg(path, 16000)
            wav = wav.mean(dim=0)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            samples = wav.numel()
            if not samples or not torch.isfinite(wav).all():
                raise ValueError("Empty or nonfinite waveform")
            if self.T_target is None:
                duration = duration_features_for_utterance(self.durations[utt], samples / 16000)
                count = frame_count(samples, self.stride, self.receptive_field)
                valid = torch.ones(count, dtype=torch.bool)
            else:
                crop_start = max(0, (samples - 64000) // 2)
                crop_end = min(samples, crop_start + 64000)
                valid_start = max(0, (64000 - samples) // 2)
                valid_end = valid_start + min(samples, 64000)
                duration = duration_features_for_window(
                    self.durations[utt], crop_start / 16000, crop_end / 16000,
                )
                valid = ((self.frame_start >= valid_start) & (self.frame_end <= valid_end)
                         & self.actual_frame_mask)
                wav = pad(wav, 64000)
            if not valid.any():
                raise ValueError("Audio too short for a valid CNN frame")
            if not torch.isfinite(duration).all():
                raise ValueError("Nonfinite duration features")
        except (MemoryError, torch.OutOfMemoryError):
            raise
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            if self.skip_bad_samples:
                return SkippedSample(utt, str(exc))
            raise RuntimeError(f"{utt}: {exc}") from exc
        return {"utt_id": utt, "wav": wav, "duration_features": duration,
                "frame_padding_mask": ~valid}


def collate_rhythm_eval(batch):
    skipped = {item.utt_id: item.reason for item in batch if isinstance(item, SkippedSample)}
    batch = [item for item in batch if not isinstance(item, SkippedSample)]
    if not batch:
        return {"skipped": skipped}
    durations = pad_sequence([item["duration_features"] for item in batch],
                             batch_first=True, padding_value=-100)
    lengths = torch.tensor([len(item["duration_features"]) for item in batch])
    return {
        "skipped": skipped, "utt_ids": [item["utt_id"] for item in batch],
        "wav": pad_sequence([item["wav"] for item in batch], batch_first=True),
        "duration_features": durations,
        "frame_padding_mask": pad_sequence([item["frame_padding_mask"] for item in batch],
                                           batch_first=True, padding_value=True),
        "rhythm_padding_mask": torch.arange(durations.size(1))[None] >= lengths[:, None],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list_path", required=True, help="Labeled ASVspoof protocol")
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--duration_csv", required=True, help="Syllable/vowel/consonant duration cache")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", help="Defaults to config.json beside the checkpoint")
    parser.add_argument("--save_scores_to", required=True)
    parser.add_argument("--save_metrics_to")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batch_samples", type=int, default=640000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--audio_ext", default=".flac")
    parser.add_argument("--skip_bad_samples", action="store_true",
                        help="Record excluded IDs; report metrics only for successfully scored samples")
    return parser


@torch.no_grad()
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.batch_size, args.max_batch_samples) < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers nonnegative")
    scores_path = Path(args.save_scores_to)
    subset_path = scores_path.with_suffix(".protocol.txt")
    coverage_path = scores_path.with_suffix(".coverage.json")
    outputs = [scores_path, subset_path, coverage_path]
    if args.save_metrics_to:
        outputs.append(Path(args.save_metrics_to))
    inputs = [Path(args.list_path), Path(args.duration_csv), Path(args.model_path),
              Path(args.config_path) if args.config_path else Path(args.model_path).with_name("config.json")]
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("Output paths must be distinct")
    if {path.resolve() for path in inputs} & {path.resolve() for path in outputs}:
        parser.error("Outputs must not overwrite input files")
    for path in outputs:
        if path.exists():
            parser.error(f"Output already exists; choose a new output path: {path}")
    labels = load_protocol_labels(args.list_path)
    model, config = load_model(args.model_path, args.config_path)
    dataset = RhythmEvalDataset(
        list(labels), args.wav_dir, args.duration_csv, rhythm_sources=config["rhythm_sources"],
        T_target=config["T_target"], conv_kernel=model.ssl.config.conv_kernel,
        conv_stride=model.ssl.config.conv_stride, audio_ext=args.audio_ext,
        skip_bad_samples=args.skip_bad_samples,
    )
    devices = resolve_devices(require_cuda=True)
    place_model(model, devices)
    device = devices[0]
    batching = ({"batch_sampler": LengthBatchSampler(dataset.audio_lengths, args.batch_size, args.max_batch_samples)}
                if config.get("audio_mode") == "full_utterance" else {"batch_size": args.batch_size})
    loader = DataLoader(dataset, **batching, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", collate_fn=collate_rhythm_eval)
    print(f"Rhythm checkpoint: {args.model_path}; devices: {devices}", flush=True)
    print(f"Duration records: {len(dataset)}/{len(labels)}; skipped at init: {len(dataset.skipped)}", flush=True)
    skipped = dict(dataset.skipped)
    scored_ids = []
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("x", encoding="utf-8") as output:
        for batch in tqdm(loader, desc="Rhythm evaluation"):
            skipped.update(batch["skipped"])
            if "wav" not in batch:
                continue
            inputs = {key: batch[key].to(device, non_blocking=True) for key in (
                "wav", "duration_features", "frame_padding_mask", "rhythm_padding_mask",
            )}
            logits = model(**inputs, compute_ssl=False)["logits"]
            # Same bonafide log-odds used by Stage 2 Rhythm's dev EER.
            scores = (logits[:, 1] - logits[:, 0]).cpu()
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite model scores")
            for utt, score in zip(batch["utt_ids"], scores.tolist()):
                output.write(f"{utt} {score:.10g}\n")
                scored_ids.append(utt)
    subset_path.write_text("".join(f"{utt} {labels[utt]}\n" for utt in scored_ids), encoding="utf-8")
    coverage = {
        "protocol_path": args.list_path, "duration_csv": args.duration_csv,
        "model_path": args.model_path, "protocol_samples": len(labels),
        "audio_mode": config.get("audio_mode", "center_crop"),
        "scored_samples": len(scored_ids), "skipped_samples": len(skipped),
        "coverage_fraction": len(scored_ids) / len(labels),
        "evaluation_scope": "retained_subset" if skipped else "full_protocol",
        "scored_protocol_path": str(subset_path), "skipped": skipped,
        "score_definition": "bonafide logit minus spoof logit; uncalibrated",
    }
    coverage_path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scored {len(scored_ids)}/{len(labels)}; excluded {len(skipped)}. Coverage: {coverage_path}", flush=True)
    if not scored_ids:
        raise ValueError("No usable audio/duration samples were scored")
    if args.save_metrics_to:
        metrics = evaluate_score_file(scores_path, subset_path)
        metrics["coverage"] = {key: value for key, value in coverage.items() if key != "skipped"}
        destination = Path(args.save_metrics_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print_metrics(metrics)
        print(f"Metrics saved to {destination}", flush=True)


if __name__ == "__main__":
    main()
