"""依訓練設定評估 Rhythm checkpoint，支援停頓裁切與整句音訊。"""

import argparse
from functools import partial
import json
import math
from pathlib import Path
import random

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils_rhythm import collate_rhythm_inputs, load_duration_csv, load_rhythm_audio
from evaluation_metric.prosdd import evaluate_score_file, load_protocol_labels, print_metrics
from full_utterance import cnn_geometry, LengthBatchSampler
from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from prosody_utils import infer_checkpoint_prosody_dim
from utils import resolve_device, set_random_seed


def load_model(model_path, config_path=None):
    """依 checkpoint 與訓練 JSON 還原完整 Rhythm 模型，回傳 (eval 模型, 設定)。"""
    config_path = Path(config_path) if config_path else Path(model_path).with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_class") != "ProSDDStage2Rhythm":
        raise ValueError("Expected a ProSDDStage2Rhythm training config")
    if config.get("sample_rate") != 16000:
        raise ValueError("Rhythm checkpoints must use a 16000 Hz sample rate")
    mode = config.get("audio_mode")
    if mode == "full_utterance":
        if config.get("audio_seconds", 0) != 0 or config.get("target_samples") is not None:
            raise ValueError("Full-utterance checkpoint must use audio_seconds=0 and target_samples=null")
    elif mode == "pause_crop":
        seconds = config["audio_seconds"]
        if (not math.isfinite(seconds) or seconds <= 0 or round(seconds * 16000) < 1
                or config.get("target_samples") != round(seconds * 16000)):
            raise ValueError("Pause-crop audio_seconds and target_samples must agree")
    elif mode is not None or config.get("target_samples") != 64000:
        raise ValueError("Unsupported checkpoint audio policy")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.get("state_dict", state).items()}
    if infer_checkpoint_prosody_dim(state) != config["prosody_dim"]:
        raise ValueError("Checkpoint and training config disagree on prosody_dim")
    options = (
        "model_name", "prosody_dim", "T_target", "rhythm_sources", "nhead",
        "n_rhythm_encoder_layers", "n_cls_encoder_layers", "dropout", "max_position_embeddings",
    )
    model = ProSDDStage2Rhythm(**{key: config[key] for key in options})
    # A partial load could silently leave an untrained classifier in place.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


class RhythmEvalDataset(Dataset):
    """讀取分類所需音訊與 rhythm，不載入 SSL teacher 或 speaker targets。

    輸入 protocol IDs、音訊目錄、duration CSV 與訓練音訊／CNN 設定。
    每筆回傳分類輸入與有效 sample 範圍；可選擇以 ID／原因記錄失敗樣本。
    相同 seed 與 ID 固定裁切位置，不受取樣順序及 worker 數影響。
    """

    def __init__(self, utt_ids, wav_dir, duration_csv, *, rhythm_sources, audio_seconds,
                 T_target, conv_kernel, conv_stride, audio_ext=".flac", seed=1234,
                 skip_bad_samples=False):
        self.wav_dir = Path(wav_dir)
        self.audio_ext = "." + audio_ext.lstrip(".")
        self.audio_seconds = audio_seconds
        self.T_target = T_target
        self.stride, self.receptive = cnn_geometry(conv_kernel, conv_stride)
        self.seed = seed
        self.skip_bad_samples = skip_bad_samples
        self.records, errors = load_duration_csv(duration_csv, [Path(utt).stem for utt in utt_ids], rhythm_sources)
        self.skipped = {utt: errors[Path(utt).stem] for utt in utt_ids if Path(utt).stem in errors}
        if self.skipped and not skip_bad_samples:
            utt = next(iter(self.skipped))
            raise ValueError(f"{utt}: {self.skipped[utt]}")
        self.utt_ids = [utt for utt in utt_ids if utt not in self.skipped]

    def audio_path(self, utt):
        return self.wav_dir / (utt if utt.endswith(self.audio_ext) else utt + self.audio_ext)

    def sample_bounds(self):
        """回傳每筆裁切／補零後 sample 數的保守上界，供 LengthBatchSampler 使用。"""
        lengths = []
        for utt in self.utt_ids:
            try:
                info = sf.info(self.audio_path(utt))
                length = math.ceil(info.frames * 16000 / info.samplerate)
            except (OSError, RuntimeError):
                if not self.skip_bad_samples:
                    raise
                length = 1  # 取樣時集中記錄讀取失敗。
            lengths.append(max(1, length, round(self.audio_seconds * 16000)))
        return lengths

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        utt = self.utt_ids[index]
        try:
            wav, rhythm, bounds = load_rhythm_audio(
                self.audio_path(utt), self.records[Path(utt).stem], max_len=self.audio_seconds,
                rng=random.Random(f"{self.seed}:{utt}"),
            )
            first = (bounds[0] + self.stride - 1) // self.stride
            if (first * self.stride + self.receptive > bounds[1]
                    or (self.T_target is not None and first >= self.T_target)):
                raise ValueError("Selected audio has no complete CNN frame within T_target")
        except (MemoryError, torch.OutOfMemoryError):
            raise
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            if not self.skip_bad_samples:
                raise ValueError(f"{utt}: {error}") from error
            return {"utt_id": utt, "reason": f"{type(error).__name__}: {error}"}
        return {"utt_id": utt, "wav": wav, "duration_features": rhythm, "valid_samples": bounds}


def collate_rhythm_eval(batch, *, T_target=None, conv_kernel, conv_stride):
    """由評估樣本組成分類 kwargs 與 skipped 記錄；padding 規則和訓練共用。"""
    skipped = {item["utt_id"]: item["reason"] for item in batch if "reason" in item}
    batch = [item for item in batch if "reason" not in item]
    if not batch:
        return {"skipped": skipped}
    inputs = collate_rhythm_inputs(
        [item["wav"] for item in batch], [item["duration_features"] for item in batch],
        [item["valid_samples"] for item in batch], T_target=T_target,
        conv_kernel=conv_kernel, conv_stride=conv_stride,
    )
    return {**inputs, "utt_ids": [item["utt_id"] for item in batch], "skipped": skipped}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list_path", required=True, help="Labeled ASVspoof protocol")
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--duration_csv", required=True, help="Syllable/vowel/consonant duration cache")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", help="Defaults to config.json beside the checkpoint")
    parser.add_argument("--save_scores_to", required=True)
    parser.add_argument("--save_metrics_to", help="成功評分子集的 EER、Cllr、minDCF、actDCF JSON")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batch_samples", type=int, default=640000,
                        help="Batch 補零後的 sample 預算；0 只依 batch_size 分批")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--audio_ext", default=".flac")
    parser.add_argument("--seed", type=int, default=1234, help="固定每個 utterance 的停頓裁切位置")
    parser.add_argument("--skip_bad_samples", action="store_true",
                        help="Record excluded IDs; report metrics only for successfully scored samples")
    return parser


@torch.inference_mode()
def main(argv=None):
    """解析 argv，依訓練設定推論並輸出分數、涵蓋率與選用的 CM 指標。

    輸出分數為 bonafide logit 減 spoof logit。protocol 副本只包含成功評分的 ID，
    coverage JSON 記錄所有排除原因；指標僅計算此子集。函式不回傳模型。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or min(args.max_batch_samples, args.num_workers) < 0:
        parser.error("batch_size must be positive; max_batch_samples and num_workers must be nonnegative")
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
    set_random_seed(args.seed)
    labels = load_protocol_labels(args.list_path)
    model, config = load_model(args.model_path, args.config_path)
    if config.get("audio_mode") not in ("pause_crop", "full_utterance"):
        parser.error("Evaluation requires a pause_crop or full_utterance training config")
    geometry = dict(conv_kernel=model.ssl.config.conv_kernel, conv_stride=model.ssl.config.conv_stride)
    dataset = RhythmEvalDataset(
        list(labels), args.wav_dir, args.duration_csv, rhythm_sources=config["rhythm_sources"],
        audio_seconds=config.get("audio_seconds", 0), T_target=config["T_target"], **geometry,
        audio_ext=args.audio_ext, seed=args.seed, skip_bad_samples=args.skip_bad_samples,
    )
    device = resolve_device()
    model.to(device)
    batching = dict(batch_size=args.batch_size)
    if args.max_batch_samples and len(dataset):
        batching = dict(batch_sampler=LengthBatchSampler(dataset.sample_bounds(), args.batch_size,
                                                       args.max_batch_samples))
    workers = dict(multiprocessing_context="spawn") if args.num_workers else {}
    loader = DataLoader(
        dataset, **batching, **workers, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=partial(collate_rhythm_eval, T_target=config["T_target"], **geometry),
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Rhythm evaluation: device={device}, audio_mode={config['audio_mode']}", flush=True)
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
            scores = (logits[:, 1] - logits[:, 0]).cpu()
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite model scores")
            for utt, score in zip(batch["utt_ids"], scores.tolist()):
                output.write(f"{utt} {score:.10g}\n")
                scored_ids.append(utt)
    with subset_path.open("x", encoding="utf-8") as output:
        output.writelines(f"{utt} {labels[utt]}\n" for utt in scored_ids)
    coverage = {
        "protocol_path": args.list_path, "duration_csv": args.duration_csv,
        "model_path": args.model_path, "protocol_samples": len(labels),
        "audio_mode": config["audio_mode"], "audio_seconds": dataset.audio_seconds,
        "T_target": config["T_target"], "rhythm_sources": config["rhythm_sources"], "seed": args.seed,
        "scored_samples": len(scored_ids), "skipped_samples": len(skipped),
        "coverage_fraction": len(scored_ids) / len(labels),
        "evaluation_scope": "retained_subset" if skipped else "full_protocol",
        "scored_protocol_path": str(subset_path), "skipped": skipped,
        "score_definition": "bonafide logit minus spoof logit; uncalibrated",
    }
    with coverage_path.open("x", encoding="utf-8") as output:
        json.dump(coverage, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"Scored {len(scored_ids)}/{len(labels)}; excluded {len(skipped)}. Coverage: {coverage_path}", flush=True)
    if not scored_ids:
        raise ValueError("No usable audio/duration samples were scored")
    if args.save_metrics_to:
        metrics = evaluate_score_file(scores_path, subset_path)
        metrics["coverage"] = {key: value for key, value in coverage.items() if key != "skipped"}
        destination = Path(args.save_metrics_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as output:
            json.dump(metrics, output, indent=2, allow_nan=False)
            output.write("\n")
        print_metrics(metrics)
        print(f"Metrics saved to {destination}", flush=True)


if __name__ == "__main__":
    main()
