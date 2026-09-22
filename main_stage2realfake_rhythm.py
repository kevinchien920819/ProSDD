"""Stage 2 Rhythm 聯合分類／SSL 訓練，每次 crop 後重新抽取 prosody。

載入 Stage 1 權重，使用乾淨音訊的 teacher targets 監督增強後的音訊。
"""

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data_utils_rhythm import ProSDDStage2RhythmDataset, collate_stage2_rhythm
from data_utils_stage2realfake import SAMPLING_RATE, load_utt_spk_label, process_Rawboost_feature
from evaluation_metric.calculate_modules import compute_eer
from extract_full_prosody import load_prosody_teacher
from full_utterance import LengthBatchSampler
from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from utils import resolve_device, set_random_seed


def ssl_beta_for_epoch(epoch, beta=None):
    """Match Stage II baseline unless a fixed SSL coefficient is requested."""
    return beta if beta is not None else (0.2 if epoch <= 4 else 0.05)


def build_optimizer(model, args):
    """依模型參數建立 AdamW，分別使用 backbone、SSL head、rhythm 與分類器學習率。

    輸入 model 與包含四組 learning rate／weight_decay 的 args；
    回傳只包含可訓練參數的 optimizer，每個參數恰屬於一組。
    """
    ssl_backbone_params, ssl_head_params, rhythm_params, cls_params = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("ssl."):
            ssl_backbone_params.append(p)
        elif n == "mask_embed" or n.startswith("final_proj.") or n.startswith("pros_ln."):
            ssl_head_params.append(p)
        elif n.startswith("rhythm_fusion."):
            rhythm_params.append(p)
        elif n.startswith("cls_head."):
            cls_params.append(p)
        else:
            raise ValueError(f"No optimizer group for trainable parameter {n}")

    return torch.optim.AdamW(
        [
            {"params": ssl_backbone_params, "lr": args.lr_ssl_backbone, "name": "ssl_backbone"},
            {"params": ssl_head_params, "lr": args.lr_ssl_head, "name": "ssl_head"},
            {"params": rhythm_params, "lr": args.lr_rhythm, "name": "rhythm"},
            {"params": cls_params, "lr": args.lr_cls, "name": "cls"},
        ],
        weight_decay=args.weight_decay,
    )


def run_epoch(
    loader, model, device, criterion_cls, *, alpha, beta, optimizer=None,
    description="", on_skip=None,
):
    """Run both passes in train AND validation; only training updates weights."""
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("loss", "ssl_loss", "cls_loss", "spk_cos", "pros_cos")}
    total, correct, skipped_count, batches = 0, 0, 0, 0
    class_total, class_correct = [0, 0], [0, 0]
    scores, keys = [], []
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=description, leave=False):
            skipped = batch.get("skipped_samples", [])
            if skipped:
                if on_skip is not None:
                    on_skip(skipped)
                for item in skipped[:max(0, 5 - skipped_count)]:
                    tqdm.write(f"[SKIP] {description} {item['utt_id']}: {item['reason']}")
                skipped_count += len(skipped)
            if "labels" not in batch:
                continue
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items() if key not in ("utt_ids", "labels", "skipped_samples")
            }
            labels = batch["labels"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            out = model(**inputs, compute_ssl=True)
            cls_loss = criterion_cls(out["logits"], labels)
            ssl_loss = out["ssl_loss"]
            loss = alpha * cls_loss + beta * ssl_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite joint loss for {batch['utt_ids']}")
            if training:
                loss.backward()
                optimizer.step()

            size = labels.numel()
            batches += 1
            total += size
            values = {"loss": loss, "ssl_loss": ssl_loss, "cls_loss": cls_loss,
                      "spk_cos": out["spk_cos"], "pros_cos": out["pros_cos"]}
            for name, value in values.items():
                totals[name] += value.detach().item() * size
            preds = out["logits"].detach().argmax(dim=1)
            correct += (preds == labels).sum().item()
            for label in (0, 1):
                selected = labels == label
                class_total[label] += selected.sum().item()
                class_correct[label] += ((preds == label) & selected).sum().item()
            if not training:
                # Log-odds of bonafide; larger scores indicate real speech.
                logits = out["logits"].detach()
                scores.append((logits[:, 1] - logits[:, 0]).cpu())
                keys.append(labels.cpu())
    if not total:
        raise ValueError("No examples processed in this epoch")
    metrics = {name: value / total for name, value in totals.items()}
    per_class = [hits / count if count else None for hits, count in zip(class_correct, class_total)]
    metrics.update({
        "samples": total, "batches": batches, "skipped_samples": skipped_count, "acc": correct / total,
        "acc_spoof": per_class[0], "acc_bonafide": per_class[1],
        "acc_balanced": sum(per_class) / 2 if all(class_total) else None,
    })
    if not training:
        if not all(class_total):
            raise ValueError("Validation EER requires both bonafide and spoof examples")
        scores, keys = torch.cat(scores).numpy(), torch.cat(keys).numpy()
        metrics["eer"] = float(compute_eer(scores[keys == 1], scores[keys == 0])[0])
    return metrics


def wandb_metrics_for_epoch(*, epoch, alpha, beta, train, val):
    """Map Rhythm metrics to the established Stage 2 W&B metric namespace.

    The local ``metrics.jsonl`` keeps its existing split-first schema
    (``train/loss``). W&B uses the metric-first Stage 2 schema
    (``loss/train_total``), allowing both training variants to share W&B's
    automatically generated Loss, Cos, and Acc panels.
    """
    return {
        "epoch": epoch,
        "alpha": alpha,
        "beta": beta,
        "loss/train_total": train["loss"],
        "loss/train_ssl": train["ssl_loss"],
        "loss/train_cls": train["cls_loss"],
        "cos/train_spk": train["spk_cos"],
        "cos/train_pros": train["pros_cos"],
        "loss/val_total": val["loss"],
        "loss/val_ssl": val["ssl_loss"],
        "loss/val_cls": val["cls_loss"],
        "cos/val_spk": val["spk_cos"],
        "cos/val_pros": val["pros_cos"],
        "acc/train": train["acc"],
        "acc/train_bonafide": train["acc_bonafide"],
        "acc/train_spoof": train["acc_spoof"],
        "acc/train_balanced": train["acc_balanced"],
        "acc/val": val["acc"],
        "acc/val_bonafide": val["acc_bonafide"],
        "acc/val_spoof": val["acc_spoof"],
        "acc/val_balanced": val["acc_balanced"],
        "eer/val": val["eer"],
        "samples/train": train["samples"],
        "samples/val": val["samples"],
        "samples/skipped_train": train["skipped_samples"],
        "samples/skipped_val": val["skipped_samples"],
    }


def write_skipped_samples(file, records, *, split, epoch):
    """Write in the parent process so multiple workers never share the log."""
    for record in records:
        file.write(json.dumps({"split": split, "epoch": epoch, **record}, ensure_ascii=False) + "\n")
    file.flush()


def build_loader(dataset, args, device, *, training):
    """依 Dataset 的 frame 幾何與 args 的 batch 預算建立 DataLoader。

    training 控制 shuffle；預算以整句／補零長度上界計算，停頓 crop 不會超過此長度。
    單筆超過預算時仍獨立成批，保留完整樣本。
    """
    batching = dict(batch_size=args.batch_size, shuffle=training)
    if args.max_batch_samples:
        lengths = []
        for utt in dataset.utt_ids:
            try:
                info = sf.info(Path(dataset.wav_dir) / (utt + dataset.audio_ext))
                length = math.ceil(info.frames * dataset.sr / info.samplerate)
            except (OSError, RuntimeError):
                if not args.skip_bad_samples:
                    raise
                length = 1  # Dataset 取樣時回傳錯誤紀錄，由主程序寫入。
            lengths.append(max(length, dataset.max_len))
        batching = dict(batch_sampler=LengthBatchSampler(
            lengths, args.batch_size, args.max_batch_samples, shuffle=training, seed=args.seed,
        ))
    workers = (dict(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=1)
               if args.num_workers else {})
    return DataLoader(
        dataset, **batching, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        **workers,
        generator=torch.Generator().manual_seed(args.seed + (0 if training else 1)),
        collate_fn=partial(collate_stage2_rhythm, T_target=dataset.T_target,
                           conv_kernel=dataset.conv_kernel, conv_stride=dataset.conv_stride,
                           skip_bad_samples=args.skip_bad_samples),
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # -------- data --------
    parser.add_argument("--train_list", type=str, required=True, help="ASVspoof 2019/5 protocol; bonafide=1, spoof=0")
    parser.add_argument("--dev_list", type=str, required=True, help="ASVspoof 2019/5 protocol; bonafide=1, spoof=0")
    parser.add_argument("--wav_dir_train", type=str, required=True)
    parser.add_argument("--wav_dir_dev", type=str, required=True)
    parser.add_argument("--spkmean_txt_train", type=str, required=True, help="Speaker ID -> 192-D embedding")
    parser.add_argument("--spkmean_txt_dev", type=str, required=True, help="Speaker ID -> 192-D embedding")
    parser.add_argument("--duration_csv_train", type=str, required=True, help="Syllabification cache for this protocol's utterances")
    parser.add_argument("--duration_csv_dev", type=str, required=True, help="Syllabification cache for this protocol's utterances")
    parser.add_argument("--prosody_dim", type=int, choices=[128, 256], default=None,
                        help="Infer from the teacher if omitted; must match Stage 1")
    parser.add_argument("--teacher_kind", choices=["mpm", "vad"], default="mpm")
    parser.add_argument("--teacher_checkpoint", help="MPM model ID/directory, or the required VAD checkpoint directory")
    parser.add_argument("--prosody_layer", type=int, default=7, help="Teacher layer; must match Stage 1 targets")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--skip_missing_duration", action="store_true",
                        help="Compatibility flag: invalid durations are always excluded and saved in duration_filter.json")
    parser.add_argument("--skip_bad_samples", action="store_true",
                        help="Skip unreadable audio, missing speakers, failed teacher extraction or empty crops; log IDs and reasons")

    # -------- training --------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0, help="Fixed classification loss coefficient")
    parser.add_argument("--beta", type=float, default=None,
                        help="Fixed SSL coefficient override; default: baseline schedule (0.2 for epochs 1-4, then 0.05)")
    parser.add_argument("--seed", type=int, default=1234)

    # -------- ProSDD --------
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--mask_prob", type=float, default=0.25)
    parser.add_argument("--mask_span_len", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--num_time_neg", type=int, default=50)
    parser.add_argument("--num_spk_neg", type=int, default=50)
    parser.add_argument("--audio_seconds", type=float, default=4.0,
                        help="Target crop duration; boundaries move to word pauses. 0 keeps full utterances")
    parser.add_argument("--T_target", type=int, default=None,
                        help="Default uses actual CNN frames; an explicit value truncates/pads frames and targets together")
    parser.add_argument("--max_batch_samples", type=int, default=0,
                        help="0 uses fixed-size batches; positive values budget padded samples using full audio lengths as conservative crop bounds")
    

    # -------- Rhythm classification --------
    parser.add_argument("--rhythm_sources", type=str, nargs="+", choices=["syllable", "vowel", "consonant"],
                        default=["syllable"])
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n_rhythm_encoder_layers", type=int, default=2)
    parser.add_argument("--n_cls_encoder_layers", type=int, default=4, help="Cross-attention decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Rhythm/fusion dropout probability; matches the baseline classifier's rate")
    parser.add_argument("--max_position_embeddings", type=int, default=5000)

    # -------- discriminative LR --------
    parser.add_argument("--lr_ssl_backbone", type=float, default=1e-6)
    parser.add_argument("--lr_ssl_head", type=float, default=1e-4)
    parser.add_argument("--lr_rhythm", type=float, default=1e-5, help="Rhythm embedding, encoder and fusion decoder")
    parser.add_argument("--lr_cls", type=float, default=1e-5, help="Final hidden_dim -> 512 -> num_classes classifier")

    # -------- RawBoost augmentation (train only) --------
    parser.add_argument("--algo", type=int, choices=range(9), default=3, help="0 disables augmentation")
    parser.add_argument("--augment_prob", type=float, default=0.5)

    parser.add_argument("--nBands", type=int, default=5)
    parser.add_argument("--minF", type=int, default=20)
    parser.add_argument("--maxF", type=int, default=8000)
    parser.add_argument("--minBW", type=int, default=100)
    parser.add_argument("--maxBW", type=int, default=1000)
    parser.add_argument("--minCoeff", type=int, default=10)
    parser.add_argument("--maxCoeff", type=int, default=100)
    parser.add_argument("--minG", type=int, default=0)
    parser.add_argument("--maxG", type=int, default=0)
    parser.add_argument("--minBiasLinNonLin", type=int, default=5)
    parser.add_argument("--maxBiasLinNonLin", type=int, default=20)
    parser.add_argument("--N_f", type=int, default=5)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument("--g_sd", type=int, default=2)
    parser.add_argument("--SNRmin", type=int, default=10)
    parser.add_argument("--SNRmax", type=int, default=40)

    parser.add_argument("--log_dir", type=str, default="logs_stage2realfake_rhythm")
    parser.add_argument("--num_workers", type=int, default=0, help="CPU teacher workers; 0 extracts in the training process")
    parser.add_argument("--audio_ext", type=str, default=".flac")

    # -------- W&B --------
    parser.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"),
                        choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "ProSDD"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_name", type=str, default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb_tags", nargs="*", default=None,
                        help="Optional W&B tags, for example: --wandb_tags prosdd rhythm")
    return parser


def main(argv=None):
    """以 CLI 參數 argv 執行 train/dev 聯合 loss；None 表示讀取命令列。

    log_dir 保存模型設定、逐輪權重與指標、最低 EER 權重及排除樣本紀錄。
    W&B 使用相同指標；函式不回傳模型，失敗時保留已完成的 epoch 產物。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, minimum in (("epochs", 1), ("batch_size", 1), ("num_workers", 0), ("max_batch_samples", 0)):
        if getattr(args, name) < minimum:
            parser.error(f"--{name} must be >= {minimum}")
    if not math.isfinite(args.audio_seconds) or args.audio_seconds < 0:
        parser.error("--audio_seconds must be finite and nonnegative")
    if args.T_target is not None and args.T_target < 1:
        parser.error("--T_target must be positive")
    if args.teacher_kind == "vad" and not args.teacher_checkpoint:
        parser.error("--teacher_kind vad requires --teacher_checkpoint")
    initial_beta = ssl_beta_for_epoch(1, args.beta)
    coefficients = (args.alpha, initial_beta, args.weight_decay)
    rates = (args.lr_ssl_backbone, args.lr_ssl_head, args.lr_rhythm, args.lr_cls)
    if any(not math.isfinite(value) or value < 0 for value in (*coefficients, *rates)) \
            or args.alpha + initial_beta == 0:
        parser.error("loss coefficients, learning rates and weight_decay must be finite/nonnegative; at least one loss must be active")
    if not 0 <= args.augment_prob <= 1:
        parser.error("--augment_prob must be between 0 and 1")
    if not Path(args.stage1_ckpt).is_file():
        parser.error(f"Stage I checkpoint does not exist: {args.stage1_ckpt}")
    log_dir = Path(args.log_dir)
    if any((log_dir / name).exists() for name in ("config.json", "model_best.pth", "model_epoch_1.pth")):
        parser.error("log_dir already contains a run; choose a new --log_dir")
    args.teacher_checkpoint = args.teacher_checkpoint or "cdminix/masked_prosody_model"
    set_random_seed(args.seed)
    device = resolve_device()
    teacher = load_prosody_teacher(args.teacher_kind, args.teacher_checkpoint).requires_grad_(False)
    if args.num_workers:
        teacher.share_memory()
    if args.prosody_dim is None:
        args.prosody_dim = teacher.args.filter_size
    model = ProSDDStage2Rhythm(
        model_name=args.model_name, stage1_ckpt=args.stage1_ckpt, prosody_dim=args.prosody_dim,
        mask_prob=args.mask_prob, mask_span_len=args.mask_span_len, tau=args.tau,
        num_time_neg=args.num_time_neg, num_spk_neg=args.num_spk_neg, T_target=args.T_target,
        rhythm_sources=args.rhythm_sources, nhead=args.nhead,
        n_rhythm_encoder_layers=args.n_rhythm_encoder_layers,
        n_cls_encoder_layers=args.n_cls_encoder_layers, dropout=args.dropout,
        max_position_embeddings=args.max_position_embeddings,
    ).to(device)
    geometry = dict(conv_kernel=model.ssl.config.conv_kernel, conv_stride=model.ssl.config.conv_stride)
    datasets, loaders = {}, {}
    for split in ("train", "dev"):
        training = split == "train"
        utts, spks, labels = load_utt_spk_label(getattr(args, f"{split}_list"))
        dataset = ProSDDStage2RhythmDataset(
            utts, spks, labels, getattr(args, f"wav_dir_{split}"),
            getattr(args, f"spkmean_txt_{split}"), teacher, getattr(args, f"duration_csv_{split}"),
            max_len=args.audio_seconds, audio_ext=args.audio_ext, prosody_dim=args.prosody_dim,
            prosody_layer=args.prosody_layer, rhythm_sources=args.rhythm_sources, T_target=args.T_target,
            skip_bad_samples=args.skip_bad_samples, skip_missing_duration=args.skip_missing_duration,
            report_bad_samples=True, skip_bad_entries=args.skip_bad_samples,
            augment_fn=process_Rawboost_feature if training and args.algo else None,
            augment_algo=args.algo if training else 0,
            augment_prob=args.augment_prob if training else 0.0, aug_args=args, **geometry,
        )
        datasets[split] = dataset
        loaders[split] = build_loader(dataset, args, device, training=training)
    optimizer = build_optimizer(model, args)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=device))
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Teacher: {args.teacher_kind}, dim={args.prosody_dim} | "
          f"Train/dev samples: {len(datasets['train'])}/{len(datasets['dev'])}", flush=True)
    config = {
        **vars(args), **geometry, "model_class": "ProSDDStage2Rhythm", "sample_rate": SAMPLING_RATE,
        "audio_mode": "full_utterance" if args.audio_seconds == 0 else "pause_crop",
        "target_samples": round(args.audio_seconds * SAMPLING_RATE) if args.audio_seconds else None,
        "prosody_alignment": "cnn_receptive_field_centers",
    }
    (log_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    filtered = {split: dataset.skipped_samples for split, dataset in datasets.items()}
    (log_dir / "duration_filter.json").write_text(json.dumps(filtered, indent=2, ensure_ascii=False), encoding="utf-8")
    best_eer = float("inf")
    with (log_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file, \
            (log_dir / "skipped_samples.jsonl").open("w", encoding="utf-8") as skipped_file, wandb.init(
        mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_name, tags=args.wandb_tags, job_type="stage2",
        config=config, dir=str(log_dir),
    ) as run:
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        for split, errors in filtered.items():
            write_skipped_samples(skipped_file, [{"utt_id": utt, "reason": reason} for utt, reason in errors.items()],
                                  split=split, epoch=0)
        for epoch in range(1, args.epochs + 1):
            if isinstance(loaders["train"].batch_sampler, LengthBatchSampler):
                loaders["train"].batch_sampler.set_epoch(epoch)
            beta = ssl_beta_for_epoch(epoch, args.beta)
            train = run_epoch(loaders["train"], model, device, criterion, alpha=args.alpha, beta=beta,
                              optimizer=optimizer, description=f"Train {epoch}",
                              on_skip=partial(write_skipped_samples, skipped_file, split="train", epoch=epoch))
            val = run_epoch(loaders["dev"], model, device, criterion, alpha=args.alpha, beta=beta,
                            description=f"Dev {epoch}",
                            on_skip=partial(write_skipped_samples, skipped_file, split="dev", epoch=epoch))
            torch.save(model.state_dict(), log_dir / f"model_epoch_{epoch}.pth")
            if val["eer"] < best_eer:
                best_eer = val["eer"]
                torch.save(model.state_dict(), log_dir / "model_best.pth")
                run.summary.update({"best_epoch": epoch, "best_eer": best_eer})
            record = {"epoch": epoch, "alpha": args.alpha, "beta": beta,
                      **{f"train/{key}": value for key, value in train.items()},
                      **{f"val/{key}": value for key, value in val.items()}}
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
            metrics_file.flush()
            run.log(wandb_metrics_for_epoch(epoch=epoch, alpha=args.alpha, beta=beta, train=train, val=val),
                    step=epoch)
            print(f"Epoch {epoch:03d} | beta={beta:g} | train={train['loss']:.4f} | "
                  f"dev={val['loss']:.4f} | acc={val['acc']:.2%} | EER={val['eer']:.2%}", flush=True)


if __name__ == "__main__":
    main()
