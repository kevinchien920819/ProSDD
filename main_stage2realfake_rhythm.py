"""Train Stage II with shared masked SSL and clean Rhythm classification passes.

Required data: train/dev protocols, audio, speaker means, cached prosody targets,
and rhythm-transformer's syllabification CSVs. See
data_utils_stage2realfake_rhythm.py for the CSV schema and alignment policy.
By default entire utterances are used, with length-aware batches and right
padding. Regenerate prosody targets with extract_full_prosody.py beforehand.

Every epoch optimizes alpha * cls_loss + beta * ssl_loss in ONE backward/step.
By default alpha=1, with baseline beta=0.2 for epochs 1-4 and 0.05 thereafter.
An explicit --beta overrides the schedule with a fixed coefficient.
Training and validation use the same effective beta for each epoch.
Stage I weights are required; the Rhythm fusion head is trained from scratch.
Checkpoints are raw model state_dicts; config.json records their architecture.
"""

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data_utils_stage2realfake import (
    SAMPLING_RATE,
    TARGET_SAMPLES,
    process_Rawboost_feature,
)
from data_utils_stage2realfake_rhythm import (
    ProSDDStage2RhythmDataset,
    collate_stage2_rhythm,
    load_utt_spk_label,
)
from evaluation_metric.calculate_modules import compute_eer
from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from multi_gpu import place_model, resolve_devices
from utils import set_random_seed
from full_utterance import LengthBatchSampler


def ssl_beta_for_epoch(epoch, beta=None):
    """Match Stage II baseline unless a fixed SSL coefficient is requested."""
    return beta if beta is not None else (0.2 if epoch <= 4 else 0.05)


def build_optimizer(model, args):
    ssl_backbone_params, ssl_head_params, cls_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("ssl."):
            ssl_backbone_params.append(p)
        elif n == "mask_embed" or n.startswith("final_proj.") or n.startswith("pros_ln."):
            ssl_head_params.append(p)
        elif n.startswith("cls_head."):
            cls_params.append(p)
        else:
            raise ValueError(f"No optimizer group for trainable parameter {n}")

    return torch.optim.AdamW(
        [
            {"params": ssl_backbone_params, "lr": args.lr_ssl_backbone, "name": "ssl_backbone"},
            {"params": ssl_head_params, "lr": args.lr_ssl_head, "name": "ssl_head"},
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
    total, correct, skipped_count = 0, 0, 0
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
        "samples": total, "skipped_samples": skipped_count, "acc": correct / total,
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # -------- data --------
    parser.add_argument("--train_list", type=str, required=True, help="ASVspoof 2019/5 protocol; bonafide=1, spoof=0")
    parser.add_argument("--dev_list", type=str, required=True, help="ASVspoof 2019/5 protocol; bonafide=1, spoof=0")
    parser.add_argument("--wav_dir_train", type=str, required=True)
    parser.add_argument("--wav_dir_dev", type=str, required=True)
    parser.add_argument("--spkmean_txt_train", type=str, required=True, help="Speaker ID -> 192-D embedding")
    parser.add_argument("--prosody_txt_train", type=str, required=True, help="Utterance ID -> aligned frame-level SSL targets")
    parser.add_argument("--spkmean_txt_dev", type=str, required=True, help="Speaker ID -> 192-D embedding")
    parser.add_argument("--prosody_txt_dev", type=str, required=True, help="Utterance ID -> aligned frame-level SSL targets")
    parser.add_argument("--duration_csv_train", type=str, required=True, help="Syllabification cache for this protocol's utterances")
    parser.add_argument("--duration_csv_dev", type=str, required=True, help="Syllabification cache for this protocol's utterances")
    parser.add_argument("--prosody_dim", type=int, choices=[128, 256], default=None,
                        help="Infer from training targets if omitted")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--skip_missing_duration", action="store_true",
                        help="Exclude train/dev utterances with empty or nonfinite duration inputs; save excluded IDs in duration_filter.json")
    parser.add_argument("--skip_bad_samples", action="store_true",
                        help="Skip individual data errors (including missing durations/targets, unreadable audio and empty crops); log IDs and reasons")

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
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--mask_span_len", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num_time_neg", type=int, default=50)
    parser.add_argument("--num_spk_neg", type=int, default=50)
    parser.add_argument("--T_target", type=int, default=None,
                        help="Omit for full utterances; numeric values opt into legacy four-second training")
    parser.add_argument("--max_batch_samples", type=int, default=0,
                        help="0 keeps fixed baseline batches (default); a positive full-utterance padded sample budget enables variable-size length batches")
    

    # -------- Rhythm classification --------
    parser.add_argument("--rhythm_sources", type=str, nargs="+", choices=["syllable", "vowel", "consonant"],
                        default=["syllable"])
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n_rhythm_encoder_layers", type=int, default=2)
    parser.add_argument("--n_cls_encoder_layers", type=int, default=4, help="Cross-attention decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Rhythm/fusion dropout probability; matches the baseline classifier's rate")
    parser.add_argument("--max_position_embeddings", type=int, default=5000)

    # -------- discriminative LR --------
    parser.add_argument("--lr_ssl_backbone", type=float, default=1e-6)
    parser.add_argument("--lr_ssl_head", type=float, default=1e-4)
    parser.add_argument("--lr_cls", type=float, default=1e-5, help="All Rhythm/fusion parameters, including the final Linear")

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
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--audio_ext", type=str, default=".flac")

    # -------- W&B --------
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "ProSDD"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_name", type=str, default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb_tags", nargs="*", default=None,
                        help="Optional W&B tags, for example: --wandb_tags prosdd rhythm")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size) < 1 or args.num_workers < 0 or args.max_batch_samples < 0
            or (args.T_target is not None and args.T_target < 1)):
        parser.error("epochs, batch_size and T_target must be positive; num_workers and max_batch_samples must be nonnegative")
    initial_beta = ssl_beta_for_epoch(1, args.beta)
    coefficients = (args.alpha, initial_beta, args.weight_decay)
    rates = (args.lr_ssl_backbone, args.lr_ssl_head, args.lr_cls)
    if any(not math.isfinite(v) or v < 0 for v in (*coefficients, *rates)) or args.alpha + initial_beta == 0:
        parser.error("loss coefficients, learning rates and weight_decay must be finite/nonnegative; at least one loss must be active")
    if not 0 <= args.augment_prob <= 1:
        parser.error("augment_prob must be between 0 and 1")
    if not Path(args.stage1_ckpt).is_file():
        parser.error(f"Stage I checkpoint does not exist: {args.stage1_ckpt}")
    log_dir = Path(args.log_dir)
    if (log_dir / "config.json").exists() or (log_dir / "model_epoch_1.pth").exists():
        parser.error("log_dir already contains a run; choose a new --log_dir")

    set_random_seed(args.seed)
    devices = resolve_devices()
    device = devices[0]

    # load protocol lists
    train_utts, train_spks, train_labels = load_utt_spk_label(args.train_list)
    if set(train_labels) != {0, 1}:
        parser.error("train protocol must contain both bonafide and spoof examples")

    # dataset
    train_dataset = ProSDDStage2RhythmDataset(
        utt_ids=train_utts,
        spk_ids=train_spks,
        labels=train_labels,
        wav_dir=args.wav_dir_train,
        spkmean_txt=args.spkmean_txt_train,
        prosody_txt=args.prosody_txt_train,
        duration_csv=args.duration_csv_train,
        skip_missing_duration=args.skip_missing_duration,
        skip_bad_samples=args.skip_bad_samples,
        rhythm_sources=args.rhythm_sources,
        T_target=args.T_target,
        sr=SAMPLING_RATE,
        max_len=TARGET_SAMPLES if args.T_target is not None else None,
        audio_ext=args.audio_ext,
        prosody_dim=args.prosody_dim,
        augment_fn=process_Rawboost_feature if args.algo != 0 else None,
        augment_algo=args.algo,
        augment_prob=args.augment_prob,
        aug_args=args,
    )
    if set(train_dataset.labels) != {0, 1}:
        parser.error("train must retain both bonafide and spoof examples after duration filtering")
    args.prosody_dim = train_dataset.prosody_dim

    dev_utts, dev_spks, dev_labels = load_utt_spk_label(args.dev_list)
    if set(dev_labels) != {0, 1}:
        parser.error("dev protocol must contain both bonafide and spoof examples")

    dev_dataset = ProSDDStage2RhythmDataset(
        utt_ids=dev_utts,
        spk_ids=dev_spks,
        labels=dev_labels,
        wav_dir=args.wav_dir_dev,
        spkmean_txt=args.spkmean_txt_dev,
        prosody_txt=args.prosody_txt_dev,
        duration_csv=args.duration_csv_dev,
        skip_missing_duration=args.skip_missing_duration,
        skip_bad_samples=args.skip_bad_samples,
        rhythm_sources=args.rhythm_sources,
        T_target=args.T_target,
        sr=SAMPLING_RATE,
        max_len=TARGET_SAMPLES if args.T_target is not None else None,
        audio_ext=args.audio_ext,
        prosody_dim=args.prosody_dim,
        augment_fn=None,
        augment_algo=0,
        augment_prob=0.0,
        aug_args=args,
    )
    if set(dev_dataset.labels) != {0, 1}:
        parser.error("dev must retain both bonafide and spoof examples after duration filtering")
    args.prosody_dim = dev_dataset.prosody_dim

    # duration filtering report
    dataset_counts = {}
    if args.skip_bad_samples:
        duration_policy = "skip_bad_samples"
    elif args.skip_missing_duration:
        duration_policy = "skip_missing_duration"
    else:
        duration_policy = "error"
    duration_filter = {
        "policy": duration_policy,
        "rhythm_sources": args.rhythm_sources,
    }
    for split, dataset, utt_ids, protocol, duration_csv in (
        ("train", train_dataset, train_utts, args.train_list, args.duration_csv_train),
        ("dev", dev_dataset, dev_utts, args.dev_list, args.duration_csv_dev),
    ):
        protocol_samples = len(utt_ids)
        dataset_counts[split] = {
            "protocol_samples": protocol_samples,
            "used_samples": len(dataset),
            "skipped_at_init": len(dataset.skipped_samples),
            "skipped_missing_duration": len(dataset.skipped_duration_ids),
            "spoof": dataset.labels.count(0),
            "bonafide": dataset.labels.count(1),
        }
        duration_filter[split] = {
            "protocol": protocol,
            "duration_csv": duration_csv,
            **dataset_counts[split],
            "skipped_utt_ids": list(dataset.skipped_samples),
            "reasons": dataset.skipped_samples,
        }
        print(
            f"{split}: {len(dataset)}/{protocol_samples} samples, "
            f"skipped_at_init={len(dataset.skipped_samples)}, "
            f"skipped_missing_duration={len(dataset.skipped_duration_ids)}, prosody_dim={args.prosody_dim}",
            flush=True,
        )
        for utt, reason in list(dataset.skipped_samples.items())[:5]:
            print(f"[SKIP] {split}: {reason}", flush=True)

    # model
    model = ProSDDStage2Rhythm(
        model_name=args.model_name,
        stage1_ckpt=args.stage1_ckpt,
        prosody_dim=args.prosody_dim,
        mask_prob=args.mask_prob,
        mask_span_len=args.mask_span_len,
        tau=args.tau,
        num_time_neg=args.num_time_neg,
        num_spk_neg=args.num_spk_neg,
        T_target=args.T_target,
        d_model=args.d_model,
        rhythm_sources=args.rhythm_sources,
        nhead=args.nhead,
        n_rhythm_encoder_layers=args.n_rhythm_encoder_layers,
        n_cls_encoder_layers=args.n_cls_encoder_layers,
        dropout=args.dropout,
        max_position_embeddings=args.max_position_embeddings,
    )

    # data loaders
    collate = partial(
        collate_stage2_rhythm,
        T_target=args.T_target,
        conv_kernel=tuple(model.ssl.config.conv_kernel),
        conv_stride=tuple(model.ssl.config.conv_stride),
        skip_bad_samples=args.skip_bad_samples,
    )
    loaders = []
    for index, dataset in enumerate((train_dataset, dev_dataset)):
        if args.T_target is None:
            metadata = dataset.prosody_metadata
            if (metadata["conv_kernel"] != list(model.ssl.config.conv_kernel)
                    or metadata["conv_stride"] != list(model.ssl.config.conv_stride)):
                raise ValueError("Full-utterance prosody cache uses a different backbone frame grid")
        if args.T_target is None and args.max_batch_samples > 0:
            batching = {"batch_sampler": LengthBatchSampler(
                dataset.audio_lengths, args.batch_size, args.max_batch_samples,
                shuffle=index == 0, seed=args.seed,
            )}
        else:
            batching = {"batch_size": args.batch_size, "shuffle": index == 0, "drop_last": False}
        loaders.append(DataLoader(
            dataset, **batching, num_workers=args.num_workers,
            pin_memory=device.type == "cuda", collate_fn=collate,
            generator=torch.Generator().manual_seed(args.seed + index),
        ))
    train_loader, dev_loader = loaders
    place_model(model, devices)

    # param groups
    optimizer = build_optimizer(model, args)

    # class imbalance loss
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(weight=weight)
    print(f"Devices: {devices}; Stage I: {args.stage1_ckpt}", flush=True)
    print(f"Duration features: {model.duration_feature_names}", flush=True)
    beta_schedule = (
        {"policy": "baseline", "initial_beta": initial_beta, "initial_epochs": 4,
         "later_beta": ssl_beta_for_epoch(5)}
        if args.beta is None else {"policy": "fixed", "beta": args.beta}
    )
    print(f"Loss: {args.alpha} * cls_loss + beta * ssl_loss; beta schedule: {beta_schedule}", flush=True)

    # checkpoints and metrics
    log_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "beta_schedule": beta_schedule,
        "batching_policy": ("length_budget" if args.T_target is None and args.max_batch_samples > 0
                            else "fixed_batch_size"),
        "model_class": "ProSDDStage2Rhythm",
        "sample_rate": SAMPLING_RATE,
        "audio_mode": "full_utterance" if args.T_target is None else "center_crop",
        "target_samples": None if args.T_target is None else TARGET_SAMPLES,
        "duration_feature_names": list(model.duration_feature_names),
        "duration_crop_policy": ("all syllables; full-utterance statistics; no cropping"
                                 if args.T_target is None else
                                 "fully contained syllables; recompute statistics within center 4 s"),
        "prosody_alignment": ("cnn_receptive_field_centers" if args.T_target is None else "legacy_200_frames"),
        "dataset_counts": dataset_counts,
    }
    (log_dir / "duration_filter.json").write_text(json.dumps(duration_filter, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (log_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    best_eer = float("inf")
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        job_type="stage2-rhythm",
        tags=args.wandb_tags,
        config=config,
        dir=str(log_dir),
    ) as run, (log_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file, \
            (log_dir / "skipped_samples.jsonl").open("w", encoding="utf-8") as skipped_file:
        for split, dataset in (("train", train_dataset), ("dev", dev_dataset)):
            write_skipped_samples(
                skipped_file,
                ({"utt_id": utt, "reason": reason} for utt, reason in dataset.skipped_samples.items()),
                split=split,
                epoch=0,
            )
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        for epoch in range(1, args.epochs + 1):
            if isinstance(train_loader.batch_sampler, LengthBatchSampler):
                train_loader.batch_sampler.set_epoch(epoch)
            beta_eff = ssl_beta_for_epoch(epoch, args.beta)
            train = run_epoch(
                train_loader,
                model,
                device,
                criterion_cls,
                alpha=args.alpha,
                beta=beta_eff,
                optimizer=optimizer,
                description=f"Training (epoch {epoch})",
                on_skip=partial(write_skipped_samples, skipped_file, split="train", epoch=epoch),
            )
            val = run_epoch(
                dev_loader,
                model,
                device,
                criterion_cls,
                alpha=args.alpha,
                beta=beta_eff,
                description="Validating",
                on_skip=partial(write_skipped_samples, skipped_file, split="dev", epoch=epoch),
            )
            record = {
                "epoch": epoch,
                "alpha": args.alpha,
                "beta": beta_eff,
                **{f"train/{name}": value for name, value in train.items()},
                **{f"val/{name}": value for name, value in val.items()},
            }
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
            metrics_file.flush()
            run.log(
                wandb_metrics_for_epoch(
                    epoch=epoch,
                    alpha=args.alpha,
                    beta=beta_eff,
                    train=train,
                    val=val,
                ),
                step=epoch,
            )
            torch.save(model.state_dict(), log_dir / f"model_epoch_{epoch}.pth")
            if val["eer"] < best_eer:
                best_eer = val["eer"]
                torch.save(model.state_dict(), log_dir / "model_best.pth")
            print(
                f"Epoch {epoch:03d} | beta={beta_eff:.3f} | Train={train['loss']:.4f} "
                f"(CLS={train['cls_loss']:.4f}, SSL={train['ssl_loss']:.4f}) | "
                f"Val={val['loss']:.4f} (CLS={val['cls_loss']:.4f}, SSL={val['ssl_loss']:.4f}) | "
                f"Acc={val['acc']:.2%} | Balanced={val['acc_balanced']:.2%} | "
                f"EER={val['eer']:.2%} | Best EER={best_eer:.2%} | "
                f"Samples train/dev={train['samples']}/{val['samples']} | "
                f"Skipped train/dev={train['skipped_samples']}/{val['skipped_samples']}",
                flush=True,
            )
