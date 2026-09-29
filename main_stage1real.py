import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model_stage1real import ProSDDStage1
from data_utils_stage1real import ProSDDStage1Dataset
from utils import resolve_device, set_random_seed


def _batch_loss(batch, model, device):
    """將含可選 gender 的 batch 移至 device，回傳模型 loss 與 batch size。"""
    inputs = [tensor.to(device) for tensor in batch]
    return model(*inputs), inputs[0].size(0)


def train_epoch(loader, model, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        loss, bs = _batch_loss(batch, model, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(total, 1)

@torch.no_grad()
def validate(loader, model, device):
    model.eval()
    total_loss = 0.0
    total = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        loss, bs = _batch_loss(batch, model, device)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(total, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -------- data --------
    parser.add_argument("--train_prosody_txt", type=str, required=True)
    parser.add_argument("--dev_prosody_txt", type=str, required=True)
    parser.add_argument("--train_spkmean_txt", type=str, required=True)
    parser.add_argument("--dev_spkmean_txt", type=str, required=True)
    parser.add_argument("--wav_dir_train", type=str, required=True)
    parser.add_argument("--wav_dir_dev", type=str, required=True)
    parser.add_argument("--audio_ext", type=str, default=".flac")
    parser.add_argument("--prosody_dim", type=int, choices=[128, 256], default=None,
                        help="Prosody feature dimension; inferred from training data by default")
    parser.add_argument("--use_gender", action="store_true",
                        help="Concatenate rotated gender embeddings into the Stage 1 contrastive target")
    parser.add_argument("--gender_txt", help="LibriSpeech SPEAKERS.TXT shared by train and dev")
    parser.add_argument("--gender_theta", type=float, default=45.0,
                        help="Fixed counterclockwise rotation in degrees; default 45")

    # -------- training --------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ssl_lr", type=float, default=1e-6,
                        help="LR for XLS-R backbone")
    parser.add_argument("--head_lr", type=float, default=1e-4,
                        help="LR for new linear head and other randomly init params")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=0)

    # -------- ProSDD params --------
    parser.add_argument("--model_name", default="facebook/wav2vec2-xls-r-300m",
                        help="Pretrained Wav2Vec2 model ID or local directory")
    parser.add_argument("--mask_prob", type=float, default=0.25)
    parser.add_argument("--mask_span_len", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--log_dir", type=str, default="logs_stage1contrastive")
    parser.add_argument("--wandb_tags", nargs="*", default=None,
                        help="Optional W&B tags, for example: --wandb_tags prosdd stage1")

    args = parser.parse_args()
    if args.use_gender and not args.gender_txt:
        parser.error("--gender_txt is required when --use_gender is set")
    set_random_seed(args.seed)

    device = resolve_device()
    print(f"Using device: {device}", flush=True)

    # datasets
    train_dataset = ProSDDStage1Dataset(
        prosody_txt=args.train_prosody_txt,
        spkmean_txt=args.train_spkmean_txt,
        wav_dir=args.wav_dir_train,
        audio_ext=args.audio_ext,
        prosody_dim=args.prosody_dim,
        gender_txt=args.gender_txt if args.use_gender else None,
        gender_theta=args.gender_theta,
    )
    args.prosody_dim = train_dataset.prosody_dim
    print(f"Prosody dim: {args.prosody_dim}", flush=True)

    dev_dataset = ProSDDStage1Dataset(
        prosody_txt=args.dev_prosody_txt,
        spkmean_txt=args.dev_spkmean_txt,
        wav_dir=args.wav_dir_dev,
        audio_ext=args.audio_ext,
        prosody_dim=args.prosody_dim,
        gender_txt=args.gender_txt if args.use_gender else None,
        gender_theta=args.gender_theta,
    )

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Dev samples:   {len(dev_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model
    model = ProSDDStage1(
        model_name=args.model_name,
        prosody_dim=args.prosody_dim,
        mask_prob=args.mask_prob,
        mask_span_len=args.mask_span_len,
        tau=args.tau,
        use_gender=args.use_gender,
    )
    model.to(device)
    print(f"Target dim: {model.out_dim} | Gender dim: {model.gender_dim} | "
          f"Gender theta: {args.gender_theta:g} deg", flush=True)

    # collect parameters for separate LRs
    ssl_param_names = []
    ssl_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # everything under model.ssl.* gets ssl_lr
        if name.startswith("ssl."):
            ssl_params.append(p)
            ssl_param_names.append(name)
        else:
            head_params.append(p)

    print(f"SSL params: {len(ssl_params)}  Head params: {len(head_params)}", flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": ssl_params, "lr": args.ssl_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    with wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        job_type="stage1",
        tags=args.wandb_tags,
        config=vars(args),
        dir=args.log_dir,
    ) as run:
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(train_loader, model, optimizer, device)
            val_loss = validate(dev_loader, model, device)

            run.log({
                "epoch": epoch,
                "loss/train_contrastive": train_loss,
                "loss/val_contrastive": val_loss,
            }, step=epoch)

            print(f"Epoch {epoch:03d} | Train={train_loss:.6f} | Val={val_loss:.6f}", flush=True)

            if epoch == args.epochs:
                torch.save(model.state_dict(), os.path.join(args.log_dir, "model_last.pth"))
