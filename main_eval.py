import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the model file you shared (ensure the filename matches on your server)
from model_stage2realfake import ProSDDStage2
from data_utils_eval import ProSDDEvalDataset
from prosody_utils import infer_checkpoint_prosody_dim
from evaluation_metric.prosdd import evaluate_score_file, load_protocol_labels, print_metrics
from utils import resolve_device

@torch.no_grad()
def inference_forward(model, wav):
    #Feature Extractor
    z = model.ssl.feature_extractor(wav)
    if isinstance(z, dict): z = z["input_values"]
    elif isinstance(z, (tuple, list)): z = z[0]
    z = z.transpose(1, 2)
    z = model.ssl.feature_projection(z)
    if isinstance(z, (tuple, list)): z = z[0]
    
    #Padding (Match T_target=200)
    B, T, H = z.shape
    Tt = model.T_target
    if T > Tt: z = z[:, :Tt, :]
    elif T < Tt: z = torch.cat([z, z.new_zeros(B, Tt - T, H)], dim=1)

    #Encoder (Clean Pass)
    out = model.ssl.encoder(z, attention_mask=None, output_hidden_states=False, return_dict=True)
    ctx_clean = out.last_hidden_state

    #Pooling
    if model.classifier_pool == "attn":
        q = model.attn_q.expand(B, -1, -1)
        attn_out, _ = model.attn(q, ctx_clean, ctx_clean, need_weights=False)
        pooled = attn_out.squeeze(1)
    else:
        pooled = ctx_clean.mean(dim=1)

    #Simple Classifier Head (Linear -> ReLU -> Dropout -> Linear)
    logits = model.cls_head(pooled)
    return logits

def main(args):
    metrics_path = getattr(args, "save_metrics_to", None)
    metrics_protocol = getattr(args, "metrics_protocol", None)
    if metrics_protocol and not metrics_path:
        raise ValueError("--metrics_protocol requires --save_metrics_to")
    if metrics_path:
        metrics_protocol = metrics_protocol or args.list_path
        # Validate labels before running expensive inference.
        load_protocol_labels(metrics_protocol)

    device = resolve_device(require_cuda=True)
    print(f"Using device: {device}", flush=True)
    dataset = ProSDDEvalDataset(args.list_path, args.wav_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    print(f"Loading: {args.model_path}")
    state = torch.load(args.model_path, map_location="cpu")
    if "state_dict" in state: state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    prosody_dim = infer_checkpoint_prosody_dim(new_state)
    print(f"Prosody dim: {prosody_dim}", flush=True)
    model = ProSDDStage2(
        classifier_pool=args.classifier_pool,
        T_target=200,
        prosody_dim=prosody_dim,
    )
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()

    # Output File
    os.makedirs(os.path.dirname(args.save_scores_to) or ".", exist_ok=True)
    
    print("Starting evaluation...")
    with open(args.save_scores_to, "w") as f:
        for wav, utt_ids in tqdm(loader):
            wav = wav.to(device)
            logits = inference_forward(model, wav)
            scores = logits[:, 1] 
            scores = scores.cpu().numpy()
            for utt, score in zip(utt_ids, scores):
                f.write(f"{utt} {score:.6f}\n")
    
    print(f"Done. Scores saved to {args.save_scores_to}")
    if metrics_path:
        metrics = evaluate_score_file(args.save_scores_to, metrics_protocol, metrics_path)
        print_metrics(metrics)
        print(f"Metrics saved to {metrics_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_path", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_scores_to", required=True)
    parser.add_argument("--save_metrics_to", help="推論完成後計算 CM 指標並儲存 JSON")
    parser.add_argument("--metrics_protocol", help="指標標籤檔；預設使用 --list_path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--classifier_pool", type=str, default="mean", choices=["mean", "attn"])
    args = parser.parse_args()
    main(args)
