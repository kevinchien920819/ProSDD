import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from typing import Optional

from prosody_utils import SUPPORTED_PROSODY_DIMS, infer_checkpoint_prosody_dim


def infer_checkpoint_dims(state: dict, spk_dim: int = 192) -> tuple[int, int]:
    """由已去除 module. 前綴的權重回傳 (prosody 維度, gender 維度)。"""
    projection = state.get("final_proj.weight")
    gender_dim = 2 if projection is not None and projection.shape[0] - spk_dim - 2 in SUPPORTED_PROSODY_DIMS else 0
    dimensions = dict(state)
    if gender_dim:
        dimensions["final_proj.weight"] = projection[:-gender_dim]
    return infer_checkpoint_prosody_dim(dimensions, spk_dim), gender_dim


def prepare_stage1_state(
    state: dict, prosody_dim: int, spk_dim: int = 192, *, use_gender: bool = False,
) -> dict[str, torch.Tensor]:
    """將 Stage 1 checkpoint 整理為兩種 Stage 2 可載入的權重。

    輸入原始 state dict 或含 state_dict 的 checkpoint，以及 Stage 2 的
    prosody／speaker 維度。回傳移除 module. 前綴的新字典，確認維度相符；
    use_gender=True 時保留完整 projection，關閉時只載入 speaker＋prosody。
    """
    state = state.get("state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if "final_proj.weight" in state or "pros_ln.weight" in state:
        checkpoint_dim, gender_dim = infer_checkpoint_dims(state, spk_dim)
        if checkpoint_dim != prosody_dim:
            raise ValueError(
                f"Prosody dim mismatch: Stage-1 checkpoint has {checkpoint_dim}, "
                f"Stage-2 expects {prosody_dim}"
            )
        if use_gender and gender_dim != 2:
            raise ValueError("use_gender=True requires a Stage-1 checkpoint with gender projection rows")
        if gender_dim and not use_gender:
            for key in ("final_proj.weight", "final_proj.bias"):
                if key in state:
                    state[key] = state[key][:-gender_dim]
    return state


class ProSDDStage2(nn.Module):

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        mask_prob: float = 0.25,
        mask_span_len: int = 8,
        tau: float = 0.07,
        num_classes: int = 2,
        stage1_ckpt: Optional[str] = None,
        num_time_neg: int = 50,
        num_spk_neg: int = 50,
        T_target: Optional[int] = 200,
        classifier_pool: str = "mean",
        prosody_dim: int = 128,
        use_gender: bool = False,
    ):
        super().__init__()

        self.mask_prob = float(mask_prob)
        self.mask_span_len = int(mask_span_len)
        self.tau = float(tau)

        self.spk_dim = 192
        self.prosody_dim = int(prosody_dim)
        self.gender_dim = 2 if use_gender else 0
        self.out_dim = self.spk_dim + self.prosody_dim + self.gender_dim
        # None 表示保留 CNN 實際 frame 數；數字則截斷／補零到固定長度。
        self.T_target = None if T_target is None else int(T_target)

        self.num_time_neg = int(num_time_neg)
        self.num_spk_neg = int(num_spk_neg)

        self.ssl = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.hidden_dim = self.ssl.config.hidden_size  # 1024

        self.mask_embed = nn.Parameter(torch.zeros(self.hidden_dim))
        nn.init.normal_(self.mask_embed, mean=0.0, std=0.02)

        self.pros_ln = nn.LayerNorm(self.prosody_dim)

        # 單一 projection 包含 speaker、prosody 與可選的 gender。
        self.final_proj = nn.Linear(self.hidden_dim, self.out_dim)

        # classifier head (on clean ctx)
        self.classifier_pool = classifier_pool
        if classifier_pool == "attn":
            self.attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
            self.attn_q = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            nn.init.normal_(self.attn_q, mean=0.0, std=0.02)

        self.cls_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

        if stage1_ckpt is not None:
            self.load_stage1(stage1_ckpt)

    def load_stage1(self, ckpt_path: str):
        """從 Stage 1 載入 backbone、mask 與包含可選 gender 的單一 projection。"""
        print(f"Loading Stage-1 weights from {ckpt_path}", flush=True)
        self._load_stage1_state(prepare_stage1_state(
            torch.load(ckpt_path, map_location="cpu", weights_only=True),
            self.prosody_dim, self.spk_dim,
            use_gender=bool(self.gender_dim),
        ))

    def _load_stage1_state(self, new_state: dict):
        """將已整理（去除 module. 前綴、維度確認）的 Stage 1 權重載入模型。"""
        # ssl
        ssl_keys = {k.replace("ssl.", ""): v for k, v in new_state.items() if k.startswith("ssl.")}
        self.ssl.load_state_dict(ssl_keys, strict=False)

        # mask
        if "mask_embed" in new_state:
            self.mask_embed.data.copy_(new_state["mask_embed"])

        # projection head
        if "final_proj.weight" in new_state and "final_proj.bias" in new_state:
            self.final_proj.load_state_dict(
                {"weight": new_state["final_proj.weight"], "bias": new_state["final_proj.bias"]},
                strict=True,
            )

        print("Stage-1 weights loaded into Stage-2 (ssl/mask/final_proj).", flush=True)

    def _embedding_target(self, speaker, prosody, gender=None):
        """將 [B,192]、[B,T,D] 與可選 [B,2] 組成每 frame 的監督向量。"""
        batch, frames = prosody.shape[:2]
        targets = [speaker.unsqueeze(1).expand(-1, frames, -1), self.pros_ln(prosody)]
        if self.gender_dim:
            if gender is None or gender.shape != (batch, self.gender_dim):
                raise ValueError("use_gender=True requires gender_emb with shape [B, 2]")
            targets.append(gender.to(prosody).unsqueeze(1).expand(-1, frames, -1))
        return torch.cat(targets, dim=-1)

    def _compute_span_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        num_to_mask = int(self.mask_prob * T)
        num_to_mask = max(1, min(num_to_mask, T))

        for b in range(B):
            masked = 0
            while masked < num_to_mask:
                start = torch.randint(0, T, (1,), device=device).item()
                end = min(start + self.mask_span_len, T)
                newly_masked = (~mask[b, start:end]).sum().item()
                mask[b, start:end] = True
                masked += newly_masked

            if not mask[b].any():
                t = torch.randint(0, T, (1,), device=device)
                mask[b, t] = True

        return mask

    def _contrastive_loss_with_metrics(self, pred, target, mask, spk_ids):
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            z = pred.new_tensor(0.0)
            return z, z, z

        b, t = idx[:, 0], idx[:, 1]
        N = b.size(0)
        B, T, D = target.shape
        device = target.device

        p = F.normalize(pred[b, t], dim=-1)
        pos = F.normalize(target[b, t], dim=-1)

        with torch.no_grad():
            pros_end = self.spk_dim + self.prosody_dim
            p_spk, p_pros = p[:, :self.spk_dim], p[:, self.spk_dim:pros_end]
            pos_spk, pos_pros = pos[:, :self.spk_dim], pos[:, self.spk_dim:pros_end]
            spk_cos = F.cosine_similarity(p_spk, pos_spk, dim=-1).mean()
            pros_cos = F.cosine_similarity(p_pros, pos_pros, dim=-1).mean()
            message = f"pos cosine | speaker: {spk_cos.item():.3f}, prosody: {pros_cos.item():.3f}"
            if self.gender_dim:
                gender_cos = F.cosine_similarity(p[:, pros_end:], pos[:, pros_end:]).mean()
                message += f", gender: {gender_cos.item():.3f}"
            print(message, flush=True)

        # Neg A: same utt, different time
        Kt = self.num_time_neg
        t_neg = torch.randint(0, T, (N, Kt), device=device)
        t_true = t.unsqueeze(1).expand_as(t_neg)
        same_t = (t_neg == t_true)
        if same_t.any():
            t_neg[same_t] = (t_neg[same_t] + 1) % T
        neg_time = F.normalize(target[b.unsqueeze(1), t_neg], dim=-1)  # (N,Kt,D)

        # Neg B: different speaker, same time
        spk_ids_tensor = torch.as_tensor(spk_ids, device=device)
        max_spk_neg = max(0, B - 1)
        Ks = min(self.num_spk_neg, max_spk_neg)

        if Ks == 0:
            neg_spk = target.new_empty((N, 0, D))
        else:
            all_idx = torch.arange(B, device=device)
            b_neg = torch.empty((N, Ks), dtype=torch.long, device=device)
            valid_counts = torch.zeros(N, dtype=torch.long, device=device)

            for i in range(N):
                anchor_b = b[i].item()
                anchor_spk = spk_ids_tensor[anchor_b]
                candidates = all_idx[all_idx != anchor_b]
                candidates = candidates[spk_ids_tensor[candidates] != anchor_spk]

                if candidates.numel() == 0:
                    valid_counts[i] = 0
                    b_neg[i].fill_(anchor_b)
                    continue

                k = min(Ks, candidates.numel())
                perm = torch.randperm(candidates.numel(), device=device)
                chosen = candidates[perm[:k]]

                if k < Ks:
                    pad = chosen[torch.randint(0, k, (Ks - k,), device=device)]
                    chosen = torch.cat([chosen, pad], dim=0)

                b_neg[i] = chosen
                valid_counts[i] = Ks

            neg_spk = F.normalize(target[b_neg, t.unsqueeze(1)], dim=-1)  # (N,Ks,D)
            no_cands = (valid_counts == 0)
            if no_cands.any():
                neg_spk[no_cands] = 0.0

        pos_sim = torch.sum(p * pos, dim=-1, keepdim=True)
        time_sim = torch.einsum("nd,nkd->nk", p, neg_time)
        spk_sim = torch.einsum("nd,nkd->nk", p, neg_spk)

        logits = torch.cat([pos_sim, time_sim, spk_sim], dim=1) / self.tau
        labels = torch.zeros(N, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)

        return loss, spk_cos.detach(), pros_cos.detach()

    def _encode(self, wav):
        """CNN 特徵抽取與投影，輸入 wav [B,L]，回傳 latent z [B,T,H]。

        T_target 為數字時截斷或補零到固定 frame 數，None 則保留實際 frame 數。
        """
        z = self.ssl.feature_extractor(wav)
        if isinstance(z, dict):
            z = z["input_values"]
        elif isinstance(z, (tuple, list)):
            z = z[0]
        z = z.transpose(1, 2)  # (B,C,T')

        # feature projection
        z = self.ssl.feature_projection(z)
        if isinstance(z, (tuple, list)):
            z = z[0]  # (B,T,H)

        B, T, H = z.shape
        Tt = self.T_target
        if Tt is not None and T > Tt:
            z = z[:, :Tt, :]
        elif Tt is not None and T < Tt:
            z = torch.cat([z, z.new_zeros(B, Tt - T, H)], dim=1)
        return z

    def _contextualize(self, z, attention_mask=None):
        """以共用的 Transformer encoder 將 latent [B,T,H] 轉為 context [B,T,H]。

        attention_mask 為 [B,T] 布林，True 表示有效 frame；None 時不做遮罩。
        """
        out = self.ssl.encoder(
            z,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state

    def _masked_pass(self, z, spk_emb, prosody_emb, spk_ids, gender_emb=None, attention_mask=None):
        """PASS 1：對 z 做 span masking 後預測 speaker／prosody targets。

        prosody_emb [B,Tp,D] 會先對齊 z 的 frame 數；回傳 (ssl_loss, spk_cos, pros_cos)。
        """
        B, T, _ = z.shape
        device = z.device

        # align prosody to T
        Tp = prosody_emb.size(1)
        if Tp < T:
            pad = prosody_emb.new_zeros(B, T - Tp, prosody_emb.size(-1))
            prosody_emb = torch.cat([prosody_emb, pad], dim=1)
        elif Tp > T:
            prosody_emb = prosody_emb[:, :T, :]

        target = self._embedding_target(spk_emb, prosody_emb, gender_emb)

        mask = self._compute_span_mask(B, T, device=device)
        z_masked = z.clone()
        z_masked[mask] = self.mask_embed.to(device)

        ctx_masked = self._contextualize(z_masked, attention_mask)
        pred = self.final_proj(ctx_masked)
        return self._contrastive_loss_with_metrics(pred, target, mask, spk_ids)

    def forward(self, wav, spk_emb, prosody_emb, spk_ids, gender_emb=None):
        """回傳真假分類 logits 與完整 concat embedding 的 contrastive loss。

        gender_emb 為 [B,2] 的固定旋轉 target，use_gender=True 時必須提供。
        """
        z = self._encode(wav)
        B = z.size(0)

        # PASS 1 masked
        ssl_loss, spk_cos, pros_cos = self._masked_pass(z, spk_emb, prosody_emb, spk_ids, gender_emb)

        # PASS 2 clean classifier
        ctx_clean = self._contextualize(z)

        if self.classifier_pool == "attn":
            q = self.attn_q.expand(B, -1, -1)
            attn_out, _ = self.attn(q, ctx_clean, ctx_clean, need_weights=False)
            pooled = attn_out.squeeze(1)
        else:
            pooled = ctx_clean.mean(dim=1)

        logits = self.cls_head(pooled)

        return {
            "logits": logits,
            "ssl_loss": ssl_loss,
            "spk_cos": spk_cos,
            "pros_cos": pros_cos,
        }
