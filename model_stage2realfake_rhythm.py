"""以 ProSDD Stage II 為基底，將 rhythm 與 clean 聲學特徵融合後分類。

masked／clean 兩個 pass 共用 SSL backbone。RhythmTransformer 的 embedding、
encoder 與 decoder 全程使用 backbone hidden_dim，取融合後的 rhythm CLS，
接回範本的 hidden_dim → 512 → num_classes 分類器。

Training::

    out = model(wav, spk_emb, prosody_emb, spk_ids, duration_features)
    loss = criterion_cls(out["logits"], labels) + beta * out["ssl_loss"]
    loss.backward()

Inference (call model.eval() and use torch.no_grad() separately)::

    out = model(wav, duration_features=duration_features, compute_ssl=False)

Duration statistics and SSL targets must describe the same full audio as
``wav``. T_target=None uses all frames; explicit numeric values retain legacy
checkpoint behavior. This module does not recompute statistics or load CSVs.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model_stage2realfake import ProSDDStage2
from prosody_utils import infer_checkpoint_prosody_dim


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * frequency)
        pe[0, :, 1::2] = torch.cos(position * frequency[:d_model // 2])
        self.register_buffer("pe", pe)
        self.scale = math.sqrt(d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            # Generate longer sinusoidal positions without changing checkpoint
            # buffer shapes or imposing a silent utterance-length limit.
            position = torch.arange(x.size(1), device=x.device).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, x.size(2), 2, device=x.device).float()
                                  * (-math.log(10000.0) / x.size(2)))
            pe = x.new_zeros(1, x.size(1), x.size(2), dtype=torch.float32)
            pe[0, :, 0::2] = torch.sin(position * frequency)
            pe[0, :, 1::2] = torch.cos(position * frequency[:x.size(2) // 2])
        else:
            pe = self.pe[:, :x.size(1)]
        return x * self.scale + pe.to(dtype=x.dtype)


class RhythmFusion(nn.Module):
    """以 rhythm tokens 查詢 clean 聲學序列，輸出 utterance CLS 特徵。

    輸入為聲學 [B, T, H]、duration [B, N, F] 與各自的 padding mask；
    H 使用 backbone hidden_dim，F 為每種 rhythm source 的三個統計值。
    輸出為 [B, H]，分類器由 ProSDDStage2Rhythm 單獨管理。
    """

    def __init__(
        self, hidden_dim: int,
        rhythm_sources: Sequence[str], nhead: int,
        n_rhythm_encoder_layers: int, n_cls_encoder_layers: int,
        dropout: float, max_position_embeddings: int,
    ):
        super().__init__()
        self.feature_names = tuple(
            f"{source}_{suffix}" for source in rhythm_sources
            for suffix in ("d", "devi", "mu_diff")
        )
        self.rhythm_embedding = nn.Sequential(
            nn.Linear(len(self.feature_names), hidden_dim),
            _PositionalEncoding(hidden_dim, max_position_embeddings),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(dropout),
        )
        self.rhythm_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, nhead, dim_feedforward=4 * hidden_dim, dropout=dropout,
                activation="gelu", batch_first=True,
            ),
            num_layers=n_rhythm_encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_dim, nhead, dim_feedforward=4 * hidden_dim, dropout=dropout,
                activation="gelu", batch_first=True,
            ),
            num_layers=n_cls_encoder_layers,
        )
        self.pos = _PositionalEncoding(hidden_dim, max_position_embeddings)
        self.layernorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def _duration_inputs(self, features, padding_mask, frames):
        if isinstance(features, Mapping):
            missing = [name for name in self.feature_names if name not in features]
            if missing:
                raise ValueError(f"Missing duration features: {', '.join(missing)}")
            values = [features[name] for name in self.feature_names]
            if any(not isinstance(v, Tensor) or v.ndim != 2 for v in values):
                raise ValueError("Each duration field must be a [B, N] tensor")
            if any(v.shape != values[0].shape for v in values):
                raise ValueError("All duration fields must have the same [B, N] shape")
            features = torch.stack(values, dim=-1)
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError("duration_features must be a field mapping or a [B, N, F] tensor")
        if features.size(0) != frames.size(0) or features.size(2) != len(self.feature_names):
            raise ValueError(f"Expected duration shape [B, N, {len(self.feature_names)}]")
        features = features.to(device=frames.device, dtype=frames.dtype)
        sentinel = features == -100.0
        if padding_mask is None:
            padding_mask = sentinel.all(dim=-1)
        else:
            if padding_mask.dtype != torch.bool or padding_mask.shape != features.shape[:2]:
                raise ValueError("rhythm_padding_mask must be boolean [B, N] (True = padding)")
            padding_mask = padding_mask.to(frames.device)
        if not (~padding_mask).any(dim=1).all():
            raise ValueError("Each utterance needs at least one valid duration token")
        valid = ~padding_mask.unsqueeze(-1)
        if (sentinel & valid).any() or ((~torch.isfinite(features)) & valid).any():
            raise ValueError("Valid duration tokens must be finite and contain no -100 padding")
        return features.masked_fill(padding_mask.unsqueeze(-1), 0.0), padding_mask

    def forward(self, frames, duration_features, frame_padding_mask, rhythm_padding_mask=None):
        """融合兩種不同長度的序列；mask 的 True 表示人工 padding。"""
        duration, rhythm_padding_mask = self._duration_inputs(
            duration_features, rhythm_padding_mask, frames,
        )
        memory = frames.masked_fill(frame_padding_mask.unsqueeze(-1), 0.0)
        # Match RhythmTransformerWithDuration: a zero acoustic CLS before PE,
        # and a zero rhythm CLS after duration embedding/PE.
        cls = memory.new_zeros(memory.size(0), 1, memory.size(2))
        memory = torch.cat([cls, memory], dim=1)
        memory = self.dropout(self.layernorm(self.pos(memory)))
        rhythm = torch.cat([cls, self.rhythm_embedding(duration)], dim=1)
        cls_padding = frame_padding_mask.new_zeros(frames.size(0), 1)
        memory_padding = torch.cat([cls_padding, frame_padding_mask], dim=1)
        rhythm_padding = torch.cat([cls_padding, rhythm_padding_mask], dim=1)
        rhythm = self.rhythm_encoder(rhythm, src_key_padding_mask=rhythm_padding)
        fused = self.decoder(
            tgt=rhythm, memory=memory,
            tgt_key_padding_mask=rhythm_padding,
            memory_key_padding_mask=memory_padding,
        )
        return self.layernorm(fused[:, 0, :])


class ProSDDStage2Rhythm(ProSDDStage2):
    """Two-pass Stage II, using one Stage I backbone and a Rhythm fusion head.

    ``forward`` inputs:
      * wav: [B, L], mono waveforms using the Stage I preprocessing.
      * spk_emb / prosody_emb / spk_ids: [B, 192],
        [B, acoustic_frames, prosody_dim], [B]. Required when compute_ssl=True.
      * duration_features: [B, N, F] or a mapping of [B, N] tensors. Columns
        follow ``duration_feature_names`` (source first, then d/devi/mu_diff).
        N may differ from acoustic_frames. By default -100 in every field marks padding.
      * frame_padding_mask: optional boolean [B, acoustic_frames], True for artificial
        padding. Supports left/center/right padding; real pauses stay valid.
        Padding added to the latent sequence by this model is always excluded.
        Padding already present inside wav must be identified by the caller.
      * rhythm_padding_mask: optional boolean [B, N], overriding the sentinel.

    Returns logits [B, num_classes], feature [B, hidden_dim], and scalar ssl_loss,
    spk_cos, pros_cos. The three SSL values are None if compute_ssl=False.
    Callers compute classification loss and combine it with ssl_loss. eval()
    alone does not disable the masked pass, so validation can measure both losses.
    """

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
        T_target: Optional[int] = None,
        prosody_dim: int = 128,
        *,
        rhythm_sources: Sequence[str] = ("syllable", "vowel", "consonant"),
        nhead: int = 4,
        n_rhythm_encoder_layers: int = 2,
        n_cls_encoder_layers: int = 4,
        dropout: float = 0.1,
        max_position_embeddings: int = 5000,
    ):
        sources = tuple(rhythm_sources)
        if not sources or len(set(sources)) != len(sources) or any(
            s not in ("syllable", "vowel", "consonant") for s in sources
        ):
            raise ValueError("rhythm_sources must select unique syllable/vowel/consonant sources")
        if min(n_rhythm_encoder_layers, n_cls_encoder_layers) < 1 or (T_target is not None and T_target < 1):
            raise ValueError("Encoder/decoder layer counts and T_target must be positive")
        if max_position_embeddings < (T_target + 1 if T_target is not None else 1):
            raise ValueError("max_position_embeddings must cover T_target + CLS")
        if not 0 <= mask_prob <= 1 or mask_span_len < 1 or tau <= 0:
            raise ValueError("Require 0 <= mask_prob <= 1, mask_span_len >= 1, tau > 0")
        if min(num_time_neg, num_spk_neg) < 0 or not 0 <= dropout < 1:
            raise ValueError("Negative counts must be nonnegative and 0 <= dropout < 1")
        super().__init__(
            model_name=model_name, mask_prob=mask_prob, mask_span_len=mask_span_len,
            tau=tau, num_classes=num_classes, num_time_neg=num_time_neg,
            num_spk_neg=num_spk_neg, T_target=T_target or 200, prosody_dim=prosody_dim,
        )
        self.T_target = T_target
        self.classifier_pool = "rhythm"
        # 沿用範本的 cls_head；rhythm/fusion 使用獨立的 learning rate。
        self.rhythm_fusion = RhythmFusion(
            self.hidden_dim, sources, nhead,
            n_rhythm_encoder_layers, n_cls_encoder_layers, dropout,
            max_position_embeddings,
        )
        self.duration_feature_names = self.rhythm_fusion.feature_names
        if stage1_ckpt is not None:
            self.load_stage1(stage1_ckpt)

    def load_stage1(self, ckpt_path: str):
        """Restore the backbone, mask, projection AND learned target LayerNorm."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        checkpoint_dim = infer_checkpoint_prosody_dim(state, self.spk_dim)
        if checkpoint_dim != self.prosody_dim:
            raise ValueError(
                f"Prosody dim mismatch: Stage-1 checkpoint has {checkpoint_dim}, "
                f"Stage-2 expects {self.prosody_dim}"
            )
        for name in ("ssl", "final_proj", "pros_ln"):
            prefix = name + "."
            weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            getattr(self, name).load_state_dict(weights, strict=True)
        if state["mask_embed"].shape != self.mask_embed.shape:
            raise ValueError("Stage-1 mask_embed shape does not match the backbone")
        with torch.no_grad():
            self.mask_embed.copy_(state["mask_embed"])

    def _encode_latents(self, wav, frame_padding_mask):
        if wav.ndim != 2 or wav.size(0) == 0:
            raise ValueError("wav must be a nonempty [B, L] tensor")
        z = self.ssl.feature_extractor(wav).transpose(1, 2)
        z = self.ssl.feature_projection(z)[0]
        original_frames = z.size(1)
        target_frames = original_frames if self.T_target is None else self.T_target
        z = z[:, :target_frames]
        z = F.pad(z, (0, 0, 0, max(0, target_frames - z.size(1))))
        padding = (torch.arange(target_frames, device=z.device) >= original_frames)
        padding = padding.unsqueeze(0).expand(z.size(0), -1)
        if frame_padding_mask is not None:
            if frame_padding_mask.dtype != torch.bool or frame_padding_mask.shape != padding.shape:
                raise ValueError("frame_padding_mask must be boolean [B, acoustic_frames] (True = padding)")
            padding = padding | frame_padding_mask.to(z.device)
        if not (~padding).any(dim=1).all():
            raise ValueError("Each utterance needs at least one valid acoustic frame")
        return z.masked_fill(padding.unsqueeze(-1), 0.0), padding

    def _valid_span_mask(self, valid):
        mask = torch.zeros_like(valid)
        if self.mask_prob == 0:
            return mask
        for b in range(valid.size(0)):
            positions = valid[b].nonzero(as_tuple=True)[0]
            budget = max(1, int(self.mask_prob * positions.numel()))
            for start in positions[torch.randperm(positions.numel(), device=valid.device)]:
                start = int(start)
                stop = min(start + self.mask_span_len, valid.size(1))
                mask[b, start:stop] |= valid[b, start:stop]
                if int(mask[b].sum()) >= budget:
                    break
        return mask

    def _masked_ssl_loss(self, pred, target, mask, spk_ids, valid):
        """Keep ProSDD's negative types, excluding padded/absent candidates.

        Time negatives use the same utterance at a different valid frame.
        Speaker negatives use another speaker's target at the same frame index,
        following model_stage2realfake.py. Time sampling uses replacement;
        speaker sampling repeats candidates only when too few are available.
        """
        b, t = mask.nonzero(as_tuple=True)
        if b.numel() == 0:
            zero = pred.sum() * 0.0
            return zero, zero.detach(), zero.detach()
        p = F.normalize(pred[b, t].float(), dim=-1)
        targets = F.normalize(target.float(), dim=-1)
        pos = targets[b, t]
        scores = [(p * pos).sum(dim=-1, keepdim=True)]

        def sample_candidates(candidates, count, replacement=True):
            weights = candidates.float()
            present = candidates.any(dim=1)
            # torch.multinomial needs nonzero mass even for excluded rows.
            weights[~present, 0] = 1.0
            if replacement:
                indices = torch.multinomial(weights, count, replacement=True)
            else:
                indices = torch.zeros(
                    candidates.size(0), count, dtype=torch.long, device=pred.device,
                )
                available = candidates.sum(dim=1)
                for size in available.unique().tolist():
                    if size == 0:
                        continue
                    rows = available == size
                    take = min(size, count)
                    chosen = torch.multinomial(weights[rows], take, replacement=False)
                    if take < count:
                        repeats = torch.randint(take, (chosen.size(0), count - take), device=pred.device)
                        chosen = torch.cat([chosen, chosen.gather(1, repeats)], dim=1)
                    indices[rows] = chosen
            return indices, present

        if self.num_time_neg:
            candidates = valid[b].clone()
            candidates.scatter_(1, t.unsqueeze(1), False)
            times, present = sample_candidates(candidates, self.num_time_neg)
            negatives = targets[b.unsqueeze(1), times]
            similarity = torch.einsum("nd,nkd->nk", p, negatives)
            scores.append(similarity.masked_fill(~present.unsqueeze(1), -torch.inf))
        count = min(self.num_spk_neg, target.size(0) - 1)
        if count:
            candidates = spk_ids[b].unsqueeze(1) != spk_ids.unsqueeze(0)
            candidates &= valid[:, t].transpose(0, 1)
            speakers, present = sample_candidates(candidates, count, replacement=False)
            negatives = targets[speakers, t.unsqueeze(1)]
            similarity = torch.einsum("nd,nkd->nk", p, negatives)
            scores.append(similarity.masked_fill(~present.unsqueeze(1), -torch.inf))
        logits = torch.cat(scores, dim=1) / self.tau
        loss = F.cross_entropy(logits, torch.zeros_like(b))
        with torch.no_grad():
            spk_cos = F.cosine_similarity(p[:, :self.spk_dim], pos[:, :self.spk_dim]).mean()
            pros_cos = F.cosine_similarity(p[:, self.spk_dim:], pos[:, self.spk_dim:]).mean()
        return loss, spk_cos, pros_cos

    def forward(
        self, wav, spk_emb=None, prosody_emb=None, spk_ids=None,
        duration_features=None, *, frame_padding_mask=None,
        rhythm_padding_mask=None, compute_ssl: bool = True,
    ):
        """由音訊與 rhythm 計算分類 logits，以及可選的 masked SSL loss。

        輸入 wav [B, L]、duration_features [B, N, 3 * sources]；
        frame_padding_mask [B, T] 與 rhythm_padding_mask [B, N] 的 True
        表示人工 padding。SSL encoder 使用反向的有效 frame mask，不接收 length。
        compute_ssl=True 時還需 spk_emb [B, 192]、prosody_emb [B, T, D]
        與 spk_ids [B]，各資料須對應同一音訊區間。

        回傳 logits [B, num_classes]、feature [B, hidden_dim]、ssl_loss、
        spk_cos 與 pros_cos；略過 SSL 時，後三項為 None。
        """
        if duration_features is None:
            raise ValueError("duration_features is required for rhythm classification")
        if compute_ssl and any(x is None for x in (spk_emb, prosody_emb, spk_ids)):
            raise ValueError("compute_ssl=True requires spk_emb, prosody_emb and spk_ids")
        z, padding = self._encode_latents(wav, frame_padding_mask)
        valid = ~padding
        ssl_loss = spk_cos = pros_cos = None
        if compute_ssl:
            expected = (z.size(0), z.size(1), self.prosody_dim)
            if tuple(prosody_emb.shape) != expected:
                raise ValueError(f"prosody_emb must have aligned shape {expected}")
            if tuple(spk_emb.shape) != (z.size(0), self.spk_dim):
                raise ValueError("spk_emb must have shape [B, 192]")
            spk_ids = torch.as_tensor(spk_ids, device=z.device)
            if tuple(spk_ids.shape) != (z.size(0),):
                raise ValueError("spk_ids must have shape [B]")
            prosody = prosody_emb.to(z).masked_fill(padding.unsqueeze(-1), 0.0)
            speaker = spk_emb.to(z)
            if not torch.isfinite(prosody).all() or not torch.isfinite(speaker).all():
                raise ValueError("Valid SSL targets must be finite")
            target = torch.cat([
                speaker.unsqueeze(1).expand(-1, z.size(1), -1),
                self.pros_ln(prosody),
            ], dim=-1)

            # Pass 1: the masked branch keeps Stage I's embedding supervision.
            mask = self._valid_span_mask(valid)
            z_masked = torch.where(mask.unsqueeze(-1), self.mask_embed.to(z), z)
            ctx_masked = self.ssl.encoder(
                z_masked, attention_mask=valid,
                output_hidden_states=False, return_dict=True,
            ).last_hidden_state
            ssl_loss, spk_cos, pros_cos = self._masked_ssl_loss(
                self.final_proj(ctx_masked), target, mask, spk_ids, valid,
            )

        # Pass 2: call the SAME encoder with clean latents. The clone isolates
        # Wav2Vec2's in-place padding operation while preserving autograd to z.
        ctx_clean = self.ssl.encoder(
            z.clone(), attention_mask=valid,
            output_hidden_states=False, return_dict=True,
        ).last_hidden_state
        feature = self.rhythm_fusion(
            ctx_clean, duration_features, padding, rhythm_padding_mask,
        )
        logits = self.cls_head(feature)
        return {
            "logits": logits, "feature": feature, "ssl_loss": ssl_loss,
            "spk_cos": spk_cos, "pros_cos": pros_cos,
        }
