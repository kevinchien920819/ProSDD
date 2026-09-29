"""以 ProSDD Stage II 為 backbone，接上 rhythm-transformer 的 duration 融合層。

backbone（CNN、共用 encoder、masked SSL loss、Stage 1 載入與 cls_head）全部
沿用 ``ProSDDStage2``；rhythm 分支與融合 decoder 照 rhythm-transformer 的
``RhythmTransformerWithDuration`` 組裝，embedding 與位置編碼直接重用
``rhythm_transformer_embedding.py``。融合層的 d_model 使用 backbone hidden_dim。

Training::

    out = model(wav, spk_emb, prosody_emb, spk_ids, duration_features=duration_features,
                frame_padding_mask=frame_mask, rhythm_padding_mask=rhythm_mask)
    loss = criterion_cls(out["logits"], labels) + beta * out["ssl_loss"]

Inference::

    out = model(wav, duration_features=duration_features,
                frame_padding_mask=frame_mask, compute_ssl=False)
"""

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from model_stage2realfake import ProSDDStage2
from rhythm_transformer_embedding import PositionalEncoding, RhythmEmbedding

RHYTHM_SOURCES = ("syllable", "vowel", "consonant")
FEATURE_SUFFIXES = ("d", "devi", "mu_diff")
RHYTHM_PAD = -100.0


class RhythmFusion(nn.Module):
    """RhythmTransformerWithDuration 去掉 SSL／分類器後的 rhythm 分支與融合 decoder。

    輸入 clean 聲學序列 memory [B,T,D]、duration [B,N,F] 與兩個 padding mask
    （True 表示 padding）；輸出融合後的 rhythm CLS 特徵 [B,D]。
    F 為每個 rhythm source 的 d／devi／mu_diff 三個統計值。
    """

    def __init__(
        self, d_model: int, rhythm_sources: Sequence[str], nhead: int,
        n_rhythm_encoder_layers: int, n_cls_encoder_layers: int,
        dropout: float, max_len: int = 5000,
    ):
        super().__init__()
        self.feature_names = tuple(
            f"{source}_{suffix}" for source in rhythm_sources for suffix in FEATURE_SUFFIXES
        )
        self.rhythm_embedding = RhythmEmbedding(len(self.feature_names), d_model, dropout)
        self.rhythm_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                dropout=dropout, activation="gelu", batch_first=True,
            ),
            num_layers=n_rhythm_encoder_layers,
            enable_nested_tensor=False,
        )
        self.cls_encoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                dropout=dropout, activation="gelu", batch_first=True,
            ),
            num_layers=n_cls_encoder_layers,
        )
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, memory, duration, memory_padding_mask, rhythm_padding_mask):
        cls = memory.new_zeros(memory.size(0), 1, memory.size(2))
        cls_mask = memory_padding_mask.new_zeros(memory.size(0), 1)

        # 聲學 memory：插入 CLS 後做位置編碼、LayerNorm、dropout。
        memory = torch.cat([cls, memory], dim=1)
        memory = self.dropout(self.layernorm(self.pos(memory)))
        memory_padding_mask = torch.cat([cls_mask, memory_padding_mask], dim=1)

        # rhythm query：padding 位置清零後 embedding，再插入 CLS。
        duration = duration.to(memory).masked_fill(rhythm_padding_mask.unsqueeze(-1), 0.0)
        rhythm = self.rhythm_embedding(*duration.unbind(-1))
        rhythm = torch.cat([cls, rhythm], dim=1)
        rhythm_padding_mask = torch.cat([cls_mask, rhythm_padding_mask], dim=1)
        rhythm = self.rhythm_encoder(rhythm, src_key_padding_mask=rhythm_padding_mask)

        # rhythm 作為 query、聲學作為 memory。
        fused = self.cls_encoder(
            tgt=rhythm, memory=memory,
            tgt_key_padding_mask=rhythm_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.layernorm(fused[:, 0, :])


class ProSDDStage2Rhythm2(ProSDDStage2):
    """兩段式 Stage II：ProSDD backbone 加 rhythm-transformer 融合層。

    ``forward`` 輸入（前五個位置參數與 ProSDDStage2 相同，其餘為 keyword-only）：
      * wav [B,L]；duration_features [B,N,3*len(rhythm_sources)]，欄位順序見
        ``duration_feature_names``；N 可與聲學 frame 數不同，整列 -100 視為 padding。
      * frame_padding_mask [B,T]／rhythm_padding_mask [B,N]：布林，True 表示人工
        padding；前者同時作為共用 encoder 的 attention mask 與融合 decoder 的 memory mask。
      * compute_ssl=True 時另需 spk_emb [B,192]、prosody_emb [B,T,prosody_dim]、
        spk_ids [B]，use_gender=True 時再加 gender_emb [B,2]。
    回傳 logits [B,num_classes]、feature [B,hidden_dim]、ssl_loss、spk_cos、pros_cos；
    略過 SSL 時後三項為 None。
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
        use_gender: bool = False,
        rhythm_sources: Sequence[str] = RHYTHM_SOURCES,
        nhead: int = 4,
        n_rhythm_encoder_layers: int = 2,
        n_cls_encoder_layers: int = 4,
        dropout: float = 0.1,
        max_position_embeddings: int = 5000,
    ):
        super().__init__(
            model_name=model_name, mask_prob=mask_prob, mask_span_len=mask_span_len,
            tau=tau, num_classes=num_classes, num_time_neg=num_time_neg,
            num_spk_neg=num_spk_neg, T_target=T_target, prosody_dim=prosody_dim,
            use_gender=use_gender,
        )
        self.classifier_pool = "rhythm"
        self.rhythm_fusion = RhythmFusion(
            self.hidden_dim, tuple(rhythm_sources), nhead,
            n_rhythm_encoder_layers, n_cls_encoder_layers, dropout, max_position_embeddings,
        )
        self.duration_feature_names = self.rhythm_fusion.feature_names
        if stage1_ckpt is not None:
            self.load_stage1(stage1_ckpt)

    def _embedding_target(self, speaker: Tensor, prosody: Tensor, gender: Optional[Tensor] = None) -> Tensor:
        """將 speaker [B,192]、prosody [B,T,D] 與可選 gender [B,2] 組成每 frame 的監督向量。

        prosody 先經 pros_ln；輸出 [B,T,spk_dim+prosody_dim+gender_dim]，供 masked pass 對比。
        """
        frames = prosody.size(1)
        targets = [speaker.unsqueeze(1).expand(-1, frames, -1), self.pros_ln(prosody)]
        if self.gender_dim:
            if gender is None:
                raise ValueError("use_gender=True requires gender_emb with shape [B, 2]")
            targets.append(gender.to(prosody).unsqueeze(1).expand(-1, frames, -1))
        return torch.cat(targets, dim=-1)

    def _load_stage1_state(self, new_state: dict):
        """將已整理的 Stage 1 權重載入 backbone、mask、projection 與 prosody LayerNorm。

        輸入為去除 module. 前綴、維度已確認的 state dict；ssl／final_proj／pros_ln
        皆以 strict 方式載入，mask_embed 直接複製。
        """
        for name in ("ssl", "final_proj", "pros_ln"):
            prefix = name + "."
            weights = {k[len(prefix):]: v for k, v in new_state.items() if k.startswith(prefix)}
            getattr(self, name).load_state_dict(weights, strict=True)
        with torch.no_grad():
            self.mask_embed.copy_(new_state["mask_embed"])
        print("Stage-1 weights loaded into Rhythm2 (ssl/mask/final_proj/pros_ln).", flush=True)

    def forward(
        self, wav, spk_emb=None, prosody_emb=None, spk_ids=None, gender_emb=None, *,
        duration_features=None, frame_padding_mask=None, rhythm_padding_mask=None, compute_ssl: bool = True,
    ):
        z = self._encode(wav)
        valid = None if frame_padding_mask is None else ~frame_padding_mask
        ssl_loss = spk_cos = pros_cos = None
        if compute_ssl:
            ssl_loss, spk_cos, pros_cos = self._masked_pass(
                z, spk_emb, prosody_emb, spk_ids, gender_emb, attention_mask=valid,
            )
        ctx_clean = self._contextualize(z, valid)

        if frame_padding_mask is None:
            frame_padding_mask = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
        if rhythm_padding_mask is None:
            rhythm_padding_mask = (duration_features == RHYTHM_PAD).all(dim=-1)
        feature = self.rhythm_fusion(ctx_clean, duration_features, frame_padding_mask, rhythm_padding_mask)
        return {
            "logits": self.cls_head(feature), "feature": feature,
            "ssl_loss": ssl_loss, "spk_cos": spk_cos, "pros_cos": pros_cos,
        }
