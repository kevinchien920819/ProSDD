"""供模型測試共用的小型隨機初始化 Wav2Vec2。"""

from transformers import Wav2Vec2Config, Wav2Vec2Model


def tiny_backbone(*args, **kwargs):
    """接受並忽略預訓練載入參數，回傳無需下載權重的小型模型。"""
    return Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=16, conv_dim=(8,), conv_stride=(2,),
        conv_kernel=(3,), num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True,
        hidden_dropout=0.0, attention_dropout=0.0,
        activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
    ))
