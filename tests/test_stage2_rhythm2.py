"""Rhythm2：ProSDDStage2 backbone 接 rhythm-transformer 融合層的模型測試。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model

from model_stage1real import ProSDDStage1
from model_stage2realfake import ProSDDStage2
from model_stage2realfake_rhythm2 import ProSDDStage2Rhythm2, RhythmFusion
from rhythm_transformer_embedding import PositionalEncoding, RhythmEmbedding


def tiny_backbone(*args, **kwargs):
    """接受並忽略預訓練載入參數，回傳無需下載權重的小型 Wav2Vec2。"""
    return Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=16, conv_dim=(8,), conv_stride=(2,),
        conv_kernel=(3,), num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True,
        hidden_dropout=0.0, attention_dropout=0.0,
        activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
    ))


OPTIONS = dict(
    T_target=16, nhead=2, n_rhythm_encoder_layers=1, n_cls_encoder_layers=2,
    dropout=0.0, max_position_embeddings=32, mask_prob=0.4, mask_span_len=2,
    num_time_neg=3, num_spk_neg=2,
)


def build_model(**kwargs):
    options = {**OPTIONS, **kwargs}
    with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone) as load:
        model = ProSDDStage2Rhythm2(**options)
        load.assert_called_once()
    return model


def inputs(model, batch_size=3, frames=None):
    frames = frames or model.T_target
    return dict(
        wav=torch.randn(batch_size, 31),  # 小型 CNN 產生 15 個真實 frame。
        spk_emb=torch.randn(batch_size, 192),
        prosody_emb=torch.randn(batch_size, frames, model.prosody_dim),
        spk_ids=torch.arange(batch_size),
        duration_features=torch.rand(batch_size, 4, len(model.duration_feature_names)),
    )


class BaselineForwardSplitTests(unittest.TestCase):
    """拆出 _encode／_contextualize／_masked_pass 後 baseline 行為不變。"""

    def setUp(self):
        torch.manual_seed(5)
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone):
            self.model = ProSDDStage2(T_target=16, mask_prob=0.4, mask_span_len=2,
                                      num_time_neg=3, num_spk_neg=2)

    def test_forward_keeps_output_contract_and_fixed_frame_count(self):
        z = self.model._encode(torch.randn(2, 31))
        self.assertEqual(z.shape, (2, 16, 8))
        self.assertEqual(z[:, 15].abs().sum().item(), 0.0)  # 第 16 個 frame 是補零。
        with patch.object(self.model.ssl.encoder, "forward", wraps=self.model.ssl.encoder.forward) as encode:
            out = self.model(torch.randn(2, 31), torch.randn(2, 192), torch.randn(2, 16, 128), torch.tensor([0, 1]))
        self.assertEqual(encode.call_count, 2)
        self.assertEqual(sorted(out), ["logits", "pros_cos", "spk_cos", "ssl_loss"])
        self.assertEqual(out["logits"].shape, (2, 2))
        self.assertTrue(torch.isfinite(out["ssl_loss"]))

    def test_none_target_keeps_actual_frames(self):
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone):
            model = ProSDDStage2(T_target=None)
        self.assertIsNone(model.T_target)
        self.assertEqual(model._encode(torch.randn(2, 31)).shape, (2, 15, 8))


class Stage2Rhythm2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_reuses_rhythm_transformer_modules_at_backbone_width(self):
        model = build_model(rhythm_sources=("vowel", "consonant"))
        fusion = model.rhythm_fusion
        self.assertIsInstance(fusion, RhythmFusion)
        self.assertIsInstance(fusion.rhythm_embedding, RhythmEmbedding)
        self.assertIsInstance(fusion.pos, PositionalEncoding)
        self.assertEqual(model.duration_feature_names, (
            "vowel_d", "vowel_devi", "vowel_mu_diff",
            "consonant_d", "consonant_devi", "consonant_mu_diff",
        ))
        self.assertEqual((fusion.rhythm_embedding.linear.in_features,
                          fusion.rhythm_embedding.linear.out_features), (6, model.hidden_dim))
        self.assertIsInstance(fusion.rhythm_encoder, nn.TransformerEncoder)
        self.assertIsInstance(fusion.cls_encoder, nn.TransformerDecoder)
        self.assertEqual(len(fusion.rhythm_encoder.layers), 1)
        self.assertEqual(len(fusion.cls_encoder.layers), 2)

    def test_fused_feature_feeds_the_baseline_classifier(self):
        model = build_model(num_classes=3).eval()
        out = model(**inputs(model))
        self.assertEqual(out["feature"].shape, (3, model.hidden_dim))
        self.assertEqual(out["logits"].shape, (3, 3))
        self.assertEqual([type(layer) for layer in model.cls_head],
                         [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear])
        self.assertEqual((model.cls_head[0].in_features, model.cls_head[0].out_features),
                         (model.hidden_dim, 512))
        torch.testing.assert_close(out["logits"], model.cls_head(out["feature"]))
        self.assertEqual(model.classifier_pool, "rhythm")
        self.assertFalse(hasattr(model, "attn"))

    def test_two_passes_share_one_backbone_and_receive_the_valid_frame_mask(self):
        model = build_model()
        batch = inputs(model)
        padding = torch.zeros(3, 16, dtype=torch.bool)
        padding[1, 10:] = True
        calls = []

        def record(module, args, kwargs):
            calls.append((args[0].detach().clone(), kwargs["attention_mask"]))

        hook = model.ssl.encoder.register_forward_pre_hook(record, with_kwargs=True)
        self.addCleanup(hook.remove)
        with patch.object(model.ssl.feature_extractor, "forward",
                          wraps=model.ssl.feature_extractor.forward) as extract:
            out = model(**batch, frame_padding_mask=padding)
            self.assertEqual(extract.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(out["logits"].shape, (3, 2))
        for _, valid in calls:
            self.assertTrue(torch.equal(valid, ~padding))
        self.assertTrue((calls[0][0] != calls[1][0]).any())  # masked 與 clean latents 不同。
        self.assertTrue(torch.isfinite(out["ssl_loss"]))

    def test_omitting_frame_mask_matches_baseline_encoder_call(self):
        model = build_model()
        with patch.object(model.ssl.encoder, "forward", wraps=model.ssl.encoder.forward) as encode:
            model(**inputs(model))
        for call in encode.call_args_list:
            self.assertIsNone(call.kwargs["attention_mask"])

    def test_inference_uses_only_clean_pass_and_needs_no_ssl_targets(self):
        model = build_model().eval()
        batch = inputs(model)
        with patch.object(model.ssl.encoder, "forward", wraps=model.ssl.encoder.forward) as encode:
            with torch.no_grad():
                out = model(batch["wav"], duration_features=batch["duration_features"], compute_ssl=False)
            self.assertEqual(encode.call_count, 1)
        for name in ("ssl_loss", "spk_cos", "pros_cos"):
            self.assertIsNone(out[name])
        with torch.no_grad():
            validation = model(**batch)
        torch.testing.assert_close(out["logits"], validation["logits"], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(validation["ssl_loss"]))

    def test_classification_and_ssl_losses_reach_their_own_parameters(self):
        model = build_model()
        out = model(**inputs(model))
        cls_loss = F.cross_entropy(out["logits"], torch.tensor([0, 1, 0]))
        params = (
            model.ssl.feature_extractor.conv_layers[0].conv.weight,
            model.ssl.encoder.layers[0].attention.q_proj.weight,
            model.final_proj.weight,
            model.rhythm_fusion.rhythm_embedding.linear.weight,
            model.rhythm_fusion.cls_encoder.layers[0].multihead_attn.in_proj_weight,
            model.cls_head[-1].weight,
        )
        cls_grads = torch.autograd.grad(cls_loss, params, retain_graph=True, allow_unused=True)
        ssl_grads = torch.autograd.grad(out["ssl_loss"], params, retain_graph=True, allow_unused=True)
        for index in (0, 1):
            self.assertGreater(cls_grads[index].abs().sum().item(), 0)
            self.assertGreater(ssl_grads[index].abs().sum().item(), 0)
        self.assertIsNone(cls_grads[2])
        self.assertGreater(ssl_grads[2].abs().sum().item(), 0)
        for index in (3, 4, 5):
            self.assertGreater(cls_grads[index].abs().sum().item(), 0)
            self.assertIsNone(ssl_grads[index])

    def test_padded_frames_do_not_influence_prediction(self):
        model = build_model().eval()
        batch = inputs(model)
        padding = torch.zeros(3, 16, dtype=torch.bool)
        padding[0, :4] = True  # 置中補零會產生左側 padding。
        padding[1, 8:] = True
        padding[:, 15:] = True
        reference = model(batch["wav"], duration_features=batch["duration_features"],
                          frame_padding_mask=padding, compute_ssl=False)
        clean_encode = model._encode

        def noisy_encode(wav):
            z = clean_encode(wav)
            return z + 5.0 * torch.randn_like(z) * padding.unsqueeze(-1)

        with patch.object(model, "_encode", side_effect=noisy_encode):
            noisy = model(batch["wav"], duration_features=batch["duration_features"],
                          frame_padding_mask=padding, compute_ssl=False)
        torch.testing.assert_close(reference["logits"], noisy["logits"], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.equal(padding, torch.zeros_like(padding)))

    def test_padded_duration_tokens_do_not_influence_prediction(self):
        model = build_model().eval()
        batch = inputs(model)
        features = batch["duration_features"]
        features[0, 2:] = -100.0
        inferred = model(batch["wav"], duration_features=features, compute_ssl=False)
        padding = (features == -100.0).all(dim=-1)
        garbage = features.masked_fill(padding.unsqueeze(-1), 123.0)
        explicit = model(batch["wav"], duration_features=garbage, rhythm_padding_mask=padding, compute_ssl=False)
        torch.testing.assert_close(inferred["logits"], explicit["logits"], rtol=0, atol=0)
        extended = torch.cat([features, torch.full((3, 3, features.size(2)), -100.0)], dim=1)
        longer = model(batch["wav"], duration_features=extended, compute_ssl=False)
        torch.testing.assert_close(inferred["logits"], longer["logits"], rtol=1e-5, atol=1e-6)
        changed = model(batch["wav"], duration_features=features * 2, compute_ssl=False)
        self.assertFalse(torch.allclose(inferred["logits"], changed["logits"]))

        features = features.clone().requires_grad_()
        out = model(batch["wav"], duration_features=features, compute_ssl=False)
        F.cross_entropy(out["logits"], torch.tensor([0, 1, 0])).backward()
        self.assertEqual(features.grad[padding].abs().sum().item(), 0)
        self.assertGreater(features.grad[~padding].abs().sum().item(), 0)

    def test_none_target_uses_actual_cnn_frames(self):
        model = build_model(T_target=None).eval()
        self.assertIsNone(model.T_target)
        batch = inputs(model, frames=15)
        padding = torch.zeros(3, 15, dtype=torch.bool)
        padding[2, 12:] = True
        out = model(**batch, frame_padding_mask=padding)
        self.assertEqual(out["feature"].shape, (3, model.hidden_dim))
        self.assertTrue(torch.isfinite(out["ssl_loss"]))

    def test_stage1_checkpoint_transfers_backbone_and_target_normalization(self):
        for dim, wrapped in ((128, False), (256, True)):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory() as directory:
                with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone):
                    stage1 = ProSDDStage1(prosody_dim=dim)
                with torch.no_grad():
                    stage1.pros_ln.weight.fill_(1.7)
                    stage1.pros_ln.bias.fill_(0.4)
                state = stage1.state_dict()
                saved = {"state_dict": {"module." + k: v for k, v in state.items()}} if wrapped else state
                path = Path(directory) / "stage1.pth"
                torch.save(saved, path)
                model = build_model(prosody_dim=dim, stage1_ckpt=str(path))
                for name, value in state.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(model(**inputs(model))["ssl_loss"]))
                with self.assertRaisesRegex(ValueError, "Prosody dim mismatch"):
                    build_model(prosody_dim=256 if dim == 128 else 128, stage1_ckpt=str(path))

    def test_target_and_checkpoint_helpers_are_defined_locally_not_inherited(self):
        for name in ("_embedding_target", "_load_stage1_state", "forward"):
            self.assertIn(name, ProSDDStage2Rhythm2.__dict__)

    def test_gender_target_is_appended_per_frame(self):
        model = build_model(use_gender=True)
        self.assertEqual(model.gender_dim, 2)
        gender = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        target = model._embedding_target(torch.randn(3, 192), torch.randn(3, 16, model.prosody_dim), gender)
        self.assertEqual(target.shape, (3, 16, model.out_dim))
        torch.testing.assert_close(target[:, :, -2:], gender.unsqueeze(1).expand(-1, 16, -1))
        out = model(**inputs(model), gender_emb=gender)
        self.assertTrue(torch.isfinite(out["ssl_loss"]))

    def test_state_dict_round_trip_preserves_outputs(self):
        model = build_model().eval()
        restored = build_model().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        batch = inputs(model)
        torch.manual_seed(31)
        original = model(**batch)
        torch.manual_seed(31)
        candidate = restored(**batch)
        for name in original:
            torch.testing.assert_close(original[name], candidate[name], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
