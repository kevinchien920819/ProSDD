import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from model_stage1real import ProSDDStage1
from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from multi_gpu import place_model
from test_multi_gpu import tiny_backbone


def build_model(**kwargs):
    options = dict(
        T_target=16, d_model=8, nhead=2, n_rhythm_encoder_layers=1,
        n_cls_encoder_layers=2, dropout=0.0, max_position_embeddings=32,
        mask_prob=0.4, mask_span_len=2, num_time_neg=3, num_spk_neg=2,
    )
    options.update(kwargs)
    with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone) as load:
        model = ProSDDStage2Rhythm(**options)
        load.assert_called_once()
    return model


def inputs(model, batch_size=3):
    return dict(
        wav=torch.randn(batch_size, 31),  # Tiny CNN produces 15 real frames.
        spk_emb=torch.randn(batch_size, 192),
        prosody_emb=torch.randn(batch_size, model.T_target, model.prosody_dim),
        spk_ids=torch.arange(batch_size),
        duration_features=torch.rand(batch_size, 4, len(model.duration_feature_names)),
    )


class Stage2RhythmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_two_passes_share_one_backbone_and_keep_both_gradient_paths(self):
        model = build_model()
        batch = inputs(model)
        calls = []

        def record_encoder(module, args, kwargs):
            calls.append((args[0].detach().clone(), kwargs["attention_mask"].clone()))

        hook = model.ssl.encoder.register_forward_pre_hook(record_encoder, with_kwargs=True)
        self.addCleanup(hook.remove)
        with patch.object(model.ssl.feature_extractor, "forward",
                          wraps=model.ssl.feature_extractor.forward) as extract:
            out = model(**batch)
            self.assertEqual(extract.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(out["logits"].shape, (3, 2))
        self.assertEqual(out["feature"].shape, (3, 8))
        self.assertIsInstance(model.cls_head.classifier, nn.Linear)
        for _, valid in calls:
            self.assertTrue(valid[:, :15].all())
            self.assertFalse(valid[:, 15:].any())
        changed = (calls[0][0] != calls[1][0]).any(dim=-1)
        self.assertTrue(changed[:, :15].any())
        self.assertFalse(changed[:, 15:].any())

        labels = torch.tensor([0, 1, 0])
        cls_loss = F.cross_entropy(out["logits"], labels)
        params = (
            model.ssl.feature_extractor.conv_layers[0].conv.weight,
            model.ssl.encoder.layers[0].attention.q_proj.weight,
            model.final_proj.weight,
            model.cls_head.rhythm_embedding[0].weight,
            model.cls_head.classifier.weight,
        )
        cls_grads = torch.autograd.grad(cls_loss, params, retain_graph=True, allow_unused=True)
        ssl_grads = torch.autograd.grad(out["ssl_loss"], params, retain_graph=True, allow_unused=True)
        for index in (0, 1):
            self.assertGreater(cls_grads[index].abs().sum().item(), 0)
            self.assertGreater(ssl_grads[index].abs().sum().item(), 0)
        self.assertIsNone(cls_grads[2])
        self.assertGreater(ssl_grads[2].abs().sum().item(), 0)
        for index in (3, 4):
            self.assertGreater(cls_grads[index].abs().sum().item(), 0)
            self.assertIsNone(ssl_grads[index])

        before = [p.detach().clone() for p in params]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        (cls_loss + out["ssl_loss"]).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        for old, new in zip(before, params):
            self.assertFalse(torch.equal(old, new))

    def test_inference_uses_only_clean_pass_and_needs_no_ssl_targets(self):
        model = build_model().eval()
        batch = inputs(model)
        with patch.object(model.ssl.encoder, "forward", wraps=model.ssl.encoder.forward) as encode:
            with torch.no_grad():
                out = model(batch["wav"], duration_features=batch["duration_features"], compute_ssl=False)
            self.assertEqual(encode.call_count, 1)
        self.assertIsNone(out["ssl_loss"])
        self.assertIsNone(out["spk_cos"])
        self.assertIsNone(out["pros_cos"])
        with torch.no_grad():
            validation = model(**batch)
        torch.testing.assert_close(out["logits"], validation["logits"], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(validation["ssl_loss"]))

    def test_feature_mapping_matches_tensor_and_vowel_only_needs_no_other_sources(self):
        model = build_model(rhythm_sources=("vowel",)).eval()
        self.assertEqual(model.duration_feature_names, ("vowel_d", "vowel_devi", "vowel_mu_diff"))
        batch = inputs(model)
        features = batch["duration_features"]
        mapped = {name: features[:, :, index] for index, name in enumerate(model.duration_feature_names)}
        tensor_out = model(batch["wav"], duration_features=features, compute_ssl=False)
        mapped_out = model(batch["wav"], duration_features=mapped, compute_ssl=False)
        torch.testing.assert_close(tensor_out["logits"], mapped_out["logits"], rtol=0, atol=0)
        changed = model(batch["wav"], duration_features=features * 2, compute_ssl=False)
        self.assertFalse(torch.allclose(tensor_out["logits"], changed["logits"]))

    def test_padded_duration_values_and_extra_padding_do_not_change_prediction(self):
        model = build_model().eval()
        batch = inputs(model)
        features = batch["duration_features"]
        features[0, 2:] = -100.0
        inferred = model(batch["wav"], duration_features=features, compute_ssl=False)
        padding = (features == -100.0).all(dim=-1)
        explicit_features = features.masked_fill(padding.unsqueeze(-1), torch.nan)
        explicit = model(batch["wav"], duration_features=explicit_features,
                         rhythm_padding_mask=padding, compute_ssl=False)
        torch.testing.assert_close(inferred["logits"], explicit["logits"], rtol=0, atol=0)
        extended_features = torch.cat([features, torch.full((3, 3, features.size(2)), -100.0)], dim=1)
        extended = model(batch["wav"], duration_features=extended_features, compute_ssl=False)
        torch.testing.assert_close(inferred["logits"], extended["logits"], rtol=1e-5, atol=1e-6)

        features = features.clone().requires_grad_()
        out = model(batch["wav"], duration_features=features, compute_ssl=False)
        F.cross_entropy(out["logits"], torch.tensor([0, 1, 0])).backward()
        self.assertEqual(features.grad[padding].abs().sum().item(), 0)
        self.assertGreater(features.grad[~padding].abs().sum().item(), 0)

    def test_padding_is_excluded_from_ssl_anchors_and_negative_targets(self):
        model = build_model().eval()
        batch = inputs(model)
        padding = torch.zeros(3, 16, dtype=torch.bool)
        padding[0, :4] = True  # Include left padding, as used by center padding.
        padding[1, 8:] = True
        padding[:, 15:] = True  # Also appended automatically by the model.
        batch["frame_padding_mask"] = padding
        torch.manual_seed(19)
        out = model(**batch)
        batch["prosody_emb"] = batch["prosody_emb"].masked_fill(padding.unsqueeze(-1), torch.nan)
        torch.manual_seed(19)
        changed = model(**batch)
        for name in ("logits", "ssl_loss", "spk_cos", "pros_cos"):
            torch.testing.assert_close(out[name], changed[name], rtol=0, atol=0)

    def test_single_valid_frame_and_no_negative_candidates_are_finite(self):
        model = build_model()
        batch = inputs(model, batch_size=1)
        padding = torch.ones(1, 16, dtype=torch.bool)
        padding[:, 5] = False
        out = model(**batch, frame_padding_mask=padding)
        self.assertEqual(out["ssl_loss"].item(), 0.0)
        loss = F.cross_entropy(out["logits"], torch.tensor([0])) + out["ssl_loss"]
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_zero_mask_probability_produces_differentiable_zero_ssl_loss(self):
        model = build_model(mask_prob=0.0)
        out = model(**inputs(model))
        self.assertEqual(out["ssl_loss"].item(), 0.0)
        out["ssl_loss"].backward()
        self.assertIsNotNone(model.final_proj.weight.grad)

    def test_invalid_inputs_fail_explicitly(self):
        model = build_model()
        batch = inputs(model)
        with self.assertRaisesRegex(ValueError, "duration_features is required"):
            model(batch["wav"], compute_ssl=False)
        with self.assertRaisesRegex(ValueError, "compute_ssl=True requires"):
            model(batch["wav"], duration_features=batch["duration_features"])
        for features, message in (
            ({}, "Missing duration"),
            (torch.full((3, 2, 9), -100.0), "at least one valid duration"),
            (torch.zeros(3, 4, 3), "Expected duration shape"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                model(batch["wav"], duration_features=features, compute_ssl=False)
        partial = batch["duration_features"].clone()
        partial[0, 0, 0] = -100.0
        with self.assertRaisesRegex(ValueError, "contain no -100 padding"):
            model(batch["wav"], duration_features=partial, compute_ssl=False)
        with self.assertRaisesRegex(ValueError, "aligned shape"):
            model(**{**batch, "prosody_emb": batch["prosody_emb"][:, :8]})
        with self.assertRaisesRegex(ValueError, "at least one valid acoustic"):
            model(**batch, frame_padding_mask=torch.ones(3, 16, dtype=torch.bool))

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
                out = model(**inputs(model))
                self.assertTrue(torch.isfinite(out["ssl_loss"]))
                with self.assertRaisesRegex(ValueError, "Prosody dim mismatch"):
                    build_model(prosody_dim=256 if dim == 128 else 128, stage1_ckpt=str(path))

    def test_model_checkpoint_round_trip_and_layer_placement_preserve_outputs(self):
        model = build_model().eval()
        restored = build_model().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        place_model(restored, [torch.device("cpu"), torch.device("cpu:0")])
        batch = inputs(model)
        torch.manual_seed(31)
        original = model(**batch)
        torch.manual_seed(31)
        candidate = restored(**batch)
        for name in original:
            torch.testing.assert_close(original[name], candidate[name], rtol=0, atol=0)

    def test_cpu_autocast_forward_and_joint_backward_are_finite(self):
        model = build_model()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**inputs(model))
            loss = F.cross_entropy(out["logits"], torch.tensor([0, 1, 0])) + out["ssl_loss"]
        self.assertEqual(out["logits"].dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
