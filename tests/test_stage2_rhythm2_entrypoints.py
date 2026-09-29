"""Rhythm2 訓練／評估入口：重用既有 Rhythm 流程，只替換模型類別。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

import main__eval_rhythm as evaluation
import main_stage2realfake_rhythm as rhythm_entrypoint
import main_stage2realfake_rhythm2 as entrypoint
from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from model_stage2realfake_rhythm2 import ProSDDStage2Rhythm2
from test_stage2_rhythm2 import OPTIONS, build_model, inputs, tiny_backbone


class Rhythm2TrainingEntrypointTests(unittest.TestCase):
    def test_main_delegates_to_the_shared_rhythm_flow_with_the_new_model(self):
        with patch.object(rhythm_entrypoint, "main") as shared:
            entrypoint.main(["--epochs", "1"])
        shared.assert_called_once_with(["--epochs", "1"], model_cls=ProSDDStage2Rhythm2)

    def test_shared_main_records_the_model_class_it_was_given(self):
        self.assertEqual(rhythm_entrypoint.main.__kwdefaults__["model_cls"], ProSDDStage2Rhythm)

    def test_every_parameter_belongs_to_one_optimizer_group(self):
        model = build_model()
        args = SimpleNamespace(lr_ssl_backbone=1e-6, lr_ssl_head=1e-4, lr_rhythm=1e-5, lr_cls=1e-5, weight_decay=0.0)
        optimizer = rhythm_entrypoint.build_optimizer(model, args)
        grouped = sum(len(group["params"]) for group in optimizer.param_groups)
        self.assertEqual(grouped, sum(1 for p in model.parameters() if p.requires_grad))
        names = {group["name"]: len(group["params"]) for group in optimizer.param_groups}
        self.assertGreater(names["rhythm"], 0)
        self.assertGreater(names["cls"], 0)

    def test_rhythm_learning_rate_updates_only_the_fusion_module(self):
        torch.manual_seed(3)
        model = build_model().eval()
        args = SimpleNamespace(lr_ssl_backbone=0.0, lr_ssl_head=0.0, lr_rhythm=1e-3, lr_cls=0.0, weight_decay=0.0)
        optimizer = rhythm_entrypoint.build_optimizer(model, args)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        out = model(**inputs(model))
        F.cross_entropy(out["logits"], torch.tensor([0, 1, 0])).backward()
        optimizer.step()
        changed = {name for name, p in model.named_parameters() if not torch.equal(before[name], p)}
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("rhythm_fusion.") for name in changed))


class Rhythm2EvaluationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        torch.manual_seed(9)
        self.model = build_model(T_target=None).eval()
        torch.save(self.model.state_dict(), self.root / "model_best.pth")
        self.config = {
            "model_class": "ProSDDStage2Rhythm2", "model_name": "tiny", "prosody_dim": 128,
            "T_target": None, "rhythm_sources": ["syllable", "vowel", "consonant"],
            "sample_rate": 16000, "audio_mode": "full_utterance", "audio_seconds": 0,
            "target_samples": None,
            **{key: OPTIONS[key] for key in ("nhead", "n_rhythm_encoder_layers", "n_cls_encoder_layers",
                                              "dropout", "max_position_embeddings")},
        }

    def write_config(self, **overrides):
        path = self.root / "config.json"
        path.write_text(json.dumps({**self.config, **overrides}), encoding="utf-8")
        return path

    def test_load_model_restores_rhythm2_from_its_training_config(self):
        self.write_config()
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone):
            model, config = evaluation.load_model(self.root / "model_best.pth")
        self.assertIsInstance(model, ProSDDStage2Rhythm2)
        self.assertFalse(model.training)
        self.assertEqual(config["model_class"], "ProSDDStage2Rhythm2")
        batch = inputs(self.model, frames=15)
        with torch.no_grad():
            expected = self.model(batch["wav"], duration_features=batch["duration_features"], compute_ssl=False)
            actual = model(batch["wav"], duration_features=batch["duration_features"], compute_ssl=False)
        torch.testing.assert_close(expected["logits"], actual["logits"], rtol=0, atol=0)

    def test_load_model_rejects_unknown_model_classes(self):
        self.write_config(model_class="SomethingElse")
        with self.assertRaisesRegex(ValueError, "model_class"):
            evaluation.load_model(self.root / "model_best.pth")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
