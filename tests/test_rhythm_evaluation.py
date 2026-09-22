import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

import main__eval_rhythm as entrypoint
from model_stage2realfake_rhythm import ProSDDStage2Rhythm


def tiny_backbone(*args, **kwargs):
    return Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, conv_dim=(8,), conv_stride=(320,), conv_kernel=(400,),
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=True, hidden_dropout=0.0, attention_dropout=0.0,
        activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
    ))


class RhythmEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {
            "model_class": "ProSDDStage2Rhythm", "sample_rate": 16000, "target_samples": 64000,
            "model_name": "test-backbone", "prosody_dim": 128, "T_target": 200,
            "rhythm_sources": ["syllable", "vowel", "consonant"], "nhead": 2,
            "n_rhythm_encoder_layers": 1, "n_cls_encoder_layers": 1, "dropout": 0.,
            "max_position_embeddings": 201,
        }
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.checkpoint = self.root / "model_epoch_1.pth"
        self.backbone_patch = patch("model_stage2realfake.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone)
        self.backbone_patch.start()
        self.addCleanup(self.backbone_patch.stop)
        self.model = ProSDDStage2Rhythm(**{
            key: value for key, value in self.config.items()
            if key not in ("model_class", "sample_rate", "target_samples")
        }).eval()
        torch.save(self.model.state_dict(), self.checkpoint)

    def test_checkpoint_with_missing_rhythm_weights_is_rejected(self):
        state = self.model.state_dict()
        del state["rhythm_fusion.rhythm_embedding.0.weight"]
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            entrypoint.load_model(self.checkpoint)

    def test_module_prefixed_checkpoint_is_supported(self):
        torch.save({"state_dict": {"module." + key: value for key, value in self.model.state_dict().items()}}, self.checkpoint)
        restored, _ = entrypoint.load_model(self.checkpoint)
        for key, expected in self.model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected)


if __name__ == "__main__":
    unittest.main()
