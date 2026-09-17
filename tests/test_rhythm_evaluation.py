import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

import main__eval_rhythm as entrypoint
from data_utils_stage2realfake_rhythm import ProSDDStage2RhythmDataset, collate_stage2_rhythm
from evaluation_metric.prosdd import load_scores
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
        self.protocol = self.root / "protocol.txt"
        self.protocol.write_text("s1 long - A01 spoof\ns2 short - - bonafide\n")
        self.csv = self.root / "durations.csv"
        self.rows = []
        for utt, seconds, starts, ends in (
            ("short", 1, [.1, .4, .8], [.2, .6, .9]),
            ("long", 6, [.2, .8, 1.2, 2., 4.8, 5.5], [.5, 1.2, 1.4, 2.4, 5.2, 5.7]),
        ):
            sf.write(self.root / f"{utt}.wav", .1 * np.sin(np.arange(seconds * 16000) * .08), 16000)
            duration = np.round(np.array(ends) - starts, 4)
            self.rows.append({
                "flac_file_name": utt + ".flac",
                "starttime_syllable": ",".join(map(str, starts)),
                "endtime_syllable": ",".join(map(str, ends)),
                "duration_syllable": ",".join(map(str, duration)),
                "duration_vowel": ",".join(map(str, duration / 2)),
                "duration_consonant": ",".join(map(str, duration / 2)),
            })
        self.write_csv()
        self.config = {
            "model_class": "ProSDDStage2Rhythm", "sample_rate": 16000, "target_samples": 64000,
            "model_name": "test-backbone", "prosody_dim": 128, "T_target": 200,
            "d_model": 8, "rhythm_sources": ["syllable", "vowel", "consonant"], "nhead": 2,
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

    def write_csv(self):
        with self.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def dataset(self, utts=("long", "short"), **kwargs):
        return entrypoint.RhythmEvalDataset(
            list(utts), self.root, self.csv, rhythm_sources=self.config["rhythm_sources"],
            T_target=200, conv_kernel=(400,), conv_stride=(320,), audio_ext=".wav", **kwargs,
        )

    def argv(self):
        return [
            "--list_path", str(self.protocol), "--wav_dir", str(self.root),
            "--duration_csv", str(self.csv), "--model_path", str(self.checkpoint),
            "--save_scores_to", str(self.root / "scores.txt"),
            "--save_metrics_to", str(self.root / "metrics.json"),
            "--audio_ext", ".wav", "--batch_size", "2", "--num_workers", "0",
        ]

    def test_audio_duration_and_masks_match_training_preprocessing(self):
        speaker = self.root / "speaker.txt"
        speaker.write_text("".join(f"{sp} " + " ".join(["0"] * 192) + "\n" for sp in ("s1", "s2")))
        prosody = self.root / "prosody.txt"
        targets = "|".join([",".join(["0"] * 128)] * 200)
        prosody.write_text("".join(f"{utt}\t{targets}\n" for utt in ("long", "short")))
        train = ProSDDStage2RhythmDataset(
            utt_ids=["long", "short"], spk_ids=["s1", "s2"], labels=[0, 1],
            wav_dir=str(self.root), spkmean_txt=str(speaker), prosody_txt=str(prosody),
            duration_csv=str(self.csv), audio_ext=".wav", T_target=200,
        )
        training_batch = collate_stage2_rhythm(
            [train[0], train[1]], T_target=200, conv_kernel=(400,), conv_stride=(320,),
        )
        evaluation = self.dataset()
        eval_batch = entrypoint.collate_rhythm_eval([evaluation[0], evaluation[1]])
        for key in ("wav", "duration_features", "frame_padding_mask", "rhythm_padding_mask"):
            torch.testing.assert_close(training_batch[key], eval_batch[key], rtol=0, atol=0)

    def test_checkpoint_to_scores_uses_rhythm_log_odds_without_ssl_targets(self):
        data = self.dataset()
        batch = entrypoint.collate_rhythm_eval([data[0], data[1]])
        inputs = {key: batch[key] for key in ("wav", "duration_features", "frame_padding_mask", "rhythm_padding_mask")}
        with torch.no_grad():
            logits = self.model(**inputs, compute_ssl=False)["logits"]
        expected = (logits[:, 1] - logits[:, 0]).tolist()
        with patch.object(entrypoint, "resolve_devices", return_value=[torch.device("cpu")]), \
                patch.object(ProSDDStage2Rhythm, "_masked_ssl_loss", side_effect=AssertionError("SSL called")):
            entrypoint.main(self.argv())
        scores = load_scores(self.root / "scores.txt")
        self.assertEqual(list(scores), ["long", "short"])
        np.testing.assert_allclose(list(scores.values()), expected, rtol=1e-6, atol=1e-7)
        metrics = json.loads((self.root / "metrics.json").read_text())
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["coverage"]["evaluation_scope"], "full_protocol")

    def test_missing_duration_and_missing_audio_report_subset_coverage(self):
        self.protocol.write_text(self.protocol.read_text() + "s1 absent - A01 spoof\ns1 broken - A01 spoof\n")
        row = dict(self.rows[0], flac_file_name="broken.flac")
        self.rows.append(row)
        self.write_csv()
        with patch.object(entrypoint, "resolve_devices", return_value=[torch.device("cpu")]):
            entrypoint.main(self.argv() + ["--skip_bad_samples"])
        coverage = json.loads((self.root / "scores.coverage.json").read_text())
        self.assertEqual(coverage["protocol_samples"], 4)
        self.assertEqual(coverage["scored_samples"], 2)
        self.assertEqual(coverage["coverage_fraction"], .5)
        self.assertEqual(set(coverage["skipped"]), {"absent", "broken"})
        metrics = json.loads((self.root / "metrics.json").read_text())
        self.assertEqual(metrics["coverage"]["evaluation_scope"], "retained_subset")
        self.assertEqual(metrics["sample_count"], 2)

    def test_checkpoint_with_missing_rhythm_weights_is_rejected(self):
        state = self.model.state_dict()
        del state["cls_head.classifier.weight"]
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            entrypoint.load_model(self.checkpoint)

    def test_module_prefixed_checkpoint_is_supported(self):
        torch.save({"state_dict": {"module." + key: value for key, value in self.model.state_dict().items()}}, self.checkpoint)
        restored, _ = entrypoint.load_model(self.checkpoint)
        for key, expected in self.model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected)

    def test_missing_duration_is_strict_by_default_and_bad_geometry_fails(self):
        with self.assertRaisesRegex(ValueError, "missing durations"):
            self.dataset(["long", "absent"])
        with self.assertRaisesRegex(ValueError, "truncates acoustic frames"):
            entrypoint.RhythmEvalDataset(
                ["long"], self.root, self.csv, rhythm_sources=["syllable"],
                T_target=1, conv_kernel=(400,), conv_stride=(320,),
            )

    def test_output_cannot_overwrite_existing_results(self):
        output = self.root / "scores.txt"
        output.write_text("existing result\n")
        with self.assertRaises(SystemExit):
            entrypoint.main(self.argv())
        self.assertEqual(output.read_text(), "existing result\n")


if __name__ == "__main__":
    unittest.main()
