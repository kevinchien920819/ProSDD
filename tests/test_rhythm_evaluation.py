from contextlib import redirect_stderr, redirect_stdout
import csv
from functools import partial
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Config, Wav2Vec2Model

import main__eval_rhythm as entrypoint
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

    def test_current_training_audio_policies_restore_the_requested_frames(self):
        policies = (
            ("pause_crop", 2.0, 32000, None),
            ("full_utterance", 0.0, None, 100),
        )
        for mode, seconds, samples, frames in policies:
            with self.subTest(mode=mode):
                config = {**self.config, "audio_mode": mode, "audio_seconds": seconds,
                          "target_samples": samples, "T_target": frames}
                (self.root / "config.json").write_text(json.dumps(config))

                restored, _ = entrypoint.load_model(self.checkpoint)

                self.assertEqual(restored.T_target, frames)

    def prepare_evaluation(self, *, mode="pause_crop", seconds=0.4, frames=None):
        """建立真正的音訊、CSV、設定與已知 log-odds 的小型 checkpoint。"""
        self.config.update(audio_mode=mode, audio_seconds=seconds, T_target=frames,
                           target_samples=round(seconds * 16000) if seconds else None)
        (self.root / "config.json").write_text(json.dumps(self.config))
        with torch.no_grad():
            self.model.cls_head[-1].weight.zero_()
            self.model.cls_head[-1].bias.copy_(torch.tensor([1.0, 3.0]))
        torch.save(self.model.state_dict(), self.checkpoint)
        self.protocol = self.root / "eval.protocol.txt"
        self.protocol.write_text("sp real - - bonafide\nsp fake - A01 spoof\n")
        self.duration_csv = self.root / "duration.csv"
        self.rows = []
        for utt, length in (("real", 4800), ("fake", 9600)):
            sf.write(self.root / f"{utt}.wav", np.sin(np.arange(length) * 0.07), 16000)
            self.rows.append({
                "flac_file_name": f"{utt}.wav",
                "starttime_word": "0.05,0.17", "endtime_word": "0.12,0.25",
                "starttime_syllable": "0.05,0.17", "endtime_syllable": "0.12,0.25",
                "duration_syllable": "0.07,0.08", "duration_vowel": "0.03,0.04",
                "duration_consonant": "0.04,0.04",
            })
        self.write_duration_csv()

    def write_duration_csv(self):
        with self.duration_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def cli_args(self, name="scores", *extra):
        scores = self.root / "results" / f"{name}.txt"
        argv = [
            "--list_path", str(self.protocol), "--wav_dir", str(self.root),
            "--duration_csv", str(self.duration_csv), "--model_path", str(self.checkpoint),
            "--save_scores_to", str(scores), "--audio_ext", ".wav", "--num_workers", "0",
            *extra,
        ]
        return scores, argv

    def run_cli(self, name="scores", *extra):
        scores, argv = self.cli_args(name, *extra)
        with patch("torch.cuda.is_available", return_value=False), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            entrypoint.main(argv)
        return scores

    def test_cli_scores_current_checkpoint_without_ssl_targets(self):
        self.prepare_evaluation()

        scores = self.run_cli()

        self.assertEqual(load_scores(scores), {"real": 2.0, "fake": 2.0})

    def test_metrics_describe_only_the_successfully_scored_subset(self):
        self.prepare_evaluation()
        with self.protocol.open("a") as stream:
            stream.write("sp missing_duration - - bonafide\nsp broken - A01 spoof\n")
        self.rows.append({**self.rows[0], "flac_file_name": "broken.wav"})
        self.write_duration_csv()
        metrics_path = self.root / "metrics.json"

        self.run_cli("subset", "--skip_bad_samples", "--save_metrics_to", str(metrics_path))

        metrics = json.loads(metrics_path.read_text())
        self.assertEqual((metrics["sample_count"], metrics["coverage"]["evaluation_scope"],
                          metrics["coverage"]["coverage_fraction"]), (2, "retained_subset", 0.5))

    def dataset(self, **options):
        return entrypoint.RhythmEvalDataset(
            ["real", "fake"], self.root, self.duration_csv,
            rhythm_sources=self.config["rhythm_sources"], audio_seconds=self.config["audio_seconds"],
            T_target=self.config["T_target"], conv_kernel=(400,), conv_stride=(320,),
            audio_ext="wav", **options,
        )

    def test_center_padding_excludes_only_artificial_audio_frames(self):
        self.prepare_evaluation()
        batch = entrypoint.collate_rhythm_eval([self.dataset()[0]], conv_kernel=(400,), conv_stride=(320,))

        self.assertEqual(batch["frame_padding_mask"].tolist(), [[True] * 3 + [False] * 14 + [True] * 2])

    def test_full_utterance_padding_retains_every_real_audio_frame(self):
        self.prepare_evaluation(mode="full_utterance", seconds=0)
        dataset = self.dataset()
        batch = entrypoint.collate_rhythm_eval([dataset[0], dataset[1]], conv_kernel=(400,), conv_stride=(320,))

        self.assertEqual(batch["frame_padding_mask"].tolist(), [[False] * 14 + [True] * 15, [False] * 29])

    def test_explicit_frames_pad_or_truncate_like_the_checkpoint(self):
        self.prepare_evaluation(mode="full_utterance", seconds=0)
        sample = self.dataset()[0]
        for frames in (10, 20):
            with self.subTest(frames=frames):
                batch = entrypoint.collate_rhythm_eval(
                    [sample], T_target=frames, conv_kernel=(400,), conv_stride=(320,),
                )
                expected = [False] * min(14, frames) + [True] * max(0, frames - 14)
                self.assertEqual(batch["frame_padding_mask"].tolist(), [expected])

    def test_rhythm_padding_has_an_independent_mask(self):
        self.prepare_evaluation(mode="full_utterance", seconds=0)
        for key in self.rows[1]:
            if key != "flac_file_name":
                self.rows[1][key] = self.rows[1][key].split(",")[0]
        self.write_duration_csv()
        dataset = self.dataset()
        batch = entrypoint.collate_rhythm_eval([dataset[0], dataset[1]], conv_kernel=(400,), conv_stride=(320,))

        self.assertEqual(batch["rhythm_padding_mask"].tolist(), [[False, False], [False, True]])

    def test_crop_is_unchanged_by_access_order_or_worker_count(self):
        self.prepare_evaluation()
        dataset = self.dataset(seed=81)
        expected = dataset[1]["wav"].clone()
        dataset[0]
        loader = DataLoader(dataset, batch_size=1, num_workers=1, multiprocessing_context="spawn",
                            collate_fn=partial(entrypoint.collate_rhythm_eval, conv_kernel=(400,), conv_stride=(320,)))
        actual = {batch["utt_ids"][0]: batch["wav"][0] for batch in loader}

        torch.testing.assert_close(actual["fake"], expected, rtol=0, atol=0)

    def test_full_utterance_cli_accepts_explicit_frame_target(self):
        self.prepare_evaluation(mode="full_utterance", seconds=0, frames=20)

        scores = self.run_cli("full", "--max_batch_samples", "0")

        self.assertEqual(load_scores(scores), {"real": 2.0, "fake": 2.0})

    def test_coverage_identifies_both_missing_duration_and_broken_audio(self):
        self.prepare_evaluation()
        self.rows = [self.rows[0]]
        self.write_duration_csv()
        (self.root / "real.wav").unlink()

        with self.assertRaisesRegex(ValueError, "No usable"):
            self.run_cli("empty", "--skip_bad_samples", "--max_batch_samples", "0")

        report = json.loads((self.root / "results/empty.coverage.json").read_text())
        self.assertEqual(set(report["skipped"]), {"real", "fake"})

    def test_strict_duration_failure_names_the_utterance(self):
        self.prepare_evaluation()
        self.rows = [self.rows[0]]
        self.write_duration_csv()

        with self.assertRaisesRegex(ValueError, "fake: Missing rhythm data"):
            self.run_cli()

    def test_retained_single_class_cannot_produce_metrics(self):
        self.prepare_evaluation()
        (self.root / "fake.wav").unlink()

        with self.assertRaisesRegex(ValueError, "both bonafide and spoof"):
            self.run_cli("single", "--skip_bad_samples", "--save_metrics_to", str(self.root / "metrics.json"))

    def test_metrics_output_cannot_overwrite_the_input_protocol(self):
        self.prepare_evaluation()
        original = self.protocol.read_bytes()

        with self.assertRaises(SystemExit):
            self.run_cli("protected", "--save_metrics_to", str(self.protocol))

        self.assertEqual(self.protocol.read_bytes(), original)

    def test_existing_scores_are_preserved(self):
        self.prepare_evaluation()
        path = self.run_cli()
        original = path.read_bytes()

        with self.assertRaises(SystemExit):
            self.run_cli()

        self.assertEqual(path.read_bytes(), original)

    def test_real_cli_with_local_backbone_and_worker_writes_metrics(self):
        self.prepare_evaluation()
        backbone = self.root / "backbone"
        self.model.ssl.save_pretrained(backbone)
        self.config["model_name"] = str(backbone)
        (self.root / "config.json").write_text(json.dumps(self.config))
        metrics_path = self.root / "cli.metrics.json"
        _, argv = self.cli_args("process", "--num_workers", "1", "--max_batch_samples", "6400",
                                "--save_metrics_to", str(metrics_path))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

        subprocess.run([sys.executable, str(Path(entrypoint.__file__)), *argv],
                       env=env, capture_output=True, text=True, check=True, timeout=60)

        self.assertEqual(json.loads(metrics_path.read_text())["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
