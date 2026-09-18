import csv
from contextlib import redirect_stderr
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from data_utils_stage2realfake_rhythm import (
    ProSDDStage2RhythmDataset, collate_stage2_rhythm,
    duration_features_for_window, load_duration_csv, load_utt_spk_label,
)
import main_stage2realfake_rhythm as entrypoint
from model_stage1real import ProSDDStage1
from model_stage2realfake_rhythm import ProSDDStage2Rhythm


def tiny_audio_backbone(*args, **kwargs):
    # Match XLS-R's 400-sample receptive field and 320-sample frame stride.
    return Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=16, conv_dim=(8,), conv_stride=(320,), conv_kernel=(400,),
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=True, hidden_dropout=0.0, attention_dropout=0.0,
        activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
    ))


class RhythmTrainingTests(unittest.TestCase):
    def test_wandb_metrics_follow_stage2_metric_schema(self):
        train = {
            "loss": 1.0, "ssl_loss": 2.0, "cls_loss": 3.0,
            "spk_cos": 4.0, "pros_cos": 5.0, "acc": 0.6,
            "acc_bonafide": 0.7, "acc_spoof": 0.5, "acc_balanced": 0.6,
            "samples": 12, "skipped_samples": 2,
        }
        val = {
            "loss": 6.0, "ssl_loss": 7.0, "cls_loss": 8.0,
            "spk_cos": 9.0, "pros_cos": 10.0, "acc": 0.8,
            "acc_bonafide": 0.9, "acc_spoof": 0.7, "acc_balanced": 0.8,
            "samples": 10, "skipped_samples": 1, "eer": 0.2,
        }

        metrics = entrypoint.wandb_metrics_for_epoch(
            epoch=3, alpha=0.7, beta=0.3, train=train, val=val,
        )

        self.assertEqual(metrics, {
            "epoch": 3, "alpha": 0.7, "beta": 0.3,
            "loss/train_total": 1.0, "loss/train_ssl": 2.0, "loss/train_cls": 3.0,
            "cos/train_spk": 4.0, "cos/train_pros": 5.0,
            "loss/val_total": 6.0, "loss/val_ssl": 7.0, "loss/val_cls": 8.0,
            "cos/val_spk": 9.0, "cos/val_pros": 10.0,
            "acc/train": 0.6, "acc/train_bonafide": 0.7,
            "acc/train_spoof": 0.5, "acc/train_balanced": 0.6,
            "acc/val": 0.8, "acc/val_bonafide": 0.9,
            "acc/val_spoof": 0.7, "acc/val_balanced": 0.8,
            "eer/val": 0.2,
            "samples/train": 12, "samples/val": 10,
            "samples/skipped_train": 2, "samples/skipped_val": 1,
        })
        self.assertFalse(any(key.startswith(("train/", "val/")) for key in metrics))

    def setUp(self):
        torch.manual_seed(17)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.protocol = self.root / "protocol.txt"
        self.protocol.write_text("s1 long - A01 spoof\ns2 short - - bonafide\n")
        self.speaker = self.root / "speaker.txt"
        self.speaker.write_text("\n".join(
            speaker + " " + " ".join(map(str, torch.randn(192).tolist())) for speaker in ("s1", "s2")
        ))
        self.prosody = self.root / "prosody.txt"
        self.prosody.write_text("\n".join(
            utt + "\t" + "|".join(
                ",".join(map(str, frame)) for frame in torch.randn(200, 128).tolist()
            ) for utt in ("long", "short")
        ))
        self.rows = []
        # Cache rows deliberately reverse protocol order and contain wrong labels.
        for utt, starts, ends in (
            ("short", [.1, .4, .8], [.2, .6, .9]),
            ("long", [.2, .8, 1.2, 2., 4.8, 5.5], [.5, 1.2, 1.4, 2.4, 5.2, 5.7]),
        ):
            duration = np.round(np.array(ends) - starts, 4)
            self.rows.append({
                "flac_file_name": utt + ".flac", "label": "spoof" if utt == "short" else "bonafide",
                "starttime_syllable": ",".join(map(str, starts)),
                "endtime_syllable": ",".join(map(str, ends)),
                "duration_syllable": ",".join(map(str, duration)),
                "duration_vowel": ",".join(map(str, duration / 2)),
                "duration_consonant": ",".join(map(str, duration / 2)),
                "devi_mu_syllable": "999", "mu_diff_syllable": "999",
            })
        self.csv = self.root / "duration.csv"
        self.write_csv()
        for utt, seconds in (("long", 6), ("short", 1)):
            time = np.arange(seconds * 16000) / 16000
            sf.write(self.root / f"{utt}.wav", 0.1 * np.sin(2 * np.pi * 180 * time), 16000)

    def write_csv(self):
        with self.csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def add_empty_duration_row(self):
        # Match the failed ASVspoof5 row; no audio or SSL targets exist for it.
        utt = "T_0000001169"
        row = dict.fromkeys(self.rows[0], "")
        row["flac_file_name"] = utt + ".flac"
        self.rows.insert(1, row)
        self.write_csv()
        self.protocol.write_text(f"s1 long - A01 spoof\nmissing {utt} - A01 spoof\ns2 short - - bonafide\n")
        return utt

    def dataset(self, **kwargs):
        kwargs.setdefault("T_target", 200)
        utt_ids, spk_ids, labels = load_utt_spk_label(self.protocol)
        return ProSDDStage2RhythmDataset(
            utt_ids=utt_ids, spk_ids=spk_ids, labels=labels, wav_dir=str(self.root),
            spkmean_txt=str(self.speaker), prosody_txt=str(self.prosody),
            duration_csv=str(self.csv), audio_ext=".wav", **kwargs,
        )

    def test_center_crop_recomputes_statistics_and_both_padding_masks(self):
        data = self.dataset()
        long, short = data[0], data[1]
        self.assertEqual(long["utt_id"], "long")
        self.assertEqual(long["labels"], 0)  # Protocol, not cache, determines labels.
        self.assertEqual(long["valid_samples"], (0, 64000))
        self.assertEqual(short["valid_samples"], (24000, 40000))
        torch.testing.assert_close(long["duration_features"], torch.tensor([
            [.2, -.1, -.6667, .1, -.05, -.6667, .1, -.05, -.6667],
            [.4, .1, .6667, .2, .05, .6667, .2, .05, .6667],
        ]))
        self.assertEqual(short["duration_features"].shape, (3, 9))
        raw, _ = sf.read(self.root / "long.wav", dtype="float32")
        torch.testing.assert_close(long["wav"], torch.from_numpy(raw[16000:80000]), rtol=0, atol=0)
        batch = collate_stage2_rhythm([long, short], T_target=200, conv_kernel=(400,), conv_stride=(320,))
        self.assertEqual(batch["duration_features"].shape, (2, 3, 9))
        self.assertEqual(batch["rhythm_padding_mask"].tolist(), [[False, False, True], [False, False, False]])
        self.assertTrue((batch["duration_features"][0, 2] == -100).all())
        self.assertEqual((~batch["frame_padding_mask"][0]).nonzero().flatten().tolist(), list(range(199)))
        self.assertEqual((~batch["frame_padding_mask"][1]).nonzero().flatten().tolist(), list(range(75, 124)))
        # Real zero-valued pauses must not be confused with artificial padding.
        short["wav"].zero_()
        silent = collate_stage2_rhythm([short], T_target=200, conv_kernel=(400,), conv_stride=(320,))
        torch.testing.assert_close(silent["frame_padding_mask"][0], batch["frame_padding_mask"][1])

    def test_vowel_only_cache_and_augmentation_preserve_duration_and_padding(self):
        for row in self.rows:
            del row["duration_syllable"], row["duration_consonant"]
        self.write_csv()
        augment = lambda wav, sr, args, algo: wav + 1
        plain = self.dataset(rhythm_sources=("vowel",))[1]
        augmented = self.dataset(
            rhythm_sources=("vowel",), augment_fn=augment, augment_algo=3, augment_prob=1.,
        )[1]
        self.assertEqual(augmented["duration_features"].shape, (3, 3))
        torch.testing.assert_close(plain["duration_features"], augmented["duration_features"])
        self.assertTrue((augmented["wav"][:24000] == 0).all())
        self.assertTrue((augmented["wav"][40000:] == 0).all())
        torch.testing.assert_close(augmented["wav"][24000:40000], plain["wav"][24000:40000] + 1)

    def test_cache_missing_rows_bad_arrays_and_old_schema_fail_clearly(self):
        with self.subTest("missing row"):
            self.rows.pop()
            self.write_csv()
            with self.assertRaisesRegex(ValueError, "missing durations.*long"):
                self.dataset()
        with self.subTest("mismatched per-syllable length"):
            self.rows[0]["duration_vowel"] = "0.1"
            self.write_csv()
            with self.assertRaisesRegex(ValueError, "invalid durations for short"):
                load_duration_csv(self.csv, ["short"], ["syllable", "vowel"])
        with self.subTest("old ASVspoof 2019 word/vowel interval CSV"):
            self.csv.write_text("file_name,utt_start_end,word_start_end,vowel_start_end\n")
            with self.assertRaisesRegex(ValueError, "missing duration CSV columns"):
                self.dataset()

    def test_empty_duration_row_is_skipped_before_loading_ssl_targets(self):
        utt = self.add_empty_duration_row()
        with self.assertRaisesRegex(ValueError, "missing duration data.*T_0000001169"):
            self.dataset()
        data = self.dataset(skip_missing_duration=True)
        self.assertEqual(data.skipped_duration_ids, [utt])
        self.assertEqual(data.utt_ids, ["long", "short"])
        self.assertEqual(data.spk_ids, ["s1", "s2"])
        self.assertEqual(data.labels, [0, 1])
        self.assertEqual(set(data.utt2duration), {"long", "short"})
        self.assertEqual([data[i]["utt_id"] for i in range(len(data))], ["long", "short"])

    def test_skip_handles_empty_and_nonfinite_values_without_dropping_valid_zero(self):
        for cell in ("", "   ", "[]", "[ ]", "-", "nan", "0.1,,0.1", "0.1,inf,0.1"):
            with self.subTest(cell=cell):
                self.rows[0]["duration_vowel"] = cell
                self.write_csv()
                data = self.dataset(skip_missing_duration=True)
                self.assertEqual(data.utt_ids, ["long"])
                self.assertEqual(data.skipped_duration_ids, ["short"])
        self.rows[0]["duration_vowel"] = "0,0,0"
        self.write_csv()
        data = self.dataset(skip_missing_duration=True)
        self.assertEqual(data.skipped_duration_ids, [])
        self.assertTrue((data[1]["duration_features"][:, 3] == 0).all())

    def test_skip_does_not_hide_wrong_csv_or_corrupt_nonempty_rows(self):
        for cell in ("oops", "0.1", "-0.1,-0.1,-0.1"):
            with self.subTest(cell=cell):
                self.rows[0]["duration_vowel"] = cell
                self.write_csv()
                with self.assertRaisesRegex(ValueError, "invalid durations for short"):
                    self.dataset(skip_missing_duration=True)
        self.rows[0]["duration_vowel"] = ""
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "missing durations.*another_dataset"):
            load_duration_csv(self.csv, ["short", "another_dataset"], ["vowel"], skip_missing_duration=True)
        self.rows.append(dict(self.rows[0]))
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "duplicate duration row for short"):
            self.dataset(skip_missing_duration=True)

    def test_skip_rejects_an_empty_dataset(self):
        for row in self.rows:
            row["duration_syllable"] = ""
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "No utterances with complete durations"):
            self.dataset(skip_missing_duration=True)

    def test_empty_crop_and_acoustic_truncation_are_rejected(self):
        record = np.array([[.1, .3, .2], [.5, .8, .3]])
        with self.assertRaisesRegex(ValueError, "No complete syllable"):
            duration_features_for_window(record, .2, .6)
        single = duration_features_for_window(record, 0, .4)
        torch.testing.assert_close(single, torch.tensor([[.2, 0., 0.]]))
        batch = [self.dataset()[0]]
        with self.assertRaisesRegex(ValueError, "T_target truncates acoustic frames"):
            collate_stage2_rhythm(batch, T_target=100, conv_kernel=(400,), conv_stride=(320,))
        with self.assertRaisesRegex(ValueError, "expected cached prosody shape"):
            self.dataset(T_target=199)

    def test_missing_prosody_reports_the_id_and_source_instead_of_a_shape_error(self):
        self.prosody.write_text(self.prosody.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "short: missing cached prosody entry") as error:
            self.dataset()
        self.assertIn(str(self.prosody), str(error.exception))

    def test_native_asvspoof5_labels_and_invalid_protocols(self):
        self.protocol.write_text("s1 a F - 0 0 - A01 spoof -\ns2 b M - 0 0 - - bonafide -\n")
        self.assertEqual(load_utt_spk_label(self.protocol), (["a", "b"], ["s1", "s2"], [0, 1]))
        self.protocol.write_text("s1 a - - typo\n")
        with self.assertRaisesRegex(ValueError, "unknown label"):
            load_utt_spk_label(self.protocol)
        self.protocol.write_text("s1 a - - spoof\ns2 a - - bonafide\n")
        with self.assertRaisesRegex(ValueError, "duplicate utterance"):
            load_utt_spk_label(self.protocol)

    def cli_args(self, checkpoint, log_dir):
        args = [
            "--stage1_ckpt", str(checkpoint), "--log_dir", str(log_dir),
            "--T_target", "200",  # These fixtures exercise the legacy four-second path.
            "--audio_ext", ".wav", "--epochs", "1", "--batch_size", "2", "--num_workers", "0",
            "--model_name", "tiny", "--d_model", "8", "--nhead", "2",
            "--n_rhythm_encoder_layers", "1", "--n_cls_encoder_layers", "1",
            "--rhythm_sources", "syllable", "vowel", "consonant",
            "--dropout", "0", "--max_position_embeddings", "256",
            "--num_time_neg", "2", "--num_spk_neg", "1", "--algo", "0", "--wandb_mode", "disabled",
            "--lr_ssl_backbone", ".001", "--lr_ssl_head", ".002", "--lr_cls", ".003",
        ]
        for split in ("train", "dev"):
            for name, value in (
                (f"{split}_list", self.protocol), (f"wav_dir_{split}", self.root),
                (f"spkmean_txt_{split}", self.speaker), (f"prosody_txt_{split}", self.prosody),
                (f"duration_csv_{split}", self.csv),
            ):
                args.extend([f"--{name}", str(value)])
        return args

    def run_training(self, argv):
        with patch.object(sys, "argv", [entrypoint.__file__, *argv]):
            return runpy.run_path(entrypoint.__file__, run_name="__main__")

    def test_main_rejects_a_split_with_one_class_after_filtering(self):
        self.rows[1]["duration_syllable"] = ""
        self.write_csv()
        checkpoint = self.root / "stage1.pth"
        checkpoint.touch()
        stderr = io.StringIO()
        argv = self.cli_args(checkpoint, self.root / "run") + ["--skip_missing_duration"]
        with patch("model_stage2realfake_rhythm.ProSDDStage2Rhythm") as model, \
                patch("multi_gpu.resolve_devices", return_value=[torch.device("cpu")]), \
                redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.run_training(argv)
        self.assertIn("train must retain both bonafide and spoof examples after duration filtering", stderr.getvalue())
        model.assert_not_called()

    def add_bad_samples(self):
        # Cover errors found during indexing and errors only visible in workers.
        names = ["missing_target", "wrong_shape", "malformed_target", "bad_duration",
                 "missing_duration", "no_crop", "no_audio", "nan_target", "tiny_audio"]
        good_targets = self.prosody.read_text()
        target = good_targets.splitlines()[1].split("\t")[1]
        with self.prosody.open("a") as file:
            file.write("\n")
            for name in names:
                if name == "missing_target":
                    continue
                payload = target
                if name == "wrong_shape":
                    payload = "|".join(payload.split("|")[:-1])
                elif name == "malformed_target":
                    payload = "1,2|3"
                elif name == "nan_target":
                    payload = "nan," + payload.split(",", 1)[1]
                file.write(name + "\t" + payload + "\n")
        for name in names:
            if name == "missing_duration":
                continue
            row = dict(self.rows[0], flac_file_name=name + ".flac")
            if name == "bad_duration":
                row["duration_vowel"] = "broken"
            elif name == "tiny_audio":
                row.update(starttime_syllable=".001", endtime_syllable=".002",
                           duration_syllable=".001", duration_vowel="0", duration_consonant=".001")
            self.rows.append(row)
            if name == "no_crop":
                (self.root / f"{name}.wav").write_bytes((self.root / "long.wav").read_bytes())
            elif name == "nan_target":
                (self.root / f"{name}.wav").write_bytes((self.root / "short.wav").read_bytes())
            elif name == "tiny_audio":
                sf.write(self.root / f"{name}.wav", np.ones(100) * .1, 16000)
        self.write_csv()
        self.protocol.write_text("".join(f"s1 {name} - A01 spoof\n" for name in names)
                                 + self.protocol.read_text())
        return names[:5], names[5:]

    def test_bad_targets_and_durations_are_skipped_automatically(self):
        initial, runtime = self.add_bad_samples()
        data = self.dataset(skip_bad_samples=True)
        self.assertEqual(data.utt_ids, runtime + ["long", "short"])
        self.assertEqual(set(data.skipped_samples), set(initial))
        self.assertIn("missing cached prosody", data.skipped_samples["missing_target"])
        self.assertIn("(199, 128)", data.skipped_samples["wrong_shape"])
        self.assertIn("frame dimensions", data.skipped_samples["malformed_target"])
        self.assertIn("invalid durations", data.skipped_samples["bad_duration"])
        self.assertIn("missing durations", data.skipped_samples["missing_duration"])
        # Files with a wrong global schema must still stop with a useful error.
        self.csv.write_text("old,wrong,columns\n")
        with self.assertRaisesRegex(ValueError, "missing duration CSV columns"):
            self.dataset(skip_bad_samples=True)

    def test_workers_skip_partial_and_empty_batches_without_losing_valid_samples(self):
        from functools import partial
        from torch.utils.data import DataLoader
        self.add_bad_samples()
        data = self.dataset(skip_bad_samples=True)
        loader = DataLoader(
            data, batch_size=2, num_workers=2, timeout=30,
            sampler=[0, 4, 5, 1, 2, 3],
            collate_fn=partial(collate_stage2_rhythm, T_target=200,
                               conv_kernel=(400,), conv_stride=(320,), skip_bad_samples=True),
        )
        batches = list(loader)
        self.assertEqual([b.get("utt_ids", []) for b in batches], [["long"], ["short"], []])
        reasons = {item["utt_id"]: item["reason"] for b in batches for item in b["skipped_samples"]}
        self.assertEqual(set(reasons), {"no_crop", "no_audio", "nan_target", "tiny_audio"})
        self.assertIn("No complete syllable", reasons["no_crop"])
        self.assertIn("no_audio.wav", reasons["no_audio"])
        self.assertIn("nonfinite", reasons["nan_target"])
        self.assertIn("too short", reasons["tiny_audio"])
        self.assertEqual([b["labels"].tolist() for b in batches[:2]], [[0], [1]])

    def test_skip_does_not_swallow_oom_or_changed_target_files(self):
        data = self.dataset(skip_bad_samples=True)
        for failure in (MemoryError("OOM"), torch.OutOfMemoryError("OOM")):
            with self.subTest(failure=type(failure).__name__):
                with patch("data_utils_stage2realfake_rhythm.torchaudio.load", side_effect=failure), \
                        patch("data_utils_stage2realfake_rhythm._load_audio_with_ffmpeg") as fallback, \
                        self.assertRaises(type(failure)):
                    data[0]
                fallback.assert_not_called()
        with self.prosody.open("a") as file:
            file.write("\n")
        with self.assertRaisesRegex(RuntimeError, "changed after indexing"):
            data[0]

    def test_corrupt_numeric_target_is_skipped_even_when_numpy_can_parse_a_prefix(self):
        lines = self.prosody.read_text().splitlines()
        frames = lines[0].split("\t", 1)[1].split("|")
        values = frames[0].split(",")
        values[-1] = "0.1broken"
        frames[0] = ",".join(values)
        lines[0] = "long\t" + "|".join(frames)
        self.prosody.write_text("\n".join(lines))
        data = self.dataset(skip_bad_samples=True)
        skipped = data[0]
        self.assertEqual(skipped.utt_id, "long")
        self.assertIn("Invalid numeric prosody target", skipped.reason)
        self.assertEqual(data[1]["utt_id"], "short")

    def test_auto_skip_trains_and_reports_actual_samples_and_reasons(self):
        initial, runtime = self.add_bad_samples()
        checkpoint, log_dir = self.root / "stage1.pth", self.root / "auto_skip_run"
        argv = self.cli_args(checkpoint, log_dir) + ["--skip_bad_samples", "--num_workers", "2"]
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_audio_backbone), \
                patch("multi_gpu.resolve_devices", return_value=[torch.device("cpu")]):
            stage1 = ProSDDStage1(model_name="tiny", prosody_dim=128)
            torch.save(stage1.state_dict(), checkpoint)
            self.run_training(argv)
        report = json.loads((log_dir / "duration_filter.json").read_text())
        config = json.loads((log_dir / "config.json").read_text())
        metrics = json.loads((log_dir / "metrics.jsonl").read_text())
        records = [json.loads(line) for line in (log_dir / "skipped_samples.jsonl").read_text().splitlines()]
        for split in ("train", "dev"):
            self.assertEqual(set(report[split]["skipped_utt_ids"]), set(initial))
            self.assertEqual(config["dataset_counts"][split]["used_samples"], 6)
            self.assertEqual(config["dataset_counts"][split]["skipped_at_init"], 5)
            skips = [row for row in records if row["split"] == split]
            self.assertEqual({row["utt_id"] for row in skips if row["epoch"] == 0}, set(initial))
            self.assertEqual({row["utt_id"] for row in skips if row["epoch"] == 1}, set(runtime))
            self.assertTrue(all(row["reason"] for row in skips))
        for split in ("train", "val"):
            self.assertEqual(metrics[f"{split}/samples"], 2)
            self.assertEqual(metrics[f"{split}/skipped_samples"], 4)
        self.assertTrue(0 <= metrics["val/eer"] <= 1)
        self.assertEqual(list(log_dir.glob("*.pth")), [log_dir / "model_best.pth"])
        self.assertTrue((log_dir / "model_best.pth").is_file())

    def test_main_uses_baseline_beta_schedule_for_training_validation_and_logs(self):
        checkpoint, log_dir = self.root / "stage1.pth", self.root / "scheduled_run"
        argv = self.cli_args(checkpoint, log_dir) + ["--epochs", "5"]
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_audio_backbone), \
                patch("multi_gpu.resolve_devices", return_value=[torch.device("cpu")]), \
                patch("wandb.init") as wandb_init:
            stage1 = ProSDDStage1(model_name="tiny", prosody_dim=128)
            torch.save(stage1.state_dict(), checkpoint)
            saved_epochs = []
            save = torch.save

            def record_save(state, path):
                records = (log_dir / "metrics.jsonl").read_text().splitlines()
                saved_epochs.append(json.loads(records[-1])["epoch"])
                save(state, path)

            with patch("torch.save", side_effect=record_save):
                self.run_training(argv)
        records = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
        best_val_loss = float("inf")
        expected_epochs = []
        for row in records:
            if row["val/loss"] < best_val_loss:
                best_val_loss = row["val/loss"]
                expected_epochs.append(row["epoch"])
        self.assertEqual(saved_epochs, expected_epochs)
        self.assertEqual(list(log_dir.glob("*.pth")), [log_dir / "model_best.pth"])
        self.assertEqual([row["beta"] for row in records], [.2, .2, .2, .2, .05])
        for row in records:
            for split in ("train", "val"):
                self.assertAlmostEqual(
                    row[f"{split}/loss"],
                    row[f"{split}/cls_loss"] + row["beta"] * row[f"{split}/ssl_loss"],
                    places=6,
                )
        logged = wandb_init.return_value.__enter__.return_value.log.call_args_list
        self.assertEqual([call.args[0]["beta"] for call in logged], [.2, .2, .2, .2, .05])
        config = json.loads((log_dir / "config.json").read_text())
        self.assertEqual(config["beta_schedule"], {
            "policy": "baseline", "initial_beta": .2, "initial_epochs": 4, "later_beta": .05,
        })
        self.assertEqual(wandb_init.call_args.kwargs["config"]["beta_schedule"], config["beta_schedule"])

    def test_main_trains_validates_and_saves_with_stage1_and_fixed_loss_weights(self):
        skipped_utt = self.add_empty_duration_row()
        checkpoint, log_dir = self.root / "stage1.pth", self.root / "run"
        calls, captured = [], {}

        def make_model(**kwargs):
            model = ProSDDStage2Rhythm(**kwargs)
            # Strict Stage I initialization must happen before the first batch.
            for name, value in stage1.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            captured["before"] = {name: p.detach().clone() for name, p in model.named_parameters()}
            captured["model"] = model
            model.ssl.encoder.register_forward_pre_hook(
                lambda module, inputs: calls.append((module.training, torch.is_grad_enabled()))
            )
            return model

        argv = self.cli_args(checkpoint, log_dir)
        defaults = entrypoint.build_parser().parse_args(argv)
        self.assertEqual((defaults.alpha, defaults.beta), (1., None))
        self.assertFalse(defaults.skip_missing_duration)
        argv.extend(["--alpha", ".7", "--beta", ".3", "--skip_missing_duration"])
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_audio_backbone):
            stage1 = ProSDDStage1(model_name="tiny", prosody_dim=128)
            torch.save(stage1.state_dict(), checkpoint)
            with patch("model_stage2realfake_rhythm.ProSDDStage2Rhythm", side_effect=make_model), \
                    patch("multi_gpu.resolve_devices", return_value=[torch.device("cpu")]):
                result = self.run_training(argv)
        self.assertEqual(calls, [(True, True), (True, True), (False, False), (False, False)])
        record = json.loads((log_dir / "metrics.jsonl").read_text())
        for split in ("train", "val"):
            self.assertEqual(record[f"{split}/samples"], 2)
            self.assertAlmostEqual(record[f"{split}/loss"],
                                   .7 * record[f"{split}/cls_loss"] + .3 * record[f"{split}/ssl_loss"], places=6)
        self.assertEqual(record["beta"], .3)
        self.assertTrue(0 <= record["val/eer"] <= 1)
        config = json.loads((log_dir / "config.json").read_text())
        self.assertEqual(config["prosody_dim"], 128)
        self.assertEqual(len(config["duration_feature_names"]), 9)
        report = json.loads((log_dir / "duration_filter.json").read_text())
        self.assertEqual(report["policy"], "skip_missing_duration")
        for split in ("train", "dev"):
            self.assertEqual(report[split]["skipped_utt_ids"], [skipped_utt])
            self.assertEqual(config["dataset_counts"][split], {
                "protocol_samples": 3, "used_samples": 2, "skipped_missing_duration": 1,
                "skipped_at_init": 1, "spoof": 1, "bonafide": 1,
            })
        self.assertTrue((log_dir / "model_best.pth").is_file())
        self.assertEqual(list(log_dir.glob("*.pth")), [log_dir / "model_best.pth"])
        saved = torch.load(log_dir / "model_best.pth", map_location="cpu", weights_only=True)
        for name in (
            "ssl.encoder.layers.0.attention.q_proj.weight", "final_proj.weight",
            "cls_head.rhythm_embedding.0.weight", "cls_head.classifier.weight",
        ):
            self.assertFalse(torch.equal(captured["before"][name], saved[name]), name)
        captured["model"].load_state_dict(saved, strict=True)
        groups = result["optimizer"].param_groups
        self.assertEqual([g["lr"] for g in groups], [.001, .002, .003])
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in captured["model"].parameters() if p.requires_grad})
        for parameter in captured["model"].parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
