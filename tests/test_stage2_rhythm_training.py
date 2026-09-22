import csv
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model

import main_stage2realfake_rhythm as entrypoint
from model_stage1real import ProSDDStage1
from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from test_data_utils_rhythm import CropTeacher
from test_stage2_rhythm import build_model, inputs


class RhythmTrainingTests(unittest.TestCase):
    def test_parser_accepts_an_independent_rhythm_learning_rate(self):
        required = (
            "--train_list", "--dev_list", "--wav_dir_train", "--wav_dir_dev",
            "--spkmean_txt_train", "--spkmean_txt_dev",
            "--duration_csv_train", "--duration_csv_dev", "--stage1_ckpt",
        )
        argv = [part for flag in required for part in (flag, "unused")]
        args = entrypoint.build_parser().parse_args(argv + ["--lr_rhythm", "0.003", "--lr_cls", "0.002"])

        self.assertEqual((args.lr_rhythm, args.lr_cls), (0.003, 0.002))

    def test_rhythm_learning_rate_updates_fusion_independently(self):
        torch.manual_seed(71)
        model = build_model().eval()
        args = SimpleNamespace(lr_ssl_backbone=0.0, lr_ssl_head=0.0,
                               lr_rhythm=1e-3, lr_cls=0.0, weight_decay=0.0)
        optimizer = entrypoint.build_optimizer(model, args)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        out = model(**inputs(model))

        F.cross_entropy(out["logits"], torch.tensor([0, 1, 0])).backward()
        optimizer.step()

        changed = {name for name, p in model.named_parameters() if not torch.equal(before[name], p)}
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("rhythm_fusion.") for name in changed))

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



def audio_backbone(*args, **kwargs):
    """以可處理真實音訊長度的小型 Wav2Vec2 替代外部預訓練下載。"""
    return Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        conv_dim=(8,), conv_kernel=(400,), conv_stride=(320,),
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
        feat_extract_norm="layer", hidden_dropout=0.0, attention_dropout=0.0,
        activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
    ))


class RhythmEntrypointTests(unittest.TestCase):
    """從 CLI 參數跑真實資料、SSL backward、驗證，檢查可讀取的訓練產物。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.log_dir = self.root / "run"
        self.protocol = self.root / "protocol.txt"
        self.protocol.write_text("s1 real - - bonafide\ns2 fake - A01 spoof\n")
        self.speakers = self.root / "speakers.txt"
        self.speakers.write_text("\n".join(
            f"s{i} " + " ".join([str(i / 10)] * 192) for i in (1, 2)
        ))
        self.csv = self.root / "duration.csv"
        rows = []
        for i, utt in enumerate(("real", "fake")):
            t = np.arange(16000 + i * 3200) / 16000
            sf.write(self.root / f"{utt}.wav", 0.1 * np.sin(2 * np.pi * (120 + i * 80) * t), 16000)
            rows.append({
                "flac_file_name": f"{utt}.flac",
                "starttime_word": "0.1, 0.4, 0.7", "endtime_word": "0.25, 0.55, 0.85",
                "starttime_syllable": "0.1, 0.4, 0.7", "endtime_syllable": "0.25, 0.55, 0.85",
                "duration_syllable": "0.15, 0.15, 0.15",
            })
        with self.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for replacement in (
            patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=audio_backbone),
            patch("masked_prosody_model.MaskedProsodyModel.from_pretrained", return_value=CropTeacher()),
            patch("torch.cuda.is_available", return_value=False),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        stage1 = ProSDDStage1(model_name="test", prosody_dim=128)
        self.initial = {name: value.clone() for name, value in stage1.state_dict().items()}
        self.checkpoint = self.root / "stage1.pth"
        torch.save(self.initial, self.checkpoint)

    def argv(self, *extra):
        pairs = {
            "train_list": self.protocol, "dev_list": self.protocol,
            "wav_dir_train": self.root, "wav_dir_dev": self.root,
            "spkmean_txt_train": self.speakers, "spkmean_txt_dev": self.speakers,
            "duration_csv_train": self.csv, "duration_csv_dev": self.csv,
            "stage1_ckpt": self.checkpoint, "model_name": "test", "audio_ext": ".wav",
            "audio_seconds": 0, "epochs": 1, "batch_size": 2, "num_workers": 0,
            "nhead": 2, "n_rhythm_encoder_layers": 1, "n_cls_encoder_layers": 1,
            "num_time_neg": 2, "num_spk_neg": 1, "mask_span_len": 2,
            "max_position_embeddings": 512, "dropout": 0, "algo": 0,
            "wandb_mode": "disabled", "log_dir": self.log_dir,
        }
        return [part for key, value in pairs.items() for part in (f"--{key}", str(value))] + list(extra)

    def test_main_trains_from_stage1_and_saves_reloadable_epoch_weights(self):
        entrypoint.main(self.argv())

        state = torch.load(self.log_dir / "model_epoch_1.pth", weights_only=True)
        model = ProSDDStage2Rhythm(
            model_name="test", prosody_dim=128, T_target=None, rhythm_sources=["syllable"],
            nhead=2, n_rhythm_encoder_layers=1, n_cls_encoder_layers=1,
            max_position_embeddings=512, dropout=0,
        )
        model.load_state_dict(state, strict=True)
        self.assertFalse(torch.equal(state["final_proj.weight"], self.initial["final_proj.weight"]))

    def test_main_records_epochs_and_selects_the_lowest_validation_eer(self):
        with patch("wandb.init") as init:
            entrypoint.main(self.argv("--epochs", "2", "--beta", "0.37"))

        records = [json.loads(line) for line in (self.log_dir / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([(r["epoch"], r["beta"], r["train/samples"], r["val/samples"])
                          for r in records], [(1, 0.37, 2, 2), (2, 0.37, 2, 2)])
        for record in records:
            self.assertTrue(np.isfinite(record["train/ssl_loss"]))
            self.assertGreaterEqual(record["val/eer"], 0.0)
            self.assertLessEqual(record["val/eer"], 1.0)
        config = json.loads((self.log_dir / "config.json").read_text())
        self.assertEqual((config["model_class"], config["prosody_dim"], config["conv_kernel"]),
                         ("ProSDDStage2Rhythm", 128, [400]))
        best_epoch = min(records, key=lambda r: r["val/eer"])["epoch"]
        best = torch.load(self.log_dir / "model_best.pth", weights_only=True)
        expected = torch.load(self.log_dir / f"model_epoch_{best_epoch}.pth", weights_only=True)
        torch.testing.assert_close(best, expected)
        logs = init.return_value.__enter__.return_value.log.call_args_list
        self.assertEqual([call.kwargs["step"] for call in logs], [1, 2])
        self.assertEqual(logs[-1].args[0]["eer/val"], records[-1]["val/eer"])

    def test_skipped_samples_are_identifiable_even_when_an_entire_batch_fails(self):
        with self.protocol.open("a") as file:
            file.write("s1 no_duration - - bonafide\ns2 missing_audio - A01 spoof\n")
        with self.csv.open() as file:
            row = next(csv.DictReader(file))
        row["flac_file_name"] = "missing_audio.flac"
        with self.csv.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=list(row)).writerow(row)

        entrypoint.main(self.argv("--skip_bad_samples", "--batch_size", "1"))

        skipped = [json.loads(line) for line in (self.log_dir / "skipped_samples.jsonl").read_text().splitlines()]
        self.assertEqual({(r["split"], r["epoch"], r["utt_id"]) for r in skipped}, {
            (split, epoch, utt) for split in ("train", "dev")
            for epoch, utt in ((0, "no_duration"), (1, "missing_audio"))
        })
        self.assertTrue(all(r["reason"] for r in skipped))
        record = json.loads((self.log_dir / "metrics.jsonl").read_text())
        self.assertEqual((record["train/samples"], record["train/skipped_samples"],
                          record["val/samples"], record["val/skipped_samples"]), (2, 1, 2, 1))

    def test_crop_training_respects_the_sample_budget_and_explicit_frame_target(self):
        entrypoint.main(self.argv("--audio_seconds", "0.4", "--T_target", "20",
                                  "--max_batch_samples", "20000", "--algo", "2", "--augment_prob", "1"))

        record = json.loads((self.log_dir / "metrics.jsonl").read_text())
        self.assertEqual((record["train/batches"], record["val/batches"],
                          record["train/samples"], record["val/samples"]), (2, 2, 2, 2))
        self.assertTrue(np.isfinite(record["train/ssl_loss"]))

    def test_main_extracts_targets_in_workers_across_multiple_epochs(self):
        entrypoint.main(self.argv("--num_workers", "1", "--epochs", "2", "--audio_seconds", "0.4"))

        records = [json.loads(line) for line in (self.log_dir / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([(r["train/samples"], r["val/samples"]) for r in records], [(2, 2), (2, 2)])

    def test_invalid_training_settings_fail_at_the_cli_boundary(self):
        for flag, value in (("--epochs", "0"), ("--batch_size", "0"), ("--num_workers", "-1"),
                            ("--max_batch_samples", "-1"), ("--audio_seconds", "nan"),
                            ("--audio_seconds", "-1"), ("--T_target", "0"), ("--teacher_kind", "vad")):
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    entrypoint.main(self.argv(flag, value))
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
