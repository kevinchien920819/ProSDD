import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

import main__eval_rhythm as evaluation
import main_stage2realfake_rhythm as training
import test_rhythm_evaluation as legacy_eval
import test_stage2_rhythm_training as legacy_training
from data_utils_stage2realfake_rhythm import ProSDDStage2RhythmDataset, collate_stage2_rhythm, ProsodyAlignmentError
from model_stage1real import ProSDDStage1
from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from extract_full_prosody import extract_aligned_targets
import extract_full_prosody as extraction
from full_utterance import LengthBatchSampler, read_full_prosody_metadata


def layer_norm_backbone(*args, **kwargs):
    model = legacy_eval.tiny_backbone()
    model.config.feat_extract_norm = "layer"  # XLS-R normalizes each frame independently.
    return type(model)(model.config)


class FullUtteranceRhythmTests(unittest.TestCase):
    setUp = legacy_eval.RhythmEvaluationTests.setUp
    write_csv = legacy_eval.RhythmEvaluationTests.write_csv

    def full_dataset(self):
        self.speaker = self.root / "speaker.txt"
        self.speaker.write_text("".join(f"{sp} " + " ".join([".1"] * 192) + "\n" for sp in ("s1", "s2")))
        self.prosody = self.root / "prosody.txt"
        self.prosody.write_text("".join(
            utt + "\t" + "|".join(",".join(map(str, frame)) for frame in torch.randn(count, 128).tolist()) + "\n"
            for utt, count in (("long", 299), ("short", 49))
        ))
        metadata = {
            "schema_version": 1, "audio_mode": "full_utterance", "sample_rate": 16000,
            "alignment": "cnn_receptive_field_centers", "conv_kernel": [400], "conv_stride": [320],
            "cache_size_bytes": self.prosody.stat().st_size,
            "utterances": {"long": {"num_samples": 96000, "frames": 299},
                           "short": {"num_samples": 16000, "frames": 49}},
        }
        (self.root / "prosody.txt.meta.json").write_text(json.dumps(metadata))
        options = dict(utt_ids=["long", "short"], spk_ids=["s1", "s2"], labels=[0, 1],
                       wav_dir=str(self.root), spkmean_txt=str(self.speaker), prosody_txt=str(self.prosody),
                       duration_csv=str(self.csv), audio_ext=".wav", T_target=None)
        self.dataset_options = options
        return ProSDDStage2RhythmDataset(**options)

    def full_batch(self):
        dataset = self.full_dataset()
        return collate_stage2_rhythm([dataset[0], dataset[1]], T_target=None,
                                     conv_kernel=(400,), conv_stride=(320,))

    def full_model(self):
        config = {**self.config, "T_target": None, "target_samples": None, "audio_mode": "full_utterance"}
        options = {k: v for k, v in config.items() if k not in ("model_class", "sample_rate", "target_samples", "audio_mode")}
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=layer_norm_backbone):
            model = ProSDDStage2Rhythm(**options, num_time_neg=2, num_spk_neg=1).eval()
        return model, config

    def test_full_audio_and_all_syllables_survive_batching(self):
        dataset = evaluation.RhythmEvalDataset(
            ["long", "short"], self.root, self.csv, rhythm_sources=["syllable"],
            T_target=None, conv_kernel=(400,), conv_stride=(320,), audio_ext=".wav",
        )
        long, short = dataset[0], dataset[1]
        self.assertEqual(long["wav"].numel(), 96000)
        self.assertEqual(short["wav"].numel(), 16000)
        self.assertEqual(long["duration_features"].shape, (6, 3))
        batch = evaluation.collate_rhythm_eval([long, short])
        self.assertEqual(batch["wav"].shape, (2, 96000))
        self.assertEqual((~batch["frame_padding_mask"]).sum(1).tolist(), [299, 49])
        self.assertTrue((batch["wav"][1, 16000:] == 0).all())
        self.assertEqual(batch["rhythm_padding_mask"].sum(1).tolist(), [0, 3])

    def test_new_training_defaults_to_full_utterances(self):
        parser = training.build_parser()
        self.assertIsNone(parser.get_default("T_target"))
        self.assertEqual(parser.get_default("batch_size"), 32)
        self.assertEqual(parser.get_default("num_workers"), 8)
        self.assertEqual(parser.get_default("max_batch_samples"), 0)
        self.assertEqual(parser.get_default("dropout"), .1)

    def test_full_training_and_eval_inputs_match(self):
        training_batch = self.full_batch()
        dataset = evaluation.RhythmEvalDataset(
            ["long", "short"], self.root, self.csv, rhythm_sources=self.config["rhythm_sources"],
            T_target=None, conv_kernel=(400,), conv_stride=(320,), audio_ext=".wav",
        )
        batch = evaluation.collate_rhythm_eval([dataset[0], dataset[1]])
        for key in ("wav", "duration_features", "frame_padding_mask", "rhythm_padding_mask"):
            torch.testing.assert_close(training_batch[key], batch[key], rtol=0, atol=0)
        self.assertEqual(training_batch["prosody_emb"].shape, (2, 299, 128))
        self.assertTrue((training_batch["prosody_emb"][1, 49:] == 0).all())

    def test_tail_contributes_gradients_and_padding_is_ignored(self):
        batch = self.full_batch()
        inputs = {k: v for k, v in batch.items() if k not in ("utt_ids", "labels", "skipped_samples")}
        model, _ = self.full_model()
        inputs["wav"].requires_grad_()
        torch.manual_seed(77)
        out = model(**inputs)
        changed = {**inputs, "prosody_emb": inputs["prosody_emb"].masked_fill(
            inputs["frame_padding_mask"].unsqueeze(-1), torch.nan)}
        torch.manual_seed(77)
        other = model(**changed)
        torch.testing.assert_close(out["ssl_loss"], other["ssl_loss"], rtol=0, atol=0)
        (F.cross_entropy(out["logits"], batch["labels"]) + .2 * out["ssl_loss"]).backward()
        self.assertGreater(inputs["wav"].grad[0, 64000:].abs().sum().item(), 0)
        self.assertEqual(inputs["wav"].grad[1, 16000:].abs().sum().item(), 0)
        self.assertGreater(model.cls_head.classifier.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.final_proj.weight.grad).all())
        # Positional encoding grew beyond its initial buffer without changing state_dict shapes.
        self.assertEqual(model.cls_head.pos.pe.size(1), 201)

    def test_short_prediction_is_invariant_to_longer_batch_member(self):
        batch = self.full_batch()
        model, _ = self.full_model()
        keys = ("wav", "duration_features", "frame_padding_mask", "rhythm_padding_mask")
        with torch.no_grad():
            paired = model(**{k: batch[k] for k in keys}, compute_ssl=False)["logits"][1]
            single = model(
                batch["wav"][1:2, :16000], duration_features=batch["duration_features"][1:2, :3],
                frame_padding_mask=batch["frame_padding_mask"][1:2, :49], compute_ssl=False,
            )["logits"][0]
        torch.testing.assert_close(paired, single, rtol=1e-4, atol=1e-5)

    def test_legacy_or_mismatched_targets_fail_even_when_skipping_bad_samples(self):
        dataset = self.full_dataset()
        sf.write(self.root / "long.wav", np.ones(80000), 16000)
        with self.assertRaisesRegex(ProsodyAlignmentError, "audio length differs"):
            dataset[0]
        (self.root / "prosody.txt.meta.json").unlink()
        with self.assertRaisesRegex(ValueError, "four-second caches cannot be reused"):
            ProSDDStage2RhythmDataset(**self.dataset_options, skip_bad_samples=True)

    def test_full_training_saves_dynamic_config_and_eval_loads_it(self):
        self.check_full_training_batches(budget=None, expected_batches=1)

    def test_full_training_length_budget_is_explicitly_opt_in(self):
        self.check_full_training_batches(budget=100000, expected_batches=2)

    def check_full_training_batches(self, budget, expected_batches):
        self.full_dataset()
        checkpoint, log_dir = self.root / "stage1.pth", self.root / "full_run"
        argv = legacy_training.RhythmTrainingTests.cli_args(self, checkpoint, log_dir)
        index = argv.index("--T_target")
        del argv[index:index + 2]
        if budget is not None:
            argv.extend(["--max_batch_samples", str(budget)])
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=layer_norm_backbone), \
                patch("multi_gpu.resolve_devices", return_value=[torch.device("cpu")]), \
                patch("wandb.init"), patch("sys.argv", [training.__file__, *argv]):
            import runpy
            stage1 = ProSDDStage1(model_name="tiny", prosody_dim=128)
            torch.save(stage1.state_dict(), checkpoint)
            namespace = runpy.run_path(training.__file__, run_name="__main__")
            model, config = evaluation.load_model(log_dir / "model_epoch_1.pth")
        for name in ("train_loader", "dev_loader"):
            loader = namespace[name]
            self.assertEqual(len(loader), expected_batches)
            self.assertEqual(sorted(i for batch in loader.batch_sampler for i in batch), [0, 1])
            if budget is None:
                self.assertEqual(loader.batch_size, 2)
                self.assertNotIsInstance(loader.batch_sampler, LengthBatchSampler)
            else:
                self.assertIsInstance(loader.batch_sampler, LengthBatchSampler)
        self.assertEqual(config["max_batch_samples"], budget or 0)
        self.assertEqual(config["batching_policy"], "fixed_batch_size" if budget is None else "length_budget")
        self.assertIsNone(model.T_target)
        self.assertEqual(config["audio_mode"], "full_utterance")
        self.assertIsNone(config["target_samples"])
        record = json.loads((log_dir / "metrics.jsonl").read_text())
        self.assertEqual(record["train/samples"], 2)
        self.assertEqual(record["val/samples"], 2)
        self.assertEqual(record["beta"], .2)

    def test_length_batches_keep_every_utterance_without_truncation(self):
        lengths = [16000, 96000, 32000, 128000, 8000]
        sampler = LengthBatchSampler(lengths, 3, 100000, shuffle=True)
        batches = list(sampler)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(5)))
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            if len(batch) > 1:
                self.assertLessEqual(max(lengths[i] for i in batch) * len(batch), 100000)
        self.assertIn([3], batches)
        self.assertEqual(batches, list(sampler))

    def test_target_extraction_keeps_tail_and_uses_physical_frame_centers(self):
        class Teacher:
            vad_measure = SimpleNamespace(hop_length=256, sampling_rate=22050)
            offset = 0

            def process_audio(self, path, layer):
                samples, sr = sf.read(path)
                t = np.arange(int(len(samples) / sr * 22050 / 256) + 1) * (256 / 22050)
                values = t + self.offset
                self.offset += len(samples) / sr
                return np.repeat(values[:, None], 128, axis=1)

        wav = torch.ones(8 * 16000)
        targets = extract_aligned_targets(Teacher(), wav, (400,), (320,))
        centers = (np.arange(399) * 320 + 399 / 2) / 16000
        self.assertEqual(targets.shape, (399, 128))
        np.testing.assert_allclose(targets[:, 0], centers, atol=2e-6)
        self.assertGreater(targets[-1, 0], 7.9)

    def test_extraction_cli_writes_metadata_and_refuses_to_overwrite(self):
        output = self.root / "extracted.txt"
        args = ["--protocol_txt", str(self.protocol), "--audio_dir", str(self.root),
                "--out_txt", str(output), "--ext", ".wav"]
        teacher = SimpleNamespace(cpu=lambda: SimpleNamespace(eval=lambda: object()))
        with patch("masked_prosody_model.MaskedProsodyModel.from_pretrained", return_value=teacher), \
                patch.object(extraction.AutoConfig, "from_pretrained", return_value=SimpleNamespace(
                    conv_kernel=[400], conv_stride=[320])), \
                patch.object(extraction, "extract_aligned_targets", side_effect=lambda model, wav, *a, **k:
                             np.ones(((wav.numel() - 400) // 320 + 1, 128), dtype=np.float32)):
            extraction.main(args)
        metadata = read_full_prosody_metadata(output)
        self.assertEqual(metadata["utterances"]["long"], {"num_samples": 96000, "frames": 299})
        self.assertEqual(metadata["utterances"]["short"], {"num_samples": 16000, "frames": 49})
        before = output.read_bytes()
        with self.assertRaises(SystemExit):
            extraction.main(args)
        self.assertEqual(output.read_bytes(), before)

    def test_full_checkpoint_eval_scores_every_complete_utterance(self):
        model, config = self.full_model()
        (self.root / "config.json").write_text(json.dumps(config))
        torch.save(model.state_dict(), self.checkpoint)
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=layer_norm_backbone), \
                patch("main__eval_rhythm.resolve_devices", return_value=[torch.device("cpu")]):
            evaluation.main(legacy_eval.RhythmEvaluationTests.argv(self))
        metrics = json.loads((self.root / "metrics.json").read_text())
        coverage = json.loads((self.root / "scores.coverage.json").read_text())
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(coverage["audio_mode"], "full_utterance")
        self.assertEqual(coverage["skipped_samples"], 0)


if __name__ == "__main__":
    unittest.main()
