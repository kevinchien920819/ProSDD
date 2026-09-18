import runpy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from data_utils_stage1real import (
    ProSDDStage1Dataset,
    load_prosody_dict as load_stage1_prosody,
    load_vad_prosody_dict,
)
from data_utils_stage2realfake import (
    ProSDDStage2Dataset,
    load_prosody_dict as load_stage2_prosody,
)
from main_eval import main as evaluate
from model_stage1real import ProSDDStage1
from model_stage2realfake import ProSDDStage2
from test_multi_gpu import tiny_backbone


ROOT = Path(__file__).resolve().parents[1]
LOADERS = (load_stage1_prosody, load_vad_prosody_dict, load_stage2_prosody)


def write_prosody(path, dimensions, frame_count=2):
    lines = []
    for index, dim in enumerate(dimensions):
        frame = ",".join(str(value / dim) for value in range(dim))
        frames = "|".join([frame] * frame_count)
        lines.append(f"{103 + index}-1240-0000\t{frames}\n")
    path.write_text("\n" + "".join(lines))


class ProsodyDimensionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.prosody = self.directory / "prosody.txt"
        self.speaker = self.directory / "speaker.txt"
        self.speaker.write_text("103 " + " ".join(["0.1"] * 192) + "\n"
                                "104 " + " ".join(["0.2"] * 192) + "\n")

    def test_loaders_infer_both_dimensions_without_changing_values(self):
        for loader in LOADERS:
            for dim in (128, 256):
                with self.subTest(loader=loader.__name__, module=loader.__module__, dim=dim):
                    write_prosody(self.prosody, [dim, dim])
                    tensors = loader(str(self.prosody))
                    self.assertEqual(tensors["103-1240-0000"].shape, (2, dim))
                    expected = torch.arange(dim, dtype=torch.float32) / dim
                    torch.testing.assert_close(tensors["103-1240-0000"][0], expected)

    def test_loaders_reject_mixed_dimensions_and_explicit_mismatch(self):
        for loader in LOADERS:
            for first, second in ((128, 256), (256, 128)):
                with self.subTest(loader=loader.__module__, first=first):
                    write_prosody(self.prosody, [first, second])
                    with self.assertRaisesRegex(ValueError, "104-1240-0000"):
                        loader(str(self.prosody))
                    write_prosody(self.prosody, [second])
                    with self.assertRaisesRegex(ValueError, "103-1240-0000"):
                        loader(str(self.prosody), expected_dim=first)

    def test_loaders_reject_empty_files_and_unsupported_dimensions(self):
        for loader in LOADERS:
            with self.subTest(loader=loader.__module__):
                self.prosody.write_text("\n  \n")
                with self.assertRaisesRegex(ValueError, "[Ee]mpty|[Nn]o prosody"):
                    loader(str(self.prosody))
                write_prosody(self.prosody, [64])
                with self.assertRaisesRegex(ValueError, "128.*256"):
                    loader(str(self.prosody))

    def dataset(self, stage, **kwargs):
        if stage == 1:
            return ProSDDStage1Dataset(str(self.prosody), str(self.speaker),
                                      str(self.directory), **kwargs)
        return ProSDDStage2Dataset(
            ["103-1240-0000", "104-1240-0000"], ["103", "104"], [1, 0],
            str(self.directory), str(self.speaker), str(self.prosody), **kwargs,
        )

    def test_datasets_infer_dimensions_and_validate_dev_against_train(self):
        for stage in (1, 2):
            for dim, other_dim in ((128, 256), (256, 128)):
                with self.subTest(stage=stage, dim=dim):
                    write_prosody(self.prosody, [dim, dim])
                    dataset = self.dataset(stage)
                    self.assertEqual(dataset.prosody_dim, dim)
                    self.assertEqual(dataset.utt2pros["103-1240-0000"].shape, (2, dim))
                    write_prosody(self.prosody, [other_dim, other_dim])
                    with self.assertRaisesRegex(ValueError, "Prosody dim mismatch"):
                        self.dataset(stage, prosody_dim=dataset.prosody_dim)

    def run_training_entrypoint(self, stage, dim, explicit=False, dev_dim=None, audio_seconds=None):
        write_prosody(self.prosody, [dim, dim], frame_count=200)
        dev = self.directory / "dev.txt"
        write_prosody(dev, [dev_dim or dim, dev_dim or dim], frame_count=200)
        argv = [f"main_stage{stage}", "--epochs", "0", "--num_workers", "0",
                "--batch_size", "2", "--log_dir", str(self.directory / "logs"),
                "--wav_dir_train", str(self.directory), "--wav_dir_dev", str(self.directory)]
        if stage == 1:
            argv += ["--train_prosody_txt", str(self.prosody), "--dev_prosody_txt", str(dev),
                     "--train_spkmean_txt", str(self.speaker), "--dev_spkmean_txt", str(self.speaker)]
        else:
            protocol = self.directory / "protocol.txt"
            protocol.write_text("103 103-1240-0000 - - bonafide\n104 104-1240-0000 - - spoof\n")
            argv += ["--train_list", str(protocol), "--dev_list", str(protocol),
                     "--prosody_txt_train", str(self.prosody), "--prosody_txt_dev", str(dev),
                     "--spkmean_txt_train", str(self.speaker), "--spkmean_txt_dev", str(self.speaker),
                     "--algo", "0"]
        if explicit:
            argv += ["--prosody_dim", str(dim)]
        if audio_seconds is not None:
            argv += ["--audio_seconds", str(audio_seconds)]
        script = ROOT / ("main_stage1real.py" if stage == 1 else "main_stage2realfake.py")
        with patch.object(sys, "argv", argv), \
                patch("multi_gpu.resolve_devices", return_value=[torch.device("cpu")]), \
                patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone), \
                patch("wandb.init") as init:
            result = runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(init.call_args.kwargs["config"]["prosody_dim"], dim)
        return result

    def test_stage2_audio_seconds_derives_samples_and_frames(self):
        result = self.run_training_entrypoint(2, 128, audio_seconds=6)
        self.assertEqual(result["args"].target_samples, 96000)
        self.assertEqual(result["args"].T_target, 300)
        self.assertEqual(result["train_dataset"].max_len, 96000)
        self.assertEqual(result["dev_dataset"].max_len, 96000)

    def test_training_entrypoints_use_detected_or_explicit_dimensions(self):
        for stage in (1, 2):
            for dim in (128, 256):
                for explicit in (False, True):
                    with self.subTest(stage=stage, dim=dim, explicit=explicit):
                        result = self.run_training_entrypoint(stage, dim, explicit)
                        model = result["model"]
                        self.assertEqual(result["dev_dataset"].prosody_dim, dim)
                        self.assertEqual(model.prosody_dim, dim)
                        self.assertEqual(model.final_proj.out_features, 192 + dim)
                        module = "data_utils_stage1real" if stage == 1 else "data_utils_stage2realfake"
                        with patch(f"{module}.load_audio", return_value=torch.randn(31)):
                            batch = next(iter(result["train_loader"]))
                        output = model(*batch[:4])
                        loss = output if stage == 1 else output["ssl_loss"]
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertIsNotNone(model.final_proj.weight.grad)
                        self.assertIsNotNone(model.pros_ln.weight.grad)
                        self.assertTrue(torch.isfinite(model.pros_ln.weight.grad).all())

    def test_training_entrypoints_reject_different_train_and_dev_dimensions(self):
        for stage in (1, 2):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(ValueError, "Prosody dim mismatch"):
                    self.run_training_entrypoint(stage, 128, dev_dim=256)

    def test_stage1_checkpoints_transfer_to_matching_stage2_models(self):
        for dim in (128, 256):
            with self.subTest(dim=dim), \
                    patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone):
                stage1 = ProSDDStage1(prosody_dim=dim)
                checkpoint = self.directory / "stage1.pth"
                torch.save(stage1.state_dict(), checkpoint)
                stage2 = ProSDDStage2(prosody_dim=dim, stage1_ckpt=str(checkpoint))
                torch.testing.assert_close(stage1.final_proj.weight, stage2.final_proj.weight)
                other_dim = 256 if dim == 128 else 128
                with self.assertRaisesRegex(ValueError, "[Pp]rosody dim mismatch"):
                    ProSDDStage2(prosody_dim=other_dim, stage1_ckpt=str(checkpoint))

    def test_evaluation_infers_dimensions_from_plain_and_wrapped_checkpoints(self):
        for dim, wrapped in ((128, False), (256, False), (256, True)):
            with self.subTest(dim=dim, wrapped=wrapped), \
                    patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone):
                state = ProSDDStage2(prosody_dim=dim).state_dict()
                if wrapped:
                    state = {"state_dict": {f"module.{key}": value for key, value in state.items()}}
                checkpoint = self.directory / "stage2.pth"
                torch.save(state, checkpoint)
                args = SimpleNamespace(
                    list_path="unused", wav_dir="unused", batch_size=1, classifier_pool="mean",
                    model_path=str(checkpoint), save_scores_to=str(self.directory / "scores.txt"),
                )
                with patch("main_eval.resolve_devices", return_value=[torch.device("cpu")]), \
                        patch("main_eval.ProSDDEvalDataset"), \
                        patch("main_eval.DataLoader", return_value=[(torch.randn(1, 31), ["utt"])]):
                    evaluate(args)
                utt, score = Path(args.save_scores_to).read_text().split()
                self.assertEqual(utt, "utt")
                self.assertTrue(torch.isfinite(torch.tensor(float(score))))

    def test_evaluation_writes_metrics_from_saved_scores(self):
        with patch("transformers.Wav2Vec2Model.from_pretrained", side_effect=tiny_backbone):
            checkpoint = self.directory / "stage2.pth"
            torch.save(ProSDDStage2().state_dict(), checkpoint)
            protocol = self.directory / "protocol.txt"
            protocol.write_text("sp spoof - A01 spoof\nsp bona - - bonafide\n")
            metrics_path = self.directory / "metrics.json"
            args = SimpleNamespace(
                list_path=str(protocol), wav_dir="unused", batch_size=2, classifier_pool="mean",
                model_path=str(checkpoint), save_scores_to=str(self.directory / "scores.txt"),
                save_metrics_to=str(metrics_path),
            )
            with patch("main_eval.resolve_devices", return_value=[torch.device("cpu")]), \
                    patch("main_eval.ProSDDEvalDataset"), \
                    patch("main_eval.DataLoader", return_value=[(torch.randn(2, 31), ["bona", "spoof"])]), \
                    patch("main_eval.inference_forward", return_value=torch.tensor([[50.0, 1.0], [50.0, -1.0]])):
                evaluate(args)
        self.assertEqual(Path(args.save_scores_to).read_text(), "bona 1.000000\nspoof -1.000000\n")
        metrics = json.loads(metrics_path.read_text())
        self.assertEqual(metrics["eer"], 0.0)
        self.assertEqual(metrics["sample_count"], 2)
        self.assertAlmostEqual(metrics["cllr"], 0.4519410830830482)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
