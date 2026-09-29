"""從真實資料介面驗證兩階段可選的 gender embedding 訓練與推論。"""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Config, Wav2Vec2Model

from data_utils_stage1real import ProSDDStage1Dataset
from data_utils_stage2realfake import ProSDDStage2Dataset, collate_stage2
from data_utils_rhythm import collate_stage2_rhythm
import main_eval
from main__eval_rhythm import load_model as load_rhythm_model
from main_stage1real import train_epoch, validate
from main_stage2realfake import train_epoch as train_stage2, validate as validate_stage2
from main_stage2realfake_rhythm import build_parser as rhythm_parser, run_epoch as run_rhythm_epoch
from model_stage1real import ProSDDStage1
from model_stage2realfake import ProSDDStage2
from model_stage2realfake_rhythm import ProSDDStage2Rhythm


class Stage1GenderTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.backbone = Path(tmp.name)
        config = Wav2Vec2Config(
            hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, conv_dim=(8,), conv_kernel=(4,), conv_stride=(4,),
            num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
            feat_extract_norm="layer", hidden_dropout=0.0, attention_dropout=0.0,
            activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
        )
        Wav2Vec2Model(config).save_pretrained(cls.backbone)

    def setUp(self):
        torch.manual_seed(123)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.gender_path = self.root / "SPEAKERS.TXT"
        self.gender_path.write_text("17 | M\n19 | F\n", encoding="utf-8")
        self.spk_path = self.root / "speakers.txt"
        self.pros_path = self.root / "prosody.txt"
        gen = torch.Generator().manual_seed(42)
        with self.spk_path.open("w") as speakers, self.pros_path.open("w") as prosody:
            for spk in (17, 19):
                utt = f"{spk}-1-0000"
                speaker = torch.nn.functional.normalize(torch.randn(192, generator=gen), dim=0)
                speakers.write(f"{spk} " + " ".join(map(str, speaker.tolist())) + "\n")
                frames = torch.randn(200, 128, generator=gen)
                prosody.write(utt + "\t" + "|".join(
                    ",".join(map(str, frame.tolist())) for frame in frames
                ) + "\n")
                sf.write(self.root / f"{utt}.flac", torch.randn(800, generator=gen).numpy() * 0.1, 16000)

    def dataset(self, **kwargs):
        return ProSDDStage1Dataset(
            self.pros_path, self.spk_path, self.root, max_len=800, **kwargs,
        )

    def model(self, **kwargs):
        return ProSDDStage1(
            model_name=str(self.backbone), mask_prob=0.02, mask_span_len=2,
            num_time_neg=2, num_spk_neg=1, **kwargs,
        )

    def stage2_dataset(self, **kwargs):
        return ProSDDStage2Dataset(
            ["17-1-0000", "19-1-0000"], ["17", "19"], [1, 0],
            self.root, self.spk_path, self.pros_path, max_len=800, **kwargs,
        )

    def test_dataset_appends_rotated_gender_for_the_matching_speaker(self):
        dataset = self.dataset(gender_txt=self.gender_path, gender_theta=45)

        actual = {sample[3]: sample[4] for sample in dataset}

        torch.testing.assert_close(actual, {
            17: torch.tensor([0.70710678, 0.70710678]),
            19: torch.tensor([-0.70710678, 0.70710678]),
        })

    def test_disabled_gender_keeps_a_collatable_fifth_field(self):
        batch = next(iter(DataLoader(self.dataset(), batch_size=2)))

        self.assertEqual(batch[4].shape, (2, 0))

    def test_stage1_checkpoint_uses_one_projection_for_all_targets(self):
        for prosody_dim, enabled, out_dim in (
            (128, False, 320), (128, True, 322),
            (256, False, 448), (256, True, 450),
        ):
            with self.subTest(prosody_dim=prosody_dim, use_gender=enabled):
                state = self.model(prosody_dim=prosody_dim, use_gender=enabled).state_dict()
                shapes = {key: tuple(value.shape) for key, value in state.items()
                          if key.startswith(("final_proj.", "gender_proj."))}

                self.assertEqual(shapes, {
                    "final_proj.weight": (out_dim, 8), "final_proj.bias": (out_dim,),
                })

    def test_gender_targets_change_the_contrastive_loss(self):
        model = self.model(use_gender=True).eval()
        batch = next(iter(DataLoader(self.dataset(gender_txt=self.gender_path), batch_size=2)))
        torch.manual_seed(7)
        original = model(*batch)
        torch.manual_seed(7)
        swapped = model(*batch[:4], gender_emb=batch[4].flip(0))

        self.assertNotAlmostEqual(original.item(), swapped.item(), places=7)

    def test_gender_targets_change_shared_backbone_gradients(self):
        model = self.model(use_gender=True).eval()
        batch = next(iter(DataLoader(self.dataset(gender_txt=self.gender_path), batch_size=2)))
        parameter = model.get_parameter("ssl.feature_projection.projection.weight")
        torch.manual_seed(7)
        original, = torch.autograd.grad(model(*batch), parameter)
        torch.manual_seed(7)
        swapped, = torch.autograd.grad(model(*batch[:4], gender_emb=batch[4].flip(0)), parameter)

        self.assertGreater((original - swapped).abs().max().item(), 1e-9)

    def test_disabled_gender_preserves_the_original_four_input_loss(self):
        model = self.model().eval()
        batch = next(iter(DataLoader(self.dataset(), batch_size=2)))
        torch.manual_seed(7)
        original = model(*batch[:4])
        torch.manual_seed(7)
        current = model(*batch)

        torch.testing.assert_close(current, original, rtol=0, atol=0)

    def test_gender_training_supports_256_dimensional_prosody(self):
        model = self.model(use_gender=True, prosody_dim=256)
        wav, speaker, prosody, spk_ids, gender = next(iter(DataLoader(
            self.dataset(gender_txt=self.gender_path), batch_size=2,
        )))

        loss = model(wav, speaker, torch.cat([prosody, prosody], dim=-1), spk_ids, gender)

        self.assertTrue(torch.isfinite(loss).item())

    def test_enabled_model_requires_two_gender_coordinates_per_sample(self):
        model = self.model(use_gender=True)
        batch = next(iter(DataLoader(self.dataset(), batch_size=2)))
        for gender in (None, batch[4], torch.zeros(2, 3)):
            with self.subTest(gender=gender):
                with self.assertRaisesRegex(ValueError, r"gender_emb.*\[B, 2\]"):
                    model(*batch[:4], gender_emb=gender)

    def test_train_epoch_updates_gender_rows_in_shared_projection(self):
        model = self.model(use_gender=True)
        loader = DataLoader(self.dataset(gender_txt=self.gender_path), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.state_dict()["final_proj.weight"][320:].clone()

        train_epoch(loader, model, optimizer, torch.device("cpu"))

        self.assertFalse(torch.equal(before, model.state_dict()["final_proj.weight"][320:]))

    def test_validation_accepts_batches_with_optional_gender(self):
        for enabled in (False, True):
            with self.subTest(use_gender=enabled):
                model = self.model(use_gender=enabled)
                dataset = self.dataset(gender_txt=self.gender_path if enabled else None)
                loss = validate(DataLoader(dataset, batch_size=2), model, torch.device("cpu"))

                self.assertTrue(math.isfinite(loss))

    def run_cli(self, *args):
        return subprocess.run([
            sys.executable, "main_stage1real.py",
            "--model_name", str(self.backbone),
            "--train_prosody_txt", str(self.pros_path),
            "--dev_prosody_txt", str(self.pros_path),
            "--train_spkmean_txt", str(self.spk_path),
            "--dev_spkmean_txt", str(self.spk_path),
            "--wav_dir_train", str(self.root), "--wav_dir_dev", str(self.root),
            "--epochs", "1", "--batch_size", "2", "--mask_prob", "0.02",
            "--log_dir", str(self.root / "run"), *args,
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled",
                 "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            timeout=60)

    def test_cli_requires_metadata_when_gender_is_enabled(self):
        result = self.run_cli("--use_gender")

        self.assertIn("--gender_txt is required", result.stderr)

    def test_cli_trains_and_saves_a_gender_enabled_checkpoint(self):
        result = self.run_cli(
            "--use_gender", "--gender_txt", str(self.gender_path),
            "--gender_theta", "45",
        )
        if result.returncode:
            self.fail(result.stdout + result.stderr)
        self.assertIn("Gender theta: 45 deg", result.stdout)

        state = torch.load(self.root / "run/model_last.pth", map_location="cpu", weights_only=True)
        model = self.model(use_gender=True)
        model.load_state_dict(state, strict=True)
        batch = next(iter(DataLoader(self.dataset(gender_txt=self.gender_path), batch_size=2)))
        self.assertTrue(torch.isfinite(model(*batch)).item())

    def test_cli_without_gender_saves_a_legacy_compatible_checkpoint(self):
        result = self.run_cli()
        if result.returncode:
            self.fail(result.stdout + result.stderr)

        state = torch.load(self.root / "run/model_last.pth", map_location="cpu", weights_only=True)
        model = self.model()
        model.load_state_dict(state, strict=True)
        self.assertEqual(model.out_dim, 320)

    def test_cli_default_rotation_matches_explicit_45_degrees(self):
        states = []
        for options in ((), ("--gender_theta", "45")):
            result = self.run_cli("--use_gender", "--gender_txt", str(self.gender_path), *options)
            if result.returncode:
                self.fail(result.stdout + result.stderr)
            states.append(torch.load(self.root / "run/model_last.pth", map_location="cpu", weights_only=True))

        torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)

    def test_gender_checkpoint_transfers_backbone_and_speaker_prosody_projection_to_stage2(self):
        model = self.model(use_gender=True)
        loader = DataLoader(self.dataset(gender_txt=self.gender_path), batch_size=2)
        train_epoch(loader, model, torch.optim.SGD(model.parameters(), lr=0.01), torch.device("cpu"))
        state = model.state_dict()
        checkpoint = self.root / "stage1.pth"
        torch.save(state, checkpoint)
        expected = {key: value for key, value in state.items() if key.startswith(("ssl.", "final_proj."))}
        expected["final_proj.weight"] = state["final_proj.weight"][:320]
        expected["final_proj.bias"] = state["final_proj.bias"][:320]

        for model_class in (ProSDDStage2, ProSDDStage2Rhythm):
            with self.subTest(model_class=model_class.__name__):
                stage2 = model_class(model_name=str(self.backbone), stage1_ckpt=checkpoint)
                actual = {key: value for key, value in stage2.state_dict().items() if key in expected}

                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_stage2_can_backpropagate_after_loading_each_stage1_checkpoint_format(self):
        wav, speaker, prosody, spk_ids, _ = next(iter(DataLoader(self.dataset(), batch_size=2)))
        for prosody_dim, enabled in ((128, False), (128, True), (256, False), (256, True)):
            state = self.model(prosody_dim=prosody_dim, use_gender=enabled).state_dict()
            targets = prosody if prosody_dim == 128 else torch.cat([prosody, prosody], dim=-1)
            for wrapped in (False, True):
                checkpoint = self.root / "stage1.pth"
                payload = {"state_dict": {"module." + key: value for key, value in state.items()}} if wrapped else state
                torch.save(payload, checkpoint)
                for model_class in (ProSDDStage2, ProSDDStage2Rhythm):
                    with self.subTest(model_class=model_class.__name__, prosody_dim=prosody_dim,
                                      use_gender=enabled, wrapped=wrapped):
                        model = model_class(
                            model_name=str(self.backbone), stage1_ckpt=checkpoint,
                            prosody_dim=prosody_dim, mask_prob=0.02, mask_span_len=2,
                            num_time_neg=2, num_spk_neg=1,
                        )
                        kwargs = {"duration_features": torch.rand(2, 4, 9)} if model_class is ProSDDStage2Rhythm else {}
                        output = model(wav, speaker, targets, spk_ids, **kwargs)
                        loss = output["ssl_loss"] + torch.nn.functional.cross_entropy(
                            output["logits"], torch.tensor([0, 1]),
                        )
                        loss.backward()
                        gradient = model.final_proj.weight.grad

                        self.assertTrue(gradient is not None and torch.isfinite(gradient).all()
                                        and gradient.abs().sum() > 0)

    def test_stage2_gender_supervision_preserves_the_full_stage1_projection(self):
        state = self.model(prosody_dim=256, use_gender=True).state_dict()
        checkpoint = self.root / "stage1_gender.pth"
        torch.save(state, checkpoint)
        for model_class in (ProSDDStage2, ProSDDStage2Rhythm):
            with self.subTest(model_class=model_class.__name__):
                model = model_class(model_name=str(self.backbone), stage1_ckpt=checkpoint,
                                    prosody_dim=256, use_gender=True)

                torch.testing.assert_close(model.final_proj.state_dict(), {
                    "weight": state["final_proj.weight"], "bias": state["final_proj.bias"],
                }, rtol=0, atol=0)

    def test_gender_targets_change_stage2_shared_backbone_gradients(self):
        wav, speaker, prosody, spk_ids, gender = next(iter(DataLoader(
            self.dataset(gender_txt=self.gender_path), batch_size=2,
        )))
        for model_class in (ProSDDStage2, ProSDDStage2Rhythm):
            for dim in (128, 256):
                with self.subTest(model_class=model_class.__name__, prosody_dim=dim):
                    model = model_class(model_name=str(self.backbone), prosody_dim=dim,
                                        use_gender=True, mask_prob=0.02, mask_span_len=2,
                                        num_time_neg=2, num_spk_neg=1).eval()
                    kwargs = {"duration_features": torch.rand(2, 4, 9)} if model_class is ProSDDStage2Rhythm else {}
                    target = prosody if dim == 128 else torch.cat([prosody, prosody], dim=-1)
                    parameter = model.get_parameter("ssl.feature_projection.projection.weight")
                    torch.manual_seed(7)
                    original, = torch.autograd.grad(model(wav, speaker, target, spk_ids,
                        gender_emb=gender, **kwargs)["ssl_loss"], parameter)
                    torch.manual_seed(7)
                    swapped, = torch.autograd.grad(model(wav, speaker, target, spk_ids,
                        gender_emb=gender.flip(0), **kwargs)["ssl_loss"], parameter)

                    self.assertGreater((original - swapped).abs().max().item(), 1e-9)

    def test_stage2_batch_includes_gender_for_bonafide_and_spoof(self):
        loader = DataLoader(self.stage2_dataset(gender_txt=self.gender_path, gender_theta=45),
                            batch_size=2, collate_fn=collate_stage2)

        batch = next(iter(loader))

        torch.testing.assert_close(batch[5], torch.tensor([
            [0.70710678, 0.70710678], [-0.70710678, 0.70710678],
        ]))

    def test_stage2_training_updates_the_gender_projection_rows(self):
        model = ProSDDStage2(model_name=str(self.backbone), use_gender=True,
                            mask_prob=0.02, mask_span_len=2, num_time_neg=2, num_spk_neg=1)
        loader = DataLoader(self.stage2_dataset(gender_txt=self.gender_path), batch_size=2,
                            collate_fn=collate_stage2)
        before = model.final_proj.weight[320:].detach().clone()
        train_stage2(loader, model, torch.optim.SGD(model.parameters(), lr=0.01),
                     torch.device("cpu"), 1, 0, 1.0, 0.2, torch.nn.CrossEntropyLoss())

        self.assertFalse(torch.equal(before, model.final_proj.weight[320:]))

    def test_stage2_rejects_missing_speaker_gender_labels(self):
        self.gender_path.write_text("17 | M\n")

        with self.assertRaisesRegex(ValueError, "Missing gender labels for speakers: 19"):
            self.stage2_dataset(gender_txt=self.gender_path)

    def test_stage2_enabled_gender_requires_targets_during_training(self):
        wav, speaker, prosody, spk_ids, _ = next(iter(DataLoader(self.dataset(), batch_size=2)))
        for model_class in (ProSDDStage2, ProSDDStage2Rhythm):
            with self.subTest(model=model_class.__name__):
                model = model_class(model_name=str(self.backbone), use_gender=True)
                options = {"duration_features": torch.rand(2, 4, 9)} if model_class is ProSDDStage2Rhythm else {}

                with self.assertRaisesRegex(ValueError, r"gender_emb.*\[B, 2\]"):
                    model(wav, speaker, prosody, spk_ids, **options)

    def test_stage2_validation_supports_disabled_gender(self):
        model = ProSDDStage2(model_name=str(self.backbone), mask_prob=0.02,
                            mask_span_len=2, num_time_neg=2, num_spk_neg=1)
        loader = DataLoader(self.stage2_dataset(), batch_size=2, collate_fn=collate_stage2)
        self.assertEqual(next(iter(loader))[-1].shape, (2, 0))

        metrics = validate_stage2(loader, model, torch.device("cpu"), 1.0, 0.2, torch.nn.CrossEntropyLoss())

        self.assertTrue(all(math.isfinite(value) for value in metrics))

    def run_stage2_cli(self, *args):
        protocol = self.root / "protocol.txt"
        protocol.write_text("17 17-1-0000 - - bonafide\n19 19-1-0000 - A01 spoof\n")
        return subprocess.run([
            sys.executable, "main_stage2realfake.py", "--model_name", str(self.backbone),
            "--train_list", str(protocol), "--dev_list", str(protocol),
            "--wav_dir_train", str(self.root), "--wav_dir_dev", str(self.root),
            "--spkmean_txt_train", str(self.spk_path), "--spkmean_txt_dev", str(self.spk_path),
            "--prosody_txt_train", str(self.pros_path), "--prosody_txt_dev", str(self.pros_path),
            "--epochs", "1", "--batch_size", "2", "--num_workers", "0", "--algo", "0",
            "--audio_seconds", "0.05", "--T_target", "200", "--mask_prob", "0.02",
            "--num_time_neg", "2", "--num_spk_neg", "1",
            "--log_dir", str(self.root / "stage2_run"), *args,
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled",
                 "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, timeout=60)

    def test_stage2_cli_saves_the_full_450_dimensional_projection(self):
        rows = []
        for line in self.pros_path.read_text().splitlines():
            utt, frames = line.split("\t")
            rows.append(utt + "\t" + "|".join(frame + "," + frame for frame in frames.split("|")))
        self.pros_path.write_text("\n".join(rows) + "\n")
        checkpoint = self.root / "stage1_gender.pth"
        torch.save(self.model(prosody_dim=256, use_gender=True).state_dict(), checkpoint)

        result = self.run_stage2_cli("--use_gender", "--gender_txt", str(self.gender_path),
                                    "--gender_theta", "45", "--stage1_ckpt", str(checkpoint))
        if result.returncode:
            self.fail(result.stdout + result.stderr)
        state = torch.load(self.root / "stage2_run/model_best.pth", weights_only=True)

        self.assertEqual(tuple(state["final_proj.weight"].shape), (450, 8))

    def test_stage2_cli_requires_gender_metadata(self):
        result = self.run_stage2_cli("--use_gender")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--gender_txt is required", result.stderr)

    def test_rhythm_batches_train_the_gender_projection(self):
        samples = []
        for index, sample in enumerate(self.stage2_dataset(gender_txt=self.gender_path)):
            wav, speaker, prosody, speaker_id, label, gender = sample
            samples.append((wav, speaker, prosody, speaker_id, label,
                            torch.rand(4, 9), (0, wav.numel()), str(index), gender))
        batch = collate_stage2_rhythm(samples, conv_kernel=(4,), conv_stride=(4,))
        model = ProSDDStage2Rhythm(
            model_name=str(self.backbone), use_gender=True, mask_prob=0.02,
            mask_span_len=2, num_time_neg=2, num_spk_neg=1,
        )
        before = model.final_proj.weight[320:].detach().clone()

        run_rhythm_epoch([batch], model, torch.device("cpu"), torch.nn.CrossEntropyLoss(),
                         alpha=1.0, beta=0.2, optimizer=torch.optim.SGD(model.parameters(), lr=0.01))

        self.assertFalse(torch.equal(before, model.final_proj.weight[320:]))

    def test_standard_evaluation_loads_gender_checkpoint_without_gender_inputs(self):
        model = ProSDDStage2(model_name=str(self.backbone), use_gender=True, prosody_dim=256).eval()
        checkpoint = self.root / "stage2.pth"
        torch.save(model.state_dict(), checkpoint)
        restored = main_eval.load_model(checkpoint, model_name=str(self.backbone))
        audio = torch.randn(2, 800)

        torch.testing.assert_close(main_eval.inference_forward(restored, audio),
                                   main_eval.inference_forward(model, audio))

    def test_standard_evaluation_preserves_attention_classification_with_optional_gender(self):
        audio = torch.randn(2, 800)
        for enabled in (False, True):
            with self.subTest(use_gender=enabled):
                model = ProSDDStage2(model_name=str(self.backbone), use_gender=enabled,
                                    classifier_pool="attn", prosody_dim=256).eval()
                checkpoint = self.root / "stage2_attn.pth"
                torch.save({"state_dict": {"module." + key: value for key, value in model.state_dict().items()}}, checkpoint)

                restored = main_eval.load_model(checkpoint, model_name=str(self.backbone), classifier_pool="attn")

                torch.testing.assert_close(main_eval.inference_forward(restored, audio),
                                           main_eval.inference_forward(model, audio))

    def test_rhythm_evaluation_loads_gender_checkpoint_without_gender_inputs(self):
        options = dict(model_name=str(self.backbone), prosody_dim=256, T_target=200,
                       rhythm_sources=["syllable"], nhead=2, n_rhythm_encoder_layers=1,
                       n_cls_encoder_layers=1, dropout=0.0, max_position_embeddings=500)
        model = ProSDDStage2Rhythm(**options, use_gender=True).eval()
        checkpoint = self.root / "stage2_rhythm.pth"
        torch.save(model.state_dict(), checkpoint)
        config = dict(options, use_gender=True, model_class="ProSDDStage2Rhythm", sample_rate=16000,
                      audio_mode="pause_crop", audio_seconds=0.05, target_samples=800)
        (self.root / "config.json").write_text(json.dumps(config))

        restored, _ = load_rhythm_model(checkpoint)
        audio, duration = torch.randn(2, 800), torch.rand(2, 4, 3)
        with torch.no_grad():
            expected = model(audio, duration_features=duration, compute_ssl=False)["logits"]
            actual = restored(audio, duration_features=duration, compute_ssl=False)["logits"]

        torch.testing.assert_close(actual, expected)

    def test_rhythm_cli_accepts_optional_gender_supervision(self):
        required = ["--" + name for name in (
            "train_list", "dev_list", "wav_dir_train", "wav_dir_dev", "spkmean_txt_train",
            "spkmean_txt_dev", "duration_csv_train", "duration_csv_dev", "stage1_ckpt",
        )]
        args = rhythm_parser().parse_args([
            value for flag in required for value in (flag, "unused")
        ] + ["--use_gender", "--gender_txt", str(self.gender_path), "--gender_theta", "45"])

        self.assertEqual((args.use_gender, args.gender_theta), (True, 45))

    def test_rhythm_cli_trains_and_reloads_a_full_gender_checkpoint(self):
        from masked_prosody_model import ModelArgs
        from mask_prosody_model.masked_prosody_model_vad import MaskedProsodyModelVAD

        teacher_args = ModelArgs(n_layers=1, filter_size=256, dropout=0.0)
        teacher_args.max_length = 256
        teacher_path = self.root / "teacher"
        MaskedProsodyModelVAD(teacher_args).save_model(teacher_path)
        checkpoint = self.root / "stage1.pth"
        torch.save(self.model(prosody_dim=256, use_gender=True).state_dict(), checkpoint)
        protocol = self.root / "protocol.txt"
        protocol.write_text("17 17-1-0000 - - bonafide\n19 19-1-0000 - A01 spoof\n")
        duration = self.root / "duration.csv"
        duration.write_text(
            "flac_file_name,duration_syllable,starttime_syllable,endtime_syllable,starttime_word,endtime_word\n"
            "17-1-0000.flac,0.04,0.005,0.045,0.005,0.045\n"
            "19-1-0000.flac,0.04,0.005,0.045,0.005,0.045\n"
        )
        result = subprocess.run([
            sys.executable, "main_stage2realfake_rhythm.py", "--model_name", str(self.backbone),
            "--train_list", str(protocol), "--dev_list", str(protocol),
            "--wav_dir_train", str(self.root), "--wav_dir_dev", str(self.root),
            "--spkmean_txt_train", str(self.spk_path), "--spkmean_txt_dev", str(self.spk_path),
            "--duration_csv_train", str(duration), "--duration_csv_dev", str(duration),
            "--stage1_ckpt", str(checkpoint), "--teacher_kind", "vad",
            "--teacher_checkpoint", str(teacher_path), "--prosody_layer", "0",
            "--epochs", "1", "--batch_size", "2", "--num_workers", "0", "--algo", "0",
            "--audio_seconds", "0.05", "--mask_prob", "0.02", "--mask_span_len", "2",
            "--num_time_neg", "2", "--num_spk_neg", "1", "--nhead", "2",
            "--n_rhythm_encoder_layers", "1", "--n_cls_encoder_layers", "1",
            "--use_gender", "--gender_txt", str(self.gender_path), "--gender_theta", "45",
            "--wandb_mode", "disabled", "--log_dir", str(self.root / "rhythm_run"),
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, timeout=60)
        if result.returncode:
            self.fail(result.stdout + result.stderr)

        restored, config = load_rhythm_model(self.root / "rhythm_run/model_best.pth")

        self.assertEqual(restored.final_proj.out_features, 450)
        self.assertTrue(config["use_gender"])


if __name__ == "__main__":
    unittest.main()
