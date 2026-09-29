"""以真實音訊驗證 notebook 的有效長度統計，不需要完整資料集。"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "asvspoof_rhythm_crop_lengths.ipynb"


class RhythmCropNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "analysis" in cell.get("metadata", {}).get("tags", [])
        )
        code_path = ROOT / "output/notebook-validation/rhythm_crop_analysis.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(source, encoding="utf-8")
        cls.api = {}
        exec(compile(source, str(code_path), "exec"), cls.api)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wav_dir = Path(self.tmp.name)
        sf.write(self.wav_dir / "short.flac", np.zeros(32000), 16000)
        self.record = {
            "duration": torch.tensor([[1.0]], dtype=torch.float64),
            "syllable": torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            "word": torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        }

    def test_short_audio_length_excludes_center_padding(self):
        rows, _ = self.api["measure_crops"](
            "short", self.record, wav_dir=self.wav_dir,
            crop_seconds=(4.0,), seed=1234,
        )
        self.assertEqual(rows[0]["actual_sec"], 2.0)

    def test_original_length_reads_the_full_recording_without_cropping(self):
        sf.write(self.wav_dir / "long.flac", np.zeros(96000), 16000)
        seconds = self.api["original_length"](self.wav_dir / "long.flac")
        self.assertEqual(seconds, 6.0)

    def test_original_lengths_keep_each_crop_target_and_utterance_paired(self):
        lengths, _, _ = self.api["collect_dataset"](
            "Example", {"train": self.make_split()}, (4.0, 10.0),
            splits=("train",), workers=1, progress=False,
        )
        enriched = self.api["add_original_lengths"](
            lengths, {"train": {"wav_dir": self.wav_dir}}, workers=2, progress=False,
        )
        self.assertEqual(
            enriched[["utt_id", "target_sec", "original_sec"]].values.tolist(),
            [["short", 4.0, 2.0], ["short", 10.0, 2.0]],
        )

    def test_pause_crop_can_be_longer_than_four_seconds(self):
        sf.write(self.wav_dir / "long.flac", np.zeros(96000), 16000)
        self.record["word"] = torch.tensor([[0.0, 6.0]], dtype=torch.float64)
        rows, _ = self.api["measure_crops"](
            "long", self.record, wav_dir=self.wav_dir,
            crop_seconds=(4.0,), seed=1234,
        )
        self.assertEqual(rows[0]["actual_sec"], 6.0)

    def test_ten_second_target_keeps_true_length_of_short_audio(self):
        rows, _ = self.api["measure_crops"](
            "short", self.record, wav_dir=self.wav_dir,
            crop_seconds=(10.0,), seed=1234,
        )
        self.assertEqual((rows[0]["actual_sec"], rows[0]["tensor_sec"]), (2.0, 10.0))

    def test_resampled_audio_duration_uses_the_output_sample_rate(self):
        sf.write(self.wav_dir / "short.flac", np.zeros(16000), 8000)
        rows, _ = self.api["measure_crops"](
            "short", self.record, wav_dir=self.wav_dir,
            crop_seconds=(4.0,), seed=1234,
        )
        self.assertEqual(rows[0]["actual_sec"], 2.0)

    def test_crop_seed_is_reproducible_when_target_order_changes(self):
        sf.write(self.wav_dir / "long.flac", np.zeros(192000), 16000)
        record = {
            "duration": torch.ones((4, 1), dtype=torch.float64),
            "syllable": torch.tensor([[0, 1], [3, 4], [6, 7], [9, 10]], dtype=torch.float64),
            "word": torch.tensor([[0, 2], [3, 5], [6, 8], [9, 12]], dtype=torch.float64),
        }
        first, _ = self.api["measure_crops"](
            "long", record, wav_dir=self.wav_dir, crop_seconds=(4.0, 10.0), seed=1234,
        )
        second, _ = self.api["measure_crops"](
            "long", record, wav_dir=self.wav_dir, crop_seconds=(10.0, 4.0), seed=1234,
        )
        self.assertEqual(first, second[::-1])

    def make_split(self):
        """建立一筆有效、一筆缺音訊及一筆缺 rhythm 的小型 protocol。"""
        protocol = self.wav_dir / "protocol.txt"
        protocol.write_text(
            "speaker short - - bonafide\n"
            "speaker missing_audio - - spoof\n"
            "speaker missing_rhythm - - spoof\n",
            encoding="utf-8",
        )
        duration_csv = self.wav_dir / "duration.csv"
        with duration_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "flac_file_name", "duration_syllable", "starttime_syllable",
                "endtime_syllable", "starttime_word", "endtime_word",
            ])
            writer.writeheader()
            for utt in ("short", "missing_audio"):
                writer.writerow({
                    "flac_file_name": f"{utt}.flac", "duration_syllable": "1.0",
                    "starttime_syllable": "0", "endtime_syllable": "1",
                    "starttime_word": "0", "endtime_word": "2",
                })
        return {"protocol": protocol, "wav_dir": self.wav_dir, "duration_csv": duration_csv}

    def test_split_accounts_for_every_protocol_entry_and_crop_target(self):
        _, _, audit = self.api["collect_split"](
            **self.make_split(), crop_seconds=(4.0, 10.0),
            seed=1234, workers=2, progress=False,
        )
        self.assertEqual(
            audit[["target_sec", "selected_count", "valid_count", "skipped_count"]].values.tolist(),
            [[4.0, 3, 1, 2], [10.0, 3, 1, 2]],
        )

    def test_subset_mode_reports_the_full_protocol_and_selected_counts(self):
        _, _, audit = self.api["collect_split"](
            **self.make_split(), crop_seconds=(4.0,),
            max_utterances=1, workers=1, progress=False,
        )
        self.assertEqual(
            audit[["protocol_count", "selected_count", "valid_count"]].values.tolist(),
            [[3, 1, 1]],
        )

    def test_missing_audio_directory_fails_before_collecting_samples(self):
        spec = self.make_split()
        spec["wav_dir"] = self.wav_dir / "nonexistent"
        with self.assertRaises(FileNotFoundError):
            self.api["collect_split"](**spec, crop_seconds=(4.0,), progress=False)

    def test_skip_report_distinguishes_missing_rhythm_from_unreadable_audio(self):
        _, errors, _ = self.api["collect_split"](
            **self.make_split(), crop_seconds=(4.0,), workers=2, progress=False,
        )
        self.assertEqual(
            dict(zip(errors["utt_id"], errors["stage"])),
            {"missing_rhythm": "duration_csv", "missing_audio": "audio_crop"},
        )

    def test_histogram_includes_samples_beyond_the_crop_target(self):
        frame = self.api["pd"].DataFrame({
            "dataset": ["Example"] * 3,
            "split": ["train", "dev", "eval"],
            "target_sec": [4.0] * 3,
            "actual_sec": [2.0, 4.0, 6.0],
            "original_sec": [6.0, 8.0, 10.0],
        })
        fig = self.api["plot_lengths"](frame, dataset="Example", target_sec=4.0)
        self.addCleanup(self.api["plt"].close, fig)
        self.assertGreaterEqual(fig.axes[0].get_xlim()[1], 6.0)

    def test_figure_has_eight_original_and_cropped_panels_with_matching_counts(self):
        frame = self.api["pd"].DataFrame({
            "dataset": ["Example"] * 3,
            "split": ["train", "dev", "eval"],
            "target_sec": [4.0] * 3,
            "actual_sec": [2.0, 4.0, 6.0],
            "original_sec": [6.0, 8.0, 10.0],
        })
        fig = self.api["plot_lengths"](frame, dataset="Example", target_sec=4.0)
        self.addCleanup(self.api["plt"].close, fig)
        self.assertEqual(
            [(ax.get_title(), sum(bar.get_height() for bar in ax.patches)) for ax in fig.axes],
            [("Original | Train", 1), ("Original | Dev", 1),
             ("Original | Eval", 1), ("Original | All splits", 3),
             ("Cropped (4 s) | Train", 1), ("Cropped (4 s) | Dev", 1),
             ("Cropped (4 s) | Eval", 1), ("Cropped (4 s) | All splits", 3)],
        )

    def test_top_row_plots_original_length_and_bottom_row_plots_cropped_length(self):
        frame = self.api["pd"].DataFrame({
            "dataset": ["Example"], "split": ["train"], "target_sec": [4.0],
            "original_sec": [6.0], "actual_sec": [2.0],
        })
        fig = self.api["plot_lengths"](frame, dataset="Example", target_sec=4.0)
        self.addCleanup(self.api["plt"].close, fig)
        occupied = [next(bar for bar in fig.axes[index].patches if bar.get_height())
                    for index in (0, 4)]
        self.assertTrue(all(
            bar.get_x() <= expected <= bar.get_x() + bar.get_width() + 1e-9
            for bar, expected in zip(occupied, (6.0, 2.0))
        ))

    def test_summary_uses_actual_duration_and_counts_target_deviations(self):
        frame = self.api["pd"].DataFrame({
            "dataset": ["Example"] * 3,
            "split": ["train", "dev", "eval"],
            "target_sec": [4.0] * 3,
            "valid_samples": [32000, 64000, 96000],
            "tensor_samples": [64000, 64000, 96000],
            "actual_sec": [2.0, 4.0, 6.0],
        })
        summary = self.api["summarize_lengths"](frame).set_index("split").loc["All"]
        self.assertEqual(
            summary[["count", "mean_sec", "shorter_count", "exact_count", "longer_count"]].to_dict(),
            {"count": 3, "mean_sec": 4.0, "shorter_count": 1, "exact_count": 1, "longer_count": 1},
        )

    def test_dataset_keeps_split_identity_when_combining_results(self):
        spec = self.make_split()
        lengths, _, _ = self.api["collect_dataset"](
            "Example", {"train": spec, "dev": spec}, (4.0,),
            splits=("train", "dev"), workers=2, progress=False,
        )
        self.assertEqual(
            lengths[["dataset", "split", "actual_sec"]].values.tolist(),
            [["Example", "train", 2.0], ["Example", "dev", 2.0]],
        )

    def test_saved_lengths_preserve_true_duration_for_both_targets(self):
        result = self.api["collect_dataset"](
            "Example", {"train": self.make_split()}, (4.0, 10.0),
            splits=("train",), workers=1, progress=False,
        )
        output_dir = self.wav_dir / "results"
        self.api["save_results"]("example", output_dir, *result)
        saved = self.api["pd"].read_csv(output_dir / "example_lengths.csv.gz")
        self.assertEqual(saved[["target_sec", "actual_sec"]].values.tolist(), [[4.0, 2.0], [10.0, 2.0]])

    def test_saved_crop_results_can_be_reloaded_without_audio_decoding(self):
        result = self.api["collect_dataset"](
            "Example", {"train": self.make_split()}, (4.0,),
            splits=("train",), workers=1, progress=False,
        )
        self.api["save_results"]("example", self.wav_dir, *result)
        (self.wav_dir / "short.flac").unlink()
        lengths, _, _ = self.api["collect_dataset"](
            "Example", {}, (4.0,), cache_prefix=self.wav_dir / "example", progress=False,
        )
        self.assertEqual(lengths["actual_sec"].tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
