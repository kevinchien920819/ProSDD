"""驗證 Stage 1 gender 標註載入與旋轉後的監督向量。"""

from pathlib import Path
import tempfile
import unittest

import torch

from data_utils_stage1real import load_gender


class LoadGenderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "SPEAKERS.TXT"
        self.path.write_text(
            "; LibriSpeech reader metadata\n"
            ";ID | SEX | SUBSET | MINUTES | NAME\n"
            "\n"
            "17 | M | train-clean-100 | 25.0 | Example reader\n"
            "19 | F | dev-clean | 20.0 | Another reader\n",
            encoding="utf-8",
        )

    def test_zero_angle_loads_one_hot_targets_from_librispeech_metadata(self):
        actual = load_gender(self.path, theta=0.0)

        torch.testing.assert_close(actual, {
            "17": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "19": torch.tensor([0.0, 1.0], dtype=torch.float32),
        })

    def test_quarter_turn_rotates_both_genders_counterclockwise(self):
        actual = load_gender(self.path, theta=90)

        torch.testing.assert_close(actual, {
            "17": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "19": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        })

    def test_each_load_rotates_original_targets_by_the_given_angle(self):
        load_gender(self.path, theta=90)

        actual = load_gender(self.path, theta=45)

        torch.testing.assert_close(actual, {
            "17": torch.tensor([0.70710678, 0.70710678], dtype=torch.float32),
            "19": torch.tensor([-0.70710678, 0.70710678], dtype=torch.float32),
        })

    def test_unknown_gender_reports_the_metadata_line(self):
        self.path.write_text("17 | X | train-clean-100 | 25.0 | Reader\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, r"SPEAKERS\.TXT:1:.*'X'"):
            load_gender(self.path, theta=0.0)


if __name__ == "__main__":
    unittest.main()
