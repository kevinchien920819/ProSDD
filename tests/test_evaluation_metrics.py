import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation_metric.calculate_metrics import calculate_minDCF_EER_CLLR_actDCF
from evaluation_metric.calculate_modules import calculate_CLLR
from evaluation_metric.prosdd import evaluate_score_file, load_protocol_labels


ROOT = Path(__file__).resolve().parents[1]


class EvaluationMetricTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.scores = self.directory / "scores.txt"
        self.protocol = self.directory / "protocol.txt"
        self.output = self.directory / "results with spaces" / "metrics.json"
        value = math.log(3)
        self.scores.write_text(f"a {value}\nb {-value}\nc {value}\nd {-value}\n")
        self.protocol.write_text("sp d - A01 spoof\nsp c - - bonafide\nsp b - A01 spoof\nsp a - - bonafide\n")

    def test_known_metrics_align_shuffled_ids_and_write_json(self):
        result = evaluate_score_file(self.scores, self.protocol, self.output)
        self.assertEqual(result["eer"], 0.0)
        self.assertEqual(result["eer_percent"], 0.0)
        self.assertEqual(result["min_dcf"], 0.0)
        self.assertEqual(result["act_dcf"], 0.0)
        self.assertAlmostEqual(result["cllr"], math.log2(4 / 3))
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["bonafide_count"], 2)
        self.assertEqual(result["spoof_count"], 2)
        self.assertEqual(json.loads(self.output.read_text()), result)

    def test_asvspoof5_uses_label_column_not_attack_column(self):
        self.protocol.write_text(
            "sp a F - - - - bonafide bonafide -\n"
            "sp b M C05 2 source AC1 A26 spoof -\n"
            "sp c F - - - - bonafide bonafide -\n"
            "sp d M C05 2 source AC1 A27 spoof -\n"
        )
        self.assertEqual(evaluate_score_file(self.scores, self.protocol)["eer"], 0.0)

    def test_two_column_and_named_keys_and_score_header(self):
        for header in ("", "filename\tcm-label\n", "trial_anon\tcm_label\n"):
            with self.subTest(header=header):
                self.protocol.write_text(header + "d\tspoof\nc\tbonafide\na\tbonafide\nb\tspoof\n")
                self.assertEqual(evaluate_score_file(self.scores, self.protocol)["eer"], 0.0)
        self.scores.write_text("filename\tcm-score\n" + self.scores.read_text())
        self.assertEqual(evaluate_score_file(self.scores, self.protocol)["eer"], 0.0)

    def test_rejects_missing_extra_duplicate_and_invalid_scores(self):
        original = self.scores.read_text()
        cases = (
            ("a 1\nb -1\nc 1\n", "missing scores"),
            (original + "extra 1\n", "scores without labels"),
            (original + "a 1\n", "duplicate utterance ID a"),
            ("a nan\n", "must be finite"),
            ("a inf\n", "must be finite"),
            ("a unknown\n", "non-numeric"),
            ("a 1 label\n", "expected two columns"),
            ("\n", "no scores"),
        )
        for content, message in cases:
            with self.subTest(message=message):
                self.scores.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_score_file(self.scores, self.protocol, self.output)
                self.assertFalse(self.output.exists())

    def test_rejects_invalid_or_incomplete_protocols(self):
        cases = (
            ("a bonafide\na spoof\n", "duplicate utterance ID a"),
            ("a bonafide\nb unknown\n", "invalid CM label"),
            ("a bonafide\nb bonafide\n", "both bonafide and spoof"),
            ("filename cm-label\n", "no CM labels"),
            ("sp a\nsp b - A01 spoof\n", "invalid CM label"),
            ("a bonafide\nsp b - A01 spoof\n", "inconsistent protocol column count"),
            ("a extra bonafide\n", "expected 2-column"),
        )
        for content, message in cases:
            with self.subTest(message=message):
                self.protocol.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    load_protocol_labels(self.protocol)

    def test_metrics_output_cannot_overwrite_inputs(self):
        for path in (self.scores, self.protocol):
            with self.subTest(path=path):
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "must differ"):
                    evaluate_score_file(self.scores, self.protocol, path)
                self.assertEqual(path.read_bytes(), original)

    def test_cllr_is_finite_for_extreme_scores_and_one_bit_for_zero_llr(self):
        with np.errstate(over="raise", invalid="raise"):
            self.assertEqual(calculate_CLLR([0], [0]), 1.0)
            self.assertEqual(calculate_CLLR([1000], [-1000]), 0.0)
            self.assertAlmostEqual(calculate_CLLR([-1000], [1000]), 1000 / math.log(2))

    def test_imported_report_writer_handles_spaces(self):
        output = self.directory / "CM report.txt"
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            calculate_minDCF_EER_CLLR_actDCF(
                np.array([1.0, -1.0]), np.array(["bonafide", "spoof"]), output,
            )
        self.assertIn("CLLR", output.read_text())
        self.assertEqual(captured.getvalue(), output.read_text())

    def test_cli_evaluates_saved_scores_without_loading_torch(self):
        command = [sys.executable, "-m", "evaluation_metric", "--score_path", str(self.scores),
                   "--protocol_path", str(self.protocol), "--save_metrics_to", str(self.output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("EER=0.000000%", result.stdout)
        self.assertAlmostEqual(json.loads(self.output.read_text())["cllr"], math.log2(4 / 3))
        probe = subprocess.run(
            [sys.executable, "-c", "import sys; import evaluation_metric.prosdd; assert 'torch' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)


if __name__ == "__main__":
    unittest.main()
