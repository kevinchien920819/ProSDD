import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "train_rhythm.sh"


class TrainRhythmScriptTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.log_dir = self.root / "training logs"
        self.calls_file = self.root / "uv-calls.jsonl"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        fake_uv = bin_dir / "uv"
        fake_uv.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['RHYTHM_UV_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "key = 'RHYTHM_TRAIN_EXIT' if 'main_stage2realfake_rhythm.py' in sys.argv else 'RHYTHM_EVAL_EXIT'\n"
            "sys.exit(int(os.environ.get(key, '0')))\n"
        )
        fake_uv.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RHYTHM_UV_CALLS": str(self.calls_file),
            "RHYTHM_LOG_DIR": str(self.log_dir),
            "RHYTHM_TRAIN_EXIT": "0", "RHYTHM_EVAL_EXIT": "0",
        }

    def run_script(self, **env):
        result = subprocess.run(["bash", str(SCRIPT)], cwd=self.root, env={**self.env, **env},
                                text=True, capture_output=True)
        calls = [json.loads(line) for line in self.calls_file.read_text().splitlines()]
        return result, calls

    def test_script_evaluates_the_checkpoint_from_the_completed_training_run(self):
        result, calls = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[:4] for call in calls], [
            ["run", "--locked", "python", "main_stage2realfake_rhythm.py"],
            ["run", "--locked", "python", "main__eval_rhythm.py"],
        ])
        train, evaluation = calls
        self.assertEqual(train[train.index("--log_dir") + 1], str(self.log_dir))
        expected = {
            "--model_path": str(self.log_dir / "model_best.pth"),
            "--config_path": str(self.log_dir / "config.json"),
            "--list_path": "dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
            "--wav_dir": "dataset/ASVspoof2019/ASVspoof2019_LA_eval/flac",
            "--duration_csv": "dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_eval.csv",
            "--save_scores_to": str(self.log_dir / "eval/asvspoof2019_la_eval.txt"),
            "--save_metrics_to": str(self.log_dir / "eval/asvspoof2019_la_eval.metrics.json"),
        }
        actual = {flag: evaluation[evaluation.index(flag) + 1] for flag in expected}
        self.assertEqual(actual, expected)

    def test_failed_training_does_not_launch_evaluation(self):
        result, calls = self.run_script(RHYTHM_TRAIN_EXIT="17")

        self.assertEqual((result.returncode, [call[3] for call in calls]),
                         (17, ["main_stage2realfake_rhythm.py"]))

    def test_evaluation_failure_propagates_to_the_script_exit_status(self):
        result, _ = self.run_script(RHYTHM_EVAL_EXIT="23")

        self.assertEqual(result.returncode, 23)


if __name__ == "__main__":
    unittest.main()
