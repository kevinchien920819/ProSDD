"""驗證裁切後 prosody 的實際時間與 CNN frame 對齊。"""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from extract_Prosody_rhythm import extract_prosody_rhythm
import extract_Prosody_rhythm as extraction


class TimeTeacher(torch.nn.Module):
    """以時間作為外部 teacher 輸出，提供可手算的對齊基準。"""

    vad_measure = SimpleNamespace(hop_length=256, sampling_rate=22050)

    def process_audio(self, path, layer):
        wav, sr = sf.read(path)
        times = np.arange(int(len(wav) / sr * 22050 / 256) + 1) * (256 / 22050)
        return np.repeat(times[:, None], 128, axis=1)


class ProsodyRhythmTests(unittest.TestCase):
    def test_four_second_crop_targets_use_actual_cnn_frame_centers(self):
        targets = extract_prosody_rhythm(TimeTeacher(), torch.ones(64000))

        torch.testing.assert_close(
            targets[[0, -1], 0], torch.tensor([0.01246875, 3.97246875]),
        )

    def test_sampling_rate_controls_the_time_of_each_cnn_frame(self):
        targets = extract_prosody_rhythm(TimeTeacher(), torch.ones(32000), sr=8000)

        torch.testing.assert_close(
            targets[[0, -1], 0], torch.tensor([0.0249375, 3.9449375]),
        )

    def test_cli_extracts_only_the_requested_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav, output = Path(tmp) / "audio.wav", Path(tmp) / "targets.pt"
            sf.write(wav, np.ones(7 * 16000, dtype=np.float32), 16000)
            with patch("masked_prosody_model.MaskedProsodyModel.from_pretrained", return_value=TimeTeacher()), \
                    patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
                        conv_kernel=(400,), conv_stride=(320,))):
                extraction.main([
                    "--audio_path", str(wav), "--start", "1", "--end", "5",
                    "--out_pt", str(output),
                ])

            self.assertEqual(torch.load(output, weights_only=True).shape, (199, 128))


if __name__ == "__main__":
    unittest.main()
