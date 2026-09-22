import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from extract_full_prosody import extract_aligned_targets
import extract_full_prosody as extraction
from full_utterance import LengthBatchSampler, read_full_prosody_metadata


class FullUtteranceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.protocol = self.root / "protocol.txt"
        self.protocol.write_text("s1 long - A01 spoof\ns2 short - - bonafide\n")
        for utt, seconds in (("long", 6), ("short", 1)):
            sf.write(self.root / f"{utt}.wav", np.ones(seconds * 16000) * .1, 16000)

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



if __name__ == "__main__":
    unittest.main()
