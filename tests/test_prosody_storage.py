import gc
import multiprocessing
import pickle
import tempfile
import tracemalloc
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data_utils_stage2realfake import ProSDDStage2Dataset
from prosody_utils import load_prosody_dict


class ProsodySamples(Dataset):
    def __init__(self, targets):
        self.targets = targets
        self.ids = list(targets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.targets[self.ids[index]]


class ProsodyStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "prosody.txt"

    def write_targets(self, count=64, frames=200, dim=256):
        with self.path.open("w", encoding="utf-8") as file:
            for index in range(count):
                frame = ",".join([str(index / count)] * dim)
                file.write(f"utt_{index}\t" + "|".join([frame] * frames) + "\n")

    def test_stage2_initialization_does_not_retain_all_prosody_tensors(self):
        # Exercise the actual Dataset constructor, before DataLoader or a model
        # exists. Eager loading retains 12.5 MiB for these 64 real-sized targets.
        self.write_targets()
        speaker = self.root / "speakers.txt"
        speaker.write_text("s1 " + " ".join(["0.1"] * 192) + "\n")
        gc.collect()
        tracemalloc.start()
        try:
            dataset = ProSDDStage2Dataset(
                [f"utt_{i}" for i in range(64)], ["s1"] * 64, [0, 1] * 32,
                str(self.root), str(speaker), str(self.path),
            )
            current, peak = tracemalloc.get_traced_memory()
            self.assertLess(current, 4 * 1024**2, f"Retained {current / 1024**2:.2f} MiB")
            self.assertLess(peak, 6 * 1024**2, f"Peak {peak / 1024**2:.2f} MiB")
            self.assertEqual(dataset.prosody_dim, 256)
            self.assertEqual(dataset.utt2pros["utt_63"].shape, (200, 256))
        finally:
            tracemalloc.stop()

    def test_random_access_preserves_values_and_does_not_cache_mutations(self):
        self.write_targets(count=4, frames=3, dim=128)
        targets = load_prosody_dict(self.path)
        for index in (3, 0, 2, 1, 3):
            tensor = targets[f"utt_{index}"]
            torch.testing.assert_close(tensor, torch.full((3, 128), index / 4))
            tensor.fill_(-100)
        with self.assertRaises(KeyError):
            targets["missing"]
        self.assertNotIn("missing", targets)

    def test_pickling_and_workers_preserve_random_access(self):
        self.write_targets(count=8, frames=3, dim=128)
        targets = pickle.loads(pickle.dumps(load_prosody_dict(self.path)))
        # A parent read before forking must not leave a shared seek position.
        torch.testing.assert_close(targets["utt_7"], torch.full((3, 128), 7 / 8))
        for context in ("fork", "spawn"):
            if context not in multiprocessing.get_all_start_methods():
                continue
            with self.subTest(context=context):
                loader = DataLoader(
                    ProsodySamples(targets), batch_size=2, num_workers=2,
                    multiprocessing_context=context, timeout=30,
                )
                result = torch.cat(list(loader))
                expected = (torch.arange(8) / 8)[:, None, None].expand(8, 3, 128)
                torch.testing.assert_close(result, expected)

    def test_blank_lines_unicode_and_last_duplicate_match_dictionary_reads(self):
        frame = ",".join(["0.25"] * 128)
        self.path.write_text(f"\n\t \n語音\t{frame}\n\n語音\t{frame}|{frame}", encoding="utf-8")
        targets = load_prosody_dict(self.path)
        self.assertEqual(list(targets), ["語音"])
        self.assertEqual(targets["語音"].shape, (2, 128))

    def test_changed_source_is_rejected_instead_of_reading_stale_offsets(self):
        self.write_targets(count=2, frames=3, dim=128)
        targets = load_prosody_dict(self.path)
        self.write_targets(count=3, frames=3, dim=128)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            targets["utt_0"]


if __name__ == "__main__":
    unittest.main()
