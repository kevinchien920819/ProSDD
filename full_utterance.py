"""Shared full-utterance timing, cache contract and length-aware batching."""

import json
from pathlib import Path

import torch
from torch.utils.data import Sampler


def cnn_geometry(conv_kernel, conv_stride):
    if (not conv_kernel or len(conv_kernel) != len(conv_stride)
            or min(*conv_kernel, *conv_stride) < 1):
        raise ValueError("Expected matching nonempty CNN kernel/stride sequences")
    stride, receptive_field = 1, 1
    for kernel, step in zip(conv_kernel, conv_stride):
        receptive_field += (kernel - 1) * stride
        stride *= step
    return stride, receptive_field


def frame_count(samples, stride, receptive_field):
    return max(0, (samples - receptive_field) // stride + 1)


def frame_padding_mask(lengths, conv_kernel, conv_stride):
    stride, receptive_field = cnn_geometry(conv_kernel, conv_stride)
    counts = torch.tensor([frame_count(n, stride, receptive_field) for n in lengths])
    if (counts < 1).any():
        raise ValueError("Audio too short for a valid CNN frame")
    return torch.arange(int(counts.max()))[None] >= counts[:, None]


def read_full_prosody_metadata(path):
    path = Path(path)
    metadata_path = Path(str(path) + ".meta.json")
    if not metadata_path.is_file():
        raise ValueError(
            f"Full-utterance prosody metadata missing: {metadata_path}. "
            "Regenerate targets with extract_full_prosody.py; four-second caches cannot be reused."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("audio_mode") != "full_utterance" or metadata.get("sample_rate") != 16000
            or metadata.get("alignment") != "cnn_receptive_field_centers"
            or metadata.get("schema_version") != 1):
        raise ValueError(f"Incompatible full-utterance prosody metadata: {metadata_path}")
    cnn_geometry(metadata.get("conv_kernel"), metadata.get("conv_stride"))
    if metadata.get("cache_size_bytes") != path.stat().st_size:
        raise ValueError(f"Prosody cache differs from its full-utterance metadata: {path}")
    if not isinstance(metadata.get("utterances"), dict) or not metadata["utterances"]:
        raise ValueError(f"No utterance lengths in {metadata_path}")
    return metadata


class LengthBatchSampler(Sampler):
    """Bound padded samples per batch, never truncate or drop an utterance.

    An utterance larger than the budget is emitted alone. Training shuffles
    length buckets and batch order reproducibly; evaluation is length sorted.
    """

    def __init__(self, lengths, batch_size, max_batch_samples, *, shuffle=False, seed=1234):
        self.lengths = list(lengths)
        if not self.lengths or min(self.lengths) < 1 or batch_size < 1 or max_batch_samples < 1:
            raise ValueError("Lengths, batch_size and max_batch_samples must be positive")
        self.batch_size = batch_size
        self.max_batch_samples = max_batch_samples
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        if self.shuffle:
            # Randomize near-length neighbours without mixing the entire corpus.
            width = self.batch_size * 20
            order = [bucket[i] for start in range(0, len(order), width)
                     for bucket in [order[start:start + width]]
                     for i in torch.randperm(len(bucket), generator=generator).tolist()]
        batches, batch, longest = [], [], 0
        for index in order:
            next_longest = max(longest, self.lengths[index])
            if batch and (len(batch) >= self.batch_size
                          or next_longest * (len(batch) + 1) > self.max_batch_samples):
                batches.append(batch)
                batch, longest = [], 0
            batch.append(index)
            longest = max(longest, self.lengths[index])
        if batch:
            batches.append(batch)
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())
