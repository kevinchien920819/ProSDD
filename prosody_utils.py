from collections.abc import Mapping
import os
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import torch


SUPPORTED_PROSODY_DIMS = (128, 256)


class ProsodySourceChangedError(RuntimeError):
    """The whole index is stale; this is not a bad individual sample."""


def _parse_prosody_line(line: str) -> Tuple[str, np.ndarray]:
    utt, frames_str = line.strip().split("\t")
    # NumPy 1.x otherwise accepts numeric prefixes such as "0.1broken" and
    # only warns. Such values must become a sample error, not a valid target.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        try:
            frames = [np.fromstring(frame, sep=",", dtype=np.float32)
                      for frame in frames_str.split("|")]
        except DeprecationWarning as exc:
            raise ValueError(f"Invalid numeric prosody target for {utt}: {exc}") from exc
    return utt, np.stack(frames, axis=0)  # (T, D)


def _source_signature(file):
    stat = os.fstat(file.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class ProsodyTextIndex(Mapping):
    """Read-only utterance mapping with tensors loaded only on access.

    Construction scans the text once, retaining byte offsets and frame counts
    instead of every float32 tensor. Shapes are checked without parsing floats;
    numeric parsing happens when a sample is requested. No file handles or
    tensor cache are kept, so fork/spawn DataLoader workers read independently.
    """

    def __init__(self, prosody_txt, expected_dim=None, *, skip_bad_entries=False):
        if expected_dim is not None and expected_dim not in SUPPORTED_PROSODY_DIMS:
            raise ValueError(f"Unsupported prosody dim {expected_dim}; expected 128 or 256")
        self.path = os.path.abspath(prosody_txt)
        self._entries = {}
        self.errors = {}
        self.prosody_dim = expected_dim
        with open(self.path, "rb") as file:
            self._signature = _source_signature(file)
            size = self._signature[2]
            report_progress = size >= 1024**3
            if report_progress:
                print(f"Indexing prosody: {self.path} ({size / 1024**3:.1f} GiB)", flush=True)
            offset, next_report = 0, 16 * 1024**3
            for line_number, raw in enumerate(file, 1):
                start, offset = offset, offset + len(raw)
                line = raw.strip()
                if not line:
                    continue
                # A malformed line without a tab can be hundreds of KiB;
                # never retain its entire payload as an error-map key.
                utt = f"<line {line_number}>"
                try:
                    key, payload = line.split(b"\t")
                    utt = key.decode("utf-8")
                    dimensions = [frame.count(b",") + 1 for frame in payload.split(b"|")]
                    dim = dimensions[0]
                    if any(value != dim for value in dimensions):
                        raise ValueError(f"Inconsistent frame dimensions for {utt}")
                    shape = (len(dimensions), dim)
                    if dim not in SUPPORTED_PROSODY_DIMS:
                        raise ValueError(
                            f"Unsupported prosody dim for {utt}: got {shape}, expected 128 or 256"
                        )
                    if self.prosody_dim is None:
                        self.prosody_dim = dim
                    if dim != self.prosody_dim:
                        raise ValueError(
                            f"Prosody dim mismatch for {utt}: got {shape}, expected (*,{self.prosody_dim})"
                        )
                except ValueError as exc:
                    reason = f"{self.path}:{line_number}: {exc}"
                    if not skip_bad_entries:
                        raise ValueError(reason) from exc
                    self.errors[utt] = reason
                    self._entries.pop(utt, None)
                else:
                    self._entries[utt] = (start, len(raw), shape[0])
                    self.errors.pop(utt, None)
                if report_progress and offset >= next_report:
                    print(f"Indexed prosody: {offset / size:.0%}, {len(self)} utterances", flush=True)
                    next_report += 16 * 1024**3
            if _source_signature(file) != self._signature:
                raise ProsodySourceChangedError(f"Prosody source changed while indexing: {self.path}")
        if not self._entries:
            raise ValueError(f"No prosody entries found in {self.path}")
        if report_progress:
            print(f"Indexed prosody: {len(self)} utterances, dim={self.prosody_dim}; tensors read on demand", flush=True)

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __contains__(self, utt):
        # Mapping's default implementation would load a tensor for membership.
        return utt in self._entries

    def shape(self, utt):
        """Inspect alignment without reading and parsing the target again."""
        return self._entries[utt][2], self.prosody_dim

    def __getitem__(self, utt):
        offset, size, _ = self._entries[utt]
        with open(self.path, "rb") as file:
            if _source_signature(file) != self._signature:
                raise ProsodySourceChangedError(f"Prosody source changed after indexing: {self.path}")
            file.seek(offset)
            raw = file.read(size)
        key, pros = _parse_prosody_line(raw.decode("utf-8"))
        if key != utt or pros.shape != self.shape(utt):
            raise ValueError(f"Invalid prosody target for {utt} in {self.path}; expected {self.shape(utt)}")
        return torch.from_numpy(pros)


def load_prosody_dict(
    prosody_txt: str, expected_dim: Optional[int] = None,
    *, skip_bad_entries: bool = False,
) -> ProsodyTextIndex:
    """Keep dictionary-style reads while leaving the full target corpus on disk."""
    return ProsodyTextIndex(prosody_txt, expected_dim, skip_bad_entries=skip_bad_entries)


def infer_checkpoint_prosody_dim(state: Dict[str, torch.Tensor], spk_dim: int = 192) -> int:
    """Read D from a state dict after removing any module. prefix."""
    if "final_proj.weight" in state:
        dim = state["final_proj.weight"].shape[0] - spk_dim
    elif "pros_ln.weight" in state:
        dim = state["pros_ln.weight"].shape[0]
    else:
        raise ValueError("Cannot infer prosody dim: checkpoint has no final_proj or pros_ln weights")

    if dim not in SUPPORTED_PROSODY_DIMS:
        raise ValueError(f"Unsupported checkpoint prosody dim {dim}; expected 128 or 256")
    if "pros_ln.weight" in state and state["pros_ln.weight"].shape != (dim,):
        raise ValueError("Checkpoint prosody dim mismatch between final_proj and pros_ln")
    return dim
