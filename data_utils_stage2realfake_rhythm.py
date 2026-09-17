"""Aligned audio, SSL targets and syllable duration inputs for Stage II.

The CSV format is rhythm-transformer's syllabification cache: flac_file_name,
starttime_syllable, endtime_syllable, and duration_<source>. Array cells contain
comma-separated numbers (optionally enclosed in brackets). Vowel/consonant
durations must be aggregated PER SYLLABLE, as in that project's current cache.

T_target=None preserves full utterances and all syllables. Targets must carry
full-utterance metadata and match the backbone's frame grid. A numeric T_target
retains the legacy four-second policy for reproducing earlier experiments.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from data_utils_stage2realfake import (
    ProSDDStage2Dataset,
    _load_audio_with_ffmpeg,
    pad,
)
from prosody_utils import ProsodySourceChangedError
from full_utterance import cnn_geometry, frame_count, read_full_prosody_metadata


class ProsodyAlignmentError(RuntimeError):
    """A cache/audio mismatch must stop training, even with skip_bad_samples."""


@dataclass(frozen=True)
class SkippedSample:
    """Carry a sample failure from a DataLoader worker to the training process."""

    utt_id: str
    reason: str


def load_utt_spk_label(list_path):
    """Read native ASVspoof 2019 (5 columns) or ASVspoof5 (10 columns)."""
    utt_ids, spk_ids, labels = [], [], []
    seen = set()
    with open(list_path, encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) not in (5, 10):
                raise ValueError(f"{list_path}:{line_number}: expected 5 or 10 protocol columns")
            speaker, utt = fields[:2]
            label = fields[4 if len(fields) == 5 else 8].lower()
            if label not in ("bonafide", "spoof"):
                raise ValueError(f"{list_path}:{line_number}: unknown label {label!r}")
            if utt in seen:
                raise ValueError(f"{list_path}:{line_number}: duplicate utterance {utt}")
            seen.add(utt)
            utt_ids.append(utt)
            spk_ids.append(speaker)
            labels.append(int(label == "bonafide"))
    if not utt_ids:
        raise ValueError(f"Empty protocol: {list_path}")
    return utt_ids, spk_ids, labels


class _MissingDuration(ValueError):
    """An alignment cache has an empty or nonfinite required value."""


def _parse_duration_cell(value, field):
    cell = (value or "").strip()
    if cell.startswith("[") and cell.endswith("]"):
        cell = cell[1:-1]
    parts = [part.strip() for part in cell.split(",")]
    if any(part.lower() in ("", "-", "none", "null") for part in parts):
        raise _MissingDuration(f"{field} contains an empty value")
    values = np.array([float(part) for part in parts], dtype=np.float64)
    if not np.isfinite(values).all():
        raise _MissingDuration(f"{field} contains a nonfinite value")
    return values


def load_duration_csv(
    csv_path, utt_ids, rhythm_sources, *, skip_missing_duration=False,
    skip_bad_samples=False, errors=None,
):
    """Join by utterance ID, optionally excluding rows with incomplete inputs.

    skip_missing_duration covers empty/nonfinite values. skip_bad_samples also
    covers absent, duplicate and corrupt rows. File/schema errors always raise.
    Exclusions apply to entire utterances, never individual syllable tokens.
    """
    required = ["starttime_syllable", "endtime_syllable"] + [
        f"duration_{source}" for source in rhythm_sources
    ]
    # The native CSV has .flac suffixes; protocols normally have bare IDs.
    wanted = {Path(utt).stem: utt for utt in utt_ids}
    if len(wanted) != len(utt_ids):
        raise ValueError("Protocol utterance IDs are not unique after removing extensions")
    records, seen = {}, set()
    errors = {} if errors is None else errors
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        missing = set(["flac_file_name", *required]) - set(header)
        if missing:
            raise ValueError(f"{csv_path}: missing duration CSV columns {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            key = Path(row["flac_file_name"] or "").stem
            if key not in wanted:
                continue
            utt = wanted[key]
            if utt in seen:
                reason = f"{csv_path}:{row_number}: duplicate duration row for {utt}"
                if not skip_bad_samples:
                    raise ValueError(reason)
                errors[utt] = reason
                records.pop(utt, None)
                continue
            seen.add(utt)
            try:
                columns = [_parse_duration_cell(row[field], field) for field in required]
                record = np.stack(columns, axis=1)
                starts, ends = record[:, 0], record[:, 1]
                if (record < 0).any():
                    raise ValueError("timestamps and durations must be nonnegative")
                if (ends <= starts).any() or (starts[1:] < ends[:-1] - 1e-4).any():
                    raise ValueError("syllables must be ordered, nonoverlapping intervals")
                if (record[:, 2:] > (ends - starts)[:, None] + 2e-4).any():
                    raise ValueError("duration exceeds its syllable interval; expected per-syllable values")
            except (ValueError, TypeError) as exc:
                incomplete = isinstance(exc, _MissingDuration)
                problem = "missing duration data" if incomplete else "invalid durations"
                reason = f"{csv_path}:{row_number}: {problem} for {utt}: {exc}"
                if skip_bad_samples or (incomplete and skip_missing_duration):
                    errors[utt] = reason
                    continue
                if incomplete:
                    reason += "; use --skip_missing_duration to exclude incomplete utterances"
                raise ValueError(reason) from exc
            records[utt] = record
    missing = set(utt_ids) - seen
    if missing and not skip_bad_samples:
        raise ValueError(f"{csv_path}: missing durations for {len(missing)} utterances: {sorted(missing)[:5]}")
    for utt in missing:
        errors[utt] = f"{csv_path}: missing durations for {utt}"
    return records


def duration_features_for_window(record, start_seconds, end_seconds):
    # Four-decimal CSV timestamps may differ from the sample boundary by 0.1 ms.
    keep = (record[:, 0] >= start_seconds - 1e-4) & (record[:, 1] <= end_seconds + 1e-4)
    durations = record[keep, 2:]
    if not len(durations):
        raise ValueError("No complete syllable remains in the 4 s audio crop")
    return duration_statistics(durations)


def duration_features_for_utterance(record, seconds):
    """Use every syllable; reject out-of-audio timestamps instead of dropping it."""
    if not len(record) or (record[:, 1] > seconds + 2e-4).any():
        raise ValueError("Syllable timestamps exceed the full audio duration")
    return duration_statistics(record[:, 2:])


def duration_statistics(durations):
    features = []
    for values in durations.T:
        deviation = values - values.mean()
        denominator = (values[:-1] + values[1:]) / 2
        pairwise = np.divide(
            values[:-1] - values[1:], denominator,
            out=np.zeros_like(denominator), where=denominator != 0,
        )
        # Match rhythm-transformer: signed adjacent differences, followed by
        # their mean absolute value (nPVI), or zero for a single syllable.
        mu_diff = np.append(pairwise, np.abs(pairwise).mean() if pairwise.size else 0.0)
        features.extend([values, np.round(deviation, 4), np.round(mu_diff, 4)])
    return torch.tensor(np.stack(features, axis=1), dtype=torch.float32)


class ProSDDStage2RhythmDataset(ProSDDStage2Dataset):
    """Extend Stage II's dataset arguments with a duration CSV and source order.

    Invalid metadata is filtered during initialization with skip_bad_samples.
    Audio/target failures discovered on access return SkippedSample so workers
    can report them and the remaining samples in the batch can still train.
    """

    def __init__(
        self, *, duration_csv, rhythm_sources=("syllable", "vowel", "consonant"),
        T_target=None, skip_missing_duration=False, skip_bad_samples=False, **kwargs,
    ):
        self.rhythm_sources = tuple(rhythm_sources)
        if not self.rhythm_sources or len(set(self.rhythm_sources)) != len(self.rhythm_sources) or any(
            source not in ("syllable", "vowel", "consonant") for source in self.rhythm_sources
        ):
            raise ValueError("Select unique syllable/vowel/consonant rhythm sources")
        if not kwargs["utt_ids"]:
            raise ValueError("Stage II dataset must not be empty")
        if len({len(kwargs[field]) for field in ("utt_ids", "spk_ids", "labels")}) != 1:
            raise ValueError("Utterance, speaker and label lists must have matching lengths")
        if T_target is not None and T_target < 1:
            raise ValueError("T_target must be positive")
        self.prosody_metadata = (
            read_full_prosody_metadata(kwargs["prosody_txt"]) if T_target is None else None
        )
        self.skip_bad_samples = skip_bad_samples
        self.skipped_samples = {}
        self.utt2duration = load_duration_csv(
            duration_csv, kwargs["utt_ids"], self.rhythm_sources,
            skip_missing_duration=skip_missing_duration,
            skip_bad_samples=skip_bad_samples, errors=self.skipped_samples,
        )
        self.skipped_duration_ids = [utt for utt in kwargs["utt_ids"] if utt not in self.utt2duration]
        if self.skipped_duration_ids:
            keep = [i for i, utt in enumerate(kwargs["utt_ids"]) if utt in self.utt2duration]
            print(
                f"{duration_csv}: skipped {len(self.skipped_duration_ids)}/{len(kwargs['utt_ids'])} "
                f"utterances with unavailable duration data; retained {len(keep)}. "
                f"Examples: {self.skipped_duration_ids[:5]}", flush=True,
            )
            for field in ("utt_ids", "spk_ids", "labels"):
                kwargs[field] = [kwargs[field][i] for i in keep]
        if not kwargs["utt_ids"]:
            raise ValueError(f"{duration_csv}: No utterances with complete durations remain after filtering")
        super().__init__(**kwargs, skip_bad_entries=skip_bad_samples)
        self.T_target = T_target
        valid_speakers = {
            speaker for speaker, embedding in self.spk2emb.items()
            if embedding.shape == (192,) and torch.isfinite(embedding).all()
        }
        keep = []
        for i, (utt, speaker) in enumerate(zip(self.utt_ids, self.spk_ids)):
            reason = None
            if speaker not in valid_speakers:
                reason = f"{utt}: missing or invalid 192-D speaker embedding for {speaker}"
            elif utt not in self.utt2pros:
                reason = self.utt2pros.errors.get(
                    utt, f"{utt}: missing cached prosody entry in {self.utt2pros.path}",
                )
            elif T_target is not None and self.utt2pros.shape(utt) != (T_target, self.prosody_dim):
                reason = (f"{utt}: expected cached prosody shape ({T_target}, {self.prosody_dim}), "
                          f"got {self.utt2pros.shape(utt)}")
            elif self.prosody_metadata is not None:
                info = self.prosody_metadata["utterances"].get(utt)
                if info is None:
                    reason = f"{utt}: missing full-utterance prosody metadata"
                else:
                    stride, receptive = cnn_geometry(
                        self.prosody_metadata["conv_kernel"], self.prosody_metadata["conv_stride"],
                    )
                    expected = frame_count(info["num_samples"], stride, receptive)
                    if expected < 1 or self.utt2pros.shape(utt) != (expected, self.prosody_dim):
                        raise ProsodyAlignmentError(f"{utt}: full-utterance prosody frame count does not match audio metadata")
            if reason is not None:
                if not skip_bad_samples:
                    raise ValueError(reason)
                self.skipped_samples[utt] = reason
            else:
                keep.append(i)
        self.utt_ids = [self.utt_ids[i] for i in keep]
        self.spk_ids = [self.spk_ids[i] for i in keep]
        self.labels = [self.labels[i] for i in keep]
        if not self.utt_ids:
            raise ValueError("No usable utterances remain after filtering invalid targets")
        self.audio_lengths = (
            [self.prosody_metadata["utterances"][utt]["num_samples"] for utt in self.utt_ids]
            if self.prosody_metadata is not None else [self.max_len] * len(self.utt_ids)
        )

    def __getitem__(self, idx):
        utt, speaker = self.utt_ids[idx], self.spk_ids[idx]
        name = utt if utt.endswith(self.audio_ext) else utt + self.audio_ext
        path = str(Path(self.wav_dir) / name)
        try:
            try:
                wav, sr = torchaudio.load(path)
            except (MemoryError, torch.OutOfMemoryError):
                raise
            except (OSError, RuntimeError):
                wav, sr = _load_audio_with_ffmpeg(path, self.sr)
            wav = wav.mean(dim=0)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            samples = wav.numel()
            if not samples:
                raise ValueError("Empty waveform")
            if self.T_target is None:
                if samples != self.audio_lengths[idx]:
                    raise ProsodyAlignmentError(f"{utt}: audio length differs from full-utterance prosody cache; regenerate targets")
                valid_start, valid_end = 0, samples
                duration = duration_features_for_utterance(self.utt2duration[utt], samples / self.sr)
                wav = self._maybe_augment(wav)
            else:
                crop_start = max(0, (samples - self.max_len) // 2)
                crop_end = min(samples, crop_start + self.max_len)
                valid_start = max(0, (self.max_len - samples) // 2)
                valid_end = valid_start + min(samples, self.max_len)
                duration = duration_features_for_window(
                    self.utt2duration[utt], crop_start / self.sr, crop_end / self.sr,
                )
                wav = self._maybe_augment(pad(wav, self.max_len))
            # RawBoost preserves timing but can add noise in zero padding.
            wav[:valid_start] = 0
            wav[valid_end:] = 0
            prosody = self.utt2pros[utt]
            if not torch.isfinite(wav).all():
                raise ValueError("Waveform contains nonfinite values")
            if not torch.isfinite(prosody).all():
                raise ValueError("Prosody target contains nonfinite values")
            if not torch.isfinite(duration).all():
                raise ValueError("Duration features contain nonfinite values")
        except (MemoryError, torch.OutOfMemoryError, ProsodySourceChangedError, ProsodyAlignmentError):
            raise
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            reason = f"{utt}: could not prepare aligned sample from {path}: {exc}"
            if self.skip_bad_samples:
                return SkippedSample(utt, reason)
            raise RuntimeError(reason) from exc
        return {
            "utt_id": utt, "wav": wav, "spk_emb": self.spk2emb[speaker],
            "prosody_emb": prosody, "spk_ids": self.spk2idx[speaker],
            "labels": int(self.labels[idx]), "duration_features": duration,
            "valid_samples": (valid_start, valid_end),
        }


def collate_stage2_rhythm(batch, *, T_target, conv_kernel, conv_stride, skip_bad_samples=False):
    """Make both padding masks; conv geometry comes from the actual backbone.

    An acoustic frame is valid only when its CNN receptive field lies entirely
    inside real audio. Genuine silence/pauses inside the audio remain valid.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    stride, receptive_field = cnn_geometry(conv_kernel, conv_stride)
    if T_target is not None and T_target < 1:
        raise ValueError("T_target must be positive")
    skipped = [{"utt_id": item.utt_id, "reason": item.reason}
               for item in batch if isinstance(item, SkippedSample)]
    if skipped and not skip_bad_samples:
        raise ValueError(skipped[0]["reason"])
    batch = [item for item in batch if not isinstance(item, SkippedSample)]
    if not batch:
        return {"skipped_samples": skipped}
    full_utterance = T_target is None
    wav = pad_sequence([item["wav"] for item in batch], batch_first=True)
    actual_frames = max(0, (wav.size(1) - receptive_field) // stride + 1)
    if full_utterance:
        T_target = actual_frames
    if actual_frames > T_target:
        raise ValueError("T_target truncates acoustic frames; duration/prosody would cover a different window")
    frame_start = torch.arange(T_target) * stride
    regions = torch.tensor([item["valid_samples"] for item in batch])
    valid = (frame_start[None] >= regions[:, :1]) & (
        frame_start[None] + receptive_field <= regions[:, 1:]
    )
    valid &= torch.arange(T_target)[None] < actual_frames
    if not valid.any(dim=1).all():
        invalid = [item["utt_id"] for item, ok in zip(batch, valid.any(dim=1)) if not ok]
        if not skip_bad_samples:
            raise ValueError(f"Audio too short for a valid CNN frame: {invalid}")
        skipped.extend({"utt_id": utt, "reason": "Audio too short for a valid CNN frame"} for utt in invalid)
        keep = valid.any(dim=1)
        batch = [item for item, ok in zip(batch, keep) if ok]
        wav, valid = wav[keep], valid[keep]
        if not batch:
            return {"skipped_samples": skipped}
    durations = pad_sequence([item["duration_features"] for item in batch], batch_first=True, padding_value=-100)
    lengths = torch.tensor([item["duration_features"].size(0) for item in batch])
    if full_utterance:
        for item, count in zip(batch, valid.sum(1).tolist()):
            if item["prosody_emb"].size(0) != count:
                raise ProsodyAlignmentError(f"{item['utt_id']}: full-utterance target length does not match backbone frames")
    prosody = pad_sequence([item["prosody_emb"] for item in batch], batch_first=True)
    # The longest sample may have been skipped for invalid audio.
    if full_utterance:
        valid = valid[:, :prosody.size(1)]
        wav = wav[:, :max(item["wav"].numel() for item in batch)]
    return {
        "skipped_samples": skipped,
        "utt_ids": [item["utt_id"] for item in batch], "wav": wav,
        "spk_emb": torch.stack([item["spk_emb"] for item in batch]),
        "prosody_emb": prosody,
        "spk_ids": torch.tensor([item["spk_ids"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        "duration_features": durations, "frame_padding_mask": ~valid,
        "rhythm_padding_mask": torch.arange(durations.size(1))[None] >= lengths[:, None],
    }
