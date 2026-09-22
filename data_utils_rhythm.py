"""description:
    以原始 Stage 2 資料流程載入音訊、speaker、prosody 與 rhythm 特徵。
"""

import csv
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from data_utils_stage2realfake import (
    SAMPLING_RATE,
    ProSDDStage2Dataset as _Stage2Dataset,
    load_prosody_dict,
    load_spk_mean_embeddings,
    load_utt_spk_label,
    pad,
    padr,
    process_Rawboost_feature,
)
from full_utterance import cnn_geometry, frame_count
from extract_Prosody_rhythm import CONV_KERNEL, CONV_STRIDE, extract_prosody_rhythm
from prosody_utils import ProsodySourceChangedError

RHYTHM_SOURCES = ("syllable", "vowel", "consonant")
RHYTHM_STATISTICS = ("duration", "devi_mu", "mu_diff")


def _deviation_from_mean(values: torch.Tensor) -> torch.Tensor:
    """description:
        計算 duration 相對於片段平均值的偏差。

    args:
        input:
            values: 浮點 [N] 或 [N, 層級數] Tensor，沿音節維度計算平均值。
        output:
            torch.Tensor: 與輸入同形狀、四捨五入至小數第 4 位的偏差 Tensor。
    """
    return (values - values.mean(0)).round(decimals=4)


def _normalized_pairwise_diff(values: torch.Tensor) -> torch.Tensor:
    """description:
        計算相鄰 duration 的帶正負號正規化差異，最後一項為片段 nPVI。

    args:
        input:
            values: 浮點 [N] 或 [N, 層級數] Tensor。
        output:
            torch.Tensor: 與輸入同形狀的差異 Tensor；分母為零或只有一個音節時
                對應值為 0，並四捨五入至小數第 4 位。
    """
    diff = torch.zeros_like(values)
    denominator = (values[:-1] + values[1:]) / 2
    diff[:-1] = torch.where(
        denominator == 0, 0, (values[:-1] - values[1:]) / denominator,
    )
    if len(diff) > 1:
        diff[-1] = diff[:-1].abs().mean(0)  #nPVI
    return diff.round(decimals=4)


def load_audio(path: str | Path, start: float = 0.0, end: float | None = None,
               target_sr: int = SAMPLING_RATE) -> torch.Tensor:
    """description:
        讀取指定的 [start, end) 區間，並回傳單聲道音訊。

    args:
        input:
            path: 原始音訊檔路徑。
            start: 原始錄音中的起始秒數。
            end: 原始錄音中的結束秒數；為 None 時讀取至檔案結尾。
            target_sr: 回傳音訊的取樣率。
        output:
            torch.Tensor: 指定區間的單聲道音訊，不包含 padding 或額外裁切。
    """
    # Reject invalid intervals before SoundFile interprets the frame count.
    if (not math.isfinite(start) or start < 0
            or (end is not None and (not math.isfinite(end) or end <= start))):
        raise ValueError("Expected finite seconds with 0 <= start < end")
    # Seek to the selected interval and decode only its source frames.
    with sf.SoundFile(path) as stream:
        sr = stream.samplerate
        # Round seconds back to the original sample grid without floating-point drift.
        first = round(start * sr)
        last = stream.frames if end is None else min(stream.frames, round(end * sr))
        stream.seek(first)
        samples = stream.read(last - first, dtype="float32", always_2d=True)

    # Average channels into a mono waveform without adding padding.
    wav = torch.from_numpy(samples).mean(dim=1)

    # Resample only the selected segment when the source rate differs.
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _parse_columns(row: dict[str, str], fields: Sequence[str]) -> torch.Tensor:
    """description:
        將逗號分隔的 CSV 欄位解析成有限且非負的 [N, F] Tensor。

    args:
        input:
            row: 一筆 CSV 資料列。
            fields: 要解析的欄位名稱。
        output:
            torch.Tensor: float64 型別的 [N, F] Tensor。
    """
    columns = [[float(value) for value in (row.get(field) or "").split(",")]
               for field in fields]
    values = torch.tensor(columns, dtype=torch.float64).T.contiguous()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid values in {', '.join(fields)}")
    return values


def load_duration_csv(csv_path: str | Path, utt_ids: Sequence[str],
                      rhythm_sources: Sequence[str] = RHYTHM_SOURCES) -> tuple[dict, dict]:
    """description:
        一次讀取 duration CSV，依 utterance ID 回傳有效資料與錯誤資料。

    args:
        input:
            csv_path: rhythm CSV 檔案路徑。
            utt_ids: 要載入的 utterance ID。
            rhythm_sources: 要選取的 rhythm 層級，限 syllable、vowel、consonant。
        output:
            tuple[dict, dict]: (records, errors)。records 的每筆資料包含 float64
                Tensor：duration [N, 選取層級數]、syllable [N, 2] 與 word [W, 2]；
                errors 以 utterance ID 對應錯誤原因。時間戳皆為原始音訊秒數。
    """
    sources = tuple(rhythm_sources)
    if not sources or len(set(sources)) != len(sources) or any(s not in RHYTHM_SOURCES for s in sources):
        raise ValueError("rhythm_sources must select unique syllable/vowel/consonant levels")

    wanted, seen = set(utt_ids), set()
    records, errors = {}, {}
    with open(csv_path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            utt = Path(row["flac_file_name"] or "").stem
            if utt not in wanted:
                continue
            if utt in seen:
                records.pop(utt, None)
                errors[utt] = "Duplicate rhythm rows"
                continue
            seen.add(utt)

            # Parse the selected durations and both timing levels into one record.
            try:
                record = {"duration": _parse_columns(row, [f"duration_{source}" for source in sources])}
                for level in ("syllable", "word"):
                    times = _parse_columns(row, [f"starttime_{level}", f"endtime_{level}"])
                    if (times[:, 1] < times[:, 0]).any() or (times[1:, 0] < times[:-1, 0]).any():
                        raise ValueError(f"Invalid {level} timestamps")
                    record[level] = times
                if len(record["syllable"]) != len(record["duration"]):
                    raise ValueError("Syllable timestamps must match duration rows")
            except ValueError as exc:
                errors[utt] = str(exc)
                continue
            records[utt] = record

    errors.update({utt: "Missing rhythm data" for utt in wanted - seen})
    return records, errors


def sec_to_sample(seconds: float, sr: int) -> int:
    """description:
        找到時間戳當下或之後的第一個 sample，避免浮點進位誤差。

    args:
        input:
            seconds: 原始音訊的非負秒數。
            sr: 原始音訊取樣率。
        output:
            int: 時間不早於 seconds 的第一個 sample 索引。
    """
    sample = math.floor(seconds * sr)
    return sample if sample / sr >= seconds else sample + 1


def select_duration(record: dict[str, torch.Tensor], audio_duration: float,
                    max_len: float = 4.0, sr: int = SAMPLING_RATE) -> tuple[torch.Tensor, float, float]:
    """description:
        依 word 之間的停頓選取音訊範圍，並產生對應的 duration 特徵。

    args:
        input:
            record: load_duration_csv 回傳的單筆 duration 記錄。
            audio_duration: 原始音訊總長度（秒）。
            max_len: 選取音訊的目標長度（秒）；0 表示保留整段錄音。
            sr: 原始音訊取樣率，用於將停頓邊界對齊原始 sample。
        output:
            tuple[torch.Tensor, float, float]: duration 特徵 [N, S * 3]（預設為 [N, 9]）、
                選取區間的起始秒數與結束秒數；起訖時間皆相對於原始音訊。
    """
    length = round(audio_duration * sr)
    target = round(max_len * sr)
    start, end = 0, length
    # 僅在目標長度為正且短於音檔時裁切；否則保留完整區間。
    if 0 < target < length:
        # 從詞與詞之間的空檔收集對齊 sample 的裁切點候選區間（含端點）。
        # `previous_end` 記錄目前已合併語音區間中最遠的結束位置。
        pauses, previous_end = [], 0

        # 時間戳已排序，但相鄰詞區間可能重疊或相接。
        for first, last in record["word"].tolist():
            # 將時間戳向上對齊到下一個原始音訊 sample 邊界，並限制在音檔內。
            first = min(length, sec_to_sample(first, sr))
            last = min(length, sec_to_sample(last, sr))

            # 前段語音後的空檔可作為裁切端點；排除下一個詞的起始位置。
            if first > previous_end:
                pauses.append((previous_end, first - 1))

            # 保留最遠的語音結尾，以合併相接或重疊的詞區間。
            previous_end = max(previous_end, last)

        # 若有的話，納入最後一個詞之後的靜音區間。
        if previous_end < length:
            pauses.append((previous_end, length))

        # 即使語音從第 0 個 sample 開始，也允許使用音檔開頭作為邊界。
        if not pauses or pauses[0][0] > 0:
            pauses.insert(0, (0, 0))

        # 即使語音延續至最後一個 sample，也允許使用音檔結尾作為邊界。
        if pauses[-1][1] < length:
            pauses.append((length, length))

        # 讓端點盡量接近隨機視窗；長度差異只用於打破平手。
        wanted_start = random.randint(0, length - target)
        wanted_end = wanted_start + target
        best = (math.inf, math.inf)
        for i, (left, right) in enumerate(pauses):
            # Clamp wanted_start to this pause: use left before it, itself within it, or right after it.
            first = min(max(wanted_start, left), right)
            for left, right in pauses[i + 1:]:
                last = min(max(wanted_end, left), right)
                score = (abs(first - wanted_start) + abs(last - wanted_end),
                         abs(last - first - target))
                if score < best:
                    best, start, end = score, first, last

    # Use the same selected interval for both duration rows and audio loading.
    start = start / sr
    end = min(end / sr, audio_duration)
    times = record["syllable"]
    keep = (times[:, 0] >= start) & (times[:, 1] <= end)
    durations = record["duration"][keep]
    if not len(durations):
        raise ValueError("No complete syllable within the selected interval")
    # Stack [N, S] statistics into [N, S, 3], then return [N, S * 3] (default [N, 9]).
    features = torch.stack((
        durations,
        _deviation_from_mean(durations),
        _normalized_pairwise_diff(durations)),
        dim=-1
    )
    return features.flatten(1).float(), start, end


class ProSDDStage2RhythmDataset(_Stage2Dataset):
    """description:
        依停頓裁切音訊並即時抽取 prosody，附上對應的 rhythm 序列。

    args:
        input:
            utt_ids: utterance ID 序列。
            spk_ids: 與 utterance 對應的 speaker ID 序列。
            labels: 與 utterance 對應的分類標籤序列。
            wav_dir: 音訊檔目錄。
            spkmean_txt: speaker embedding 檔案路徑。
            prosody_model: 已載入的 MPM／VAD teacher；每次 crop 後重新抽取特徵。
            duration_csv: rhythm CSV 檔案路徑。
            sr: 音訊取樣率。
            max_len: 音訊目標長度（秒）；0 表示使用全長。
            audio_ext: 音訊副檔名。
            augment_fn: 選用的音訊增強函式。
            augment_algo: 音訊增強演算法編號。
            augment_prob: 套用音訊增強的機率。
            aug_args: 傳給音訊增強函式的額外參數。
            prosody_dim: 預期的 prosody 特徵維度。
            skip_bad_entries: 是否略過無效的 Stage 2 資料。
            rhythm_sources: 要選取的 rhythm 層級。
            T_target: 預期的 CNN frame 數。
            skip_missing_duration: 是否略過缺少 duration 的資料。
            skip_bad_samples: 是否略過讀取或處理失敗的樣本。
            prosody_layer: teacher 輸出的層數。
            conv_kernel, conv_stride: 與學生模型一致的 CNN 幾何。
            report_bad_samples: 失敗時回傳 ID／原因，供訓練主程序集中記錄。
        output:
            ProSDDStage2RhythmDataset: 可供 DataLoader 讀取的資料集。單筆資料沿用
                Stage 2 tuple 前五項，後接 duration_features、valid_samples 與 utt_id。
    """

    def __init__(
        self,
        utt_ids: Sequence[str],
        spk_ids: Sequence[str],
        labels: Sequence[int],
        wav_dir: str | Path,
        spkmean_txt: str | Path,
        prosody_model: torch.nn.Module,
        duration_csv: str | Path,
        sr: int = SAMPLING_RATE,
        max_len: float = 4.0,
        audio_ext: str = ".flac",
        augment_fn: Callable[[np.ndarray, int, Any, int], np.ndarray] | None = None,
        augment_algo: int = 0,
        augment_prob: float = 0.0,
        aug_args: Any = None,
        prosody_dim: int | None = None,
        skip_bad_entries: bool = False,
        rhythm_sources: Sequence[str] = RHYTHM_SOURCES,
        T_target: int | None = None,
        skip_missing_duration: bool = True,
        skip_bad_samples: bool = True,
        prosody_layer: int = 7,
        conv_kernel: Sequence[int] = CONV_KERNEL,
        conv_stride: Sequence[int] = CONV_STRIDE,
        report_bad_samples: bool = False,
    ) -> None:
        self.rhythm_sources = tuple(rhythm_sources)
        self.T_target = T_target
        self.skip_bad_samples = skip_bad_samples
        self.report_bad_samples = report_bad_samples
        self.prosody_model = prosody_model.cpu().eval().requires_grad_(False)
        self.prosody_layer = prosody_layer
        self.conv_kernel = tuple(conv_kernel)
        self.conv_stride = tuple(conv_stride)
        teacher_dim = self.prosody_model.args.filter_size
        if prosody_dim is not None and prosody_dim != teacher_dim:
            raise ValueError(f"Prosody dim mismatch: teacher has {teacher_dim}, expected {prosody_dim}")
        self.utt2duration, self.skipped_samples = load_duration_csv(
            duration_csv, utt_ids, self.rhythm_sources,
        )
        assert len(utt_ids) == len(spk_ids) == len(labels)
        self.skipped_duration_ids = [utt for utt in utt_ids if utt not in self.utt2duration]
        for utt in self.skipped_duration_ids:
            self.skipped_samples.setdefault(utt, "Missing rhythm data")
        keep = [i for i, utt in enumerate(utt_ids) if utt in self.utt2duration]
        seconds = float(max_len)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("max_len must be finite and nonnegative")
        samples = round(seconds * sr)
        if seconds > 0 and samples < 1:
            raise ValueError("max_len must cover at least one sample")
        super().__init__(
            utt_ids=[utt_ids[i] for i in keep],
            spk_ids=[spk_ids[i] for i in keep],
            labels=[labels[i] for i in keep],
            wav_dir=wav_dir,
            sr=sr,
            spkmean_txt=spkmean_txt,
            prosody_txt=None,
            max_len=samples,
            audio_ext=audio_ext,
            augment_fn=augment_fn,
            augment_algo=augment_algo,
            augment_prob=augment_prob,
            aug_args=aug_args,
            prosody_dim=teacher_dim,
            skip_bad_entries=skip_bad_entries,
        )

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor, tuple[int, int], str
    ] | dict[str, str] | None:
        """description:
            依索引讀取停頓邊界間的音訊與 rhythm 特徵。

        args:
            input:
                idx: 資料集中的樣本索引。
            output:
                tuple | None: 成功時回傳 wav、speaker embedding、prosody embedding、
                    speaker 索引、標籤、duration 特徵、有效音訊範圍與 utterance ID；
                    略過失敗樣本時回傳 None；report_bad_samples=True 則回傳 ID／原因。
        """
        utt_id = self.utt_ids[idx]
        spk_id_str = self.spk_ids[idx]
        label = int(self.labels[idx])

        spk_idx = int(self.spk2idx[spk_id_str])

        try:
            spk_emb = self.spk2emb[spk_id_str]

            wav_path = Path(self.wav_dir) / (utt_id + self.audio_ext)
            info = sf.info(wav_path)
            duration_features, start, end = select_duration(
                self.utt2duration[utt_id], info.duration,
                max_len=self.max_len / self.sr, sr=info.samplerate,
            )
            wav = load_audio(wav_path, start, end, self.sr)
            length = wav.numel()
            if info.duration < self.max_len / self.sr:
                wav = pad(wav, self.max_len)
            left = (wav.numel() - length) // 2
            right = left + length
            stride, receptive = cnn_geometry(self.conv_kernel, self.conv_stride)
            first_frame = ((left + stride - 1) // stride) * stride
            if first_frame + receptive > right:
                raise ValueError("Selected audio has no complete CNN frame")
            pros_emb = extract_prosody_rhythm(
                self.prosody_model, wav, sr=self.sr, layer=self.prosody_layer,
                conv_kernel=self.conv_kernel, conv_stride=self.conv_stride,
            )
            wav[left:right] = self._maybe_augment(wav[left:right])
        except (MemoryError, torch.OutOfMemoryError, ProsodySourceChangedError):
            raise
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            if not self.skip_bad_samples:
                raise
            if self.report_bad_samples:
                return {"utt_id": utt_id, "reason": f"{type(error).__name__}: {error}"}
            return None
        valid_samples = (left, right)

        return wav, spk_emb, pros_emb, spk_idx, label, duration_features, valid_samples, utt_id


def collate_stage2_rhythm(batch, *, T_target=None, conv_kernel=CONV_KERNEL,
                          conv_stride=CONV_STRIDE, skip_bad_samples=True):
    """description:
        檢查單筆 CNN frame 對齊，補齊音訊、prosody 與 rhythm，組成模型 kwargs。

    args:
        input:
            batch: Dataset 回傳的樣本序列，可包含 None 或失敗紀錄 dict。
            T_target: 與模型一致的 frame 數；為 None 時依批次最長音訊決定，
                指定數字則同步截斷或補齊 prosody 與 mask。
            conv_kernel: backbone 各卷積層的 kernel 大小。
            conv_stride: backbone 各卷積層的 stride。
            skip_bad_samples: 是否略過失敗樣本。
        output:
            dict: 模型所需的批次 kwargs。rhythm 以 -100 補齊；兩種 padding mask
                的 True 都表示人工填補。
    """
    skipped = [item for item in batch if isinstance(item, dict)]
    if not skip_bad_samples and (skipped or any(item is None for item in batch)):
        raise ValueError("Cannot collate a failed sample when skip_bad_samples=False")
    batch = [item for item in batch if item is not None and not isinstance(item, dict)]
    if not batch:
        return {"skipped_samples": skipped}
    wavs, spks, pross, spk_idxs, labels, rhythms, valid_samples, utt_ids = zip(*batch)

    stride, receptive_field = cnn_geometry(conv_kernel, conv_stride)
    for audio, target, utt_id in zip(wavs, pross, utt_ids):
        expected = frame_count(audio.numel(), stride, receptive_field)
        assert target.size(0) == expected, f"{utt_id}: prosody must have {expected} CNN frames"
    wav = pad_sequence(wavs, batch_first=True)
    pros = pad_sequence(pross, batch_first=True)
    rhythm = pad_sequence(rhythms, batch_first=True, padding_value=-100)
    frames = frame_count(wav.size(1), stride, receptive_field) if T_target is None else T_target
    pros = torch.nn.functional.pad(pros, (0, 0, 0, frames - pros.size(1)))
    starts = torch.arange(frames) * stride
    bounds = torch.tensor(valid_samples)
    frame_mask = (starts[None] < bounds[:, :1]) | (starts[None] + receptive_field > bounds[:, 1:])
    pros = pros.masked_fill(frame_mask.unsqueeze(-1), 0.0)
    lengths = torch.tensor([features.size(0) for features in rhythms])
    return {
        "utt_ids": list(utt_ids), "wav": wav,
        "spk_emb": torch.stack(spks), "prosody_emb": pros,
        "spk_ids": torch.tensor(spk_idxs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "duration_features": rhythm, "frame_padding_mask": frame_mask,
        "rhythm_padding_mask": torch.arange(rhythm.size(1))[None] >= lengths[:, None],
        "skipped_samples": skipped,
    }


ProSDDStage2Dataset = ProSDDStage2RhythmDataset
collate_stage2 = collate_stage2_rhythm
