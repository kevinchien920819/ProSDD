"""透過音訊、CSV 與 Dataset 公開介面驗證 rhythm 資料導入。"""

import csv
from functools import partial
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

import data_utils_rhythm as data
from data_utils_stage2realfake import ProSDDStage2Dataset


class CropTeacher(torch.nn.Module):
    """替代外部 teacher；可用波形中的時間訊號驗證裁切內容。"""

    def __init__(self, use_waveform=False, dim=128):
        super().__init__()
        self.args = SimpleNamespace(filter_size=dim)
        self.vad_measure = SimpleNamespace(hop_length=256, sampling_rate=22050)
        self.use_waveform = use_waveform

    def process_audio(self, path, layer):
        wav, sr = sf.read(path)
        times = np.arange(int(len(wav) / sr * 22050 / 256) + 1) * (256 / 22050)
        values = (np.interp(times, np.arange(len(wav)) / sr, wav)
                  if self.use_waveform else np.full(len(times), 0.5))
        return np.repeat(values[:, None], self.args.filter_size, axis=1)


class AudioLoadingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.wav = self.root / "audio.wav"
        self.samples = np.linspace(-0.5, 0.5, 96000, dtype=np.float32)
        sf.write(self.wav, self.samples, 16000, subtype="FLOAT")

    def test_requested_interval_reads_exact_samples_without_moving_boundaries(self):
        wav = data.load_audio(self.wav, start=0.6, end=4.4000625)

        torch.testing.assert_close(wav, torch.from_numpy(self.samples[9600:70401]))

    def test_sample_boundaries_survive_conversion_to_seconds_and_back(self):
        wav = data.load_audio(self.wav, start=2007 / 16000, end=6001 / 16000)

        torch.testing.assert_close(wav, torch.from_numpy(self.samples[2007:6001]))

    def test_default_interval_preserves_every_sample(self):
        wav = data.load_audio(self.wav)

        torch.testing.assert_close(wav, torch.from_numpy(self.samples))

    def test_omitted_end_reads_from_start_to_end_of_recording(self):
        torch.testing.assert_close(data.load_audio(self.wav, start=2), torch.from_numpy(self.samples[32000:]))

    def test_stereo_full_audio_is_mixed_to_mono(self):
        sf.write(self.wav, np.column_stack([self.samples, -self.samples]), 16000, subtype="FLOAT")

        torch.testing.assert_close(data.load_audio(self.wav), torch.zeros(96000))

    def test_selected_interval_is_resampled_to_the_requested_rate(self):
        sf.write(self.wav, np.zeros(96000, dtype=np.float32), 16000)

        torch.testing.assert_close(
            data.load_audio(self.wav, start=1.25, end=3.25, target_sr=8000), torch.zeros(16000),
        )

    def test_invalid_intervals_raise_instead_of_reading_the_remaining_audio(self):
        for start, end in ((3, 2), (1, 1), (-1, 1), (float("nan"), 2), (0, float("inf"))):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "start.*end"):
                data.load_audio(self.wav, start=start, end=end)


class DurationSelectionTests(unittest.TestCase):
    def select(self, words, start=8000, sr=16000):
        record = {
            "duration": torch.tensor([[0.2, 0.1, 0.1], [0.4, 0.15, 0.25]], dtype=torch.float64),
            "syllable": torch.tensor([[1.0, 1.2], [3.0, 3.4]], dtype=torch.float64),
            "word": torch.tensor(words, dtype=torch.float64),
        }
        with patch("random.randint", return_value=start):
            return data.select_duration(record, audio_duration=6.0, max_len=4.0, sr=sr)

    def test_random_crop_moves_both_ends_into_even_submillisecond_word_gaps(self):
        words = [(0.2, 0.6), (0.6001, 1.8), (1.8, 2.1), (2.15, 4.4), (4.4001, 5.6)]

        self.assertEqual(self.select(words)[1:], (0.6, 4.4000625))

    def test_touching_words_do_not_create_a_pause(self):
        result = self.select([(0.2, 0.5), (0.5, 4.5), (4.5, 5.6)])

        self.assertEqual(result[1:], (0.1999375, 5.6))

    def test_an_uninterrupted_recording_is_preserved_instead_of_cutting_words(self):
        self.assertEqual(self.select([(0.0, 6.0)])[1:], (0.0, 6.0))

    def test_overlapping_words_cannot_expose_a_false_pause(self):
        result = self.select([(0.2, 5.6), (0.4, 0.5), (1.0, 1.5)])

        self.assertEqual(result[1:], (0.1999375, 5.6))

    def test_a_pause_between_fractional_sample_timestamps_keeps_its_single_sample(self):
        result = self.select([(0.2, 0.60001), (0.6001, 5.6)])

        self.assertEqual(result[1:], (0.6000625, 5.6))

    def test_floating_point_roundoff_cannot_hide_a_one_sample_pause(self):
        result = self.select([(0.0, 2007 / 16000), (2008 / 16000, 5.6)], start=1600)

        self.assertEqual(result[1:], (2007 / 16000, 5.6))

    def test_pause_boundaries_use_the_supplied_source_sampling_rate(self):
        result = self.select([(0.2, 0.6), (0.6001, 4.4), (4.4001, 5.6)], start=4000, sr=8000)

        self.assertEqual(result[1:], (0.6, 4.4))


class RhythmCsvTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.csv = self.root / "rhythm.csv"
        # 手算範例：duration 載入後，統計值由選取片段重新計算。
        self.rows = [{
            "flac_file_name": "first.flac", "label": "spoof",
            "starttime_word": "0.2, 0.6", "endtime_word": "0.4, 1.0",
            "starttime_syllable": "0.2, 0.6", "endtime_syllable": "0.4, 1.0",
            "starttime_phoneme": "0.2, 0.3, 0.6, 0.75", "endtime_phoneme": "0.3, 0.4, 0.75, 1.0",
            "duration_syllable": "0.2, 0.4", "devi_mu_syllable": "-0.1, 0.1",
            "mu_diff_syllable": "-0.6667, 0.6667",
            "duration_vowel": "0.1, 0.15", "devi_mu_vowel": "-0.025, 0.025",
            "mu_diff_vowel": "-0.4, 0.4",
            "duration_consonant": "0.1, 0.25", "devi_mu_consonant": "-0.075, 0.075",
            "mu_diff_consonant": "-0.8571, 0.8571", "nPVI_syllable": "unused",
        }]
        self.expected = torch.tensor([
            [0.2, -0.1, -0.6667, 0.1, -0.025, -0.4, 0.1, -0.075, -0.8571],
            [0.4, 0.1, 0.6667, 0.15, 0.025, 0.4, 0.25, 0.075, 0.8571],
        ])
        self.write_csv()

    def write_csv(self):
        with self.csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_csv_returns_raw_duration_and_timestamps_in_one_record(self):
        records, errors = data.load_duration_csv(self.csv, ["first"])

        torch.testing.assert_close((records, errors), ({"first": {
            "duration": torch.tensor([[0.2, 0.1, 0.1], [0.4, 0.15, 0.25]], dtype=torch.float64),
            "syllable": torch.tensor([[0.2, 0.4], [0.6, 1.0]], dtype=torch.float64),
            "word": torch.tensor([[0.2, 0.4], [0.6, 1.0]], dtype=torch.float64),
        }}, {}))

    def test_word_timestamps_load_without_phoneme_columns(self):
        self.rows[0] = {key: value for key, value in self.rows[0].items() if "phoneme" not in key}
        self.rows[0].update(starttime_word="0.2, 0.6", endtime_word="0.4, 1.0")
        self.write_csv()

        records, _ = data.load_duration_csv(self.csv, ["first"])

        torch.testing.assert_close(records["first"]["word"],
                                   torch.tensor([[0.2, 0.4], [0.6, 1.0]], dtype=torch.float64))

    def test_duration_selection_returns_features_and_explicit_audio_bounds(self):
        records, _ = data.load_duration_csv(self.csv, ["first"])
        with patch("random.randint", return_value=0):
            result = data.select_duration(records["first"], audio_duration=6.0, max_len=0.5)

        torch.testing.assert_close(result, (
            torch.tensor([[0.2, 0, 0, 0.1, 0, 0, 0.1, 0, 0]]), 0.0, 0.5,
        ))

    def test_incomplete_or_invalid_selected_fields_skip_the_whole_utterance(self):
        good = {**self.rows[0], "flac_file_name": "good.flac"}
        for field in (name for name in good if name.startswith("duration_")):
            for value in ("", "0.1", "0.1, nan", "0.1, inf", "0.1, 0.2broken", "0.1, -100"):
                with self.subTest(field=field, value=value):
                    self.rows = [{**good, "flac_file_name": "bad.flac", field: value}, good]
                    self.write_csv()

                    records, _ = data.load_duration_csv(self.csv, ["bad", "good"])

                    self.assertEqual(list(records), ["good"])

    def test_selected_levels_use_only_their_fields_in_requested_order(self):
        del self.rows[0]["duration_syllable"]
        self.write_csv()

        records, _ = data.load_duration_csv(self.csv, ["first"], ("consonant", "vowel"))

        torch.testing.assert_close(records["first"]["duration"],
                                   torch.tensor([[0.1, 0.1], [0.25, 0.15]], dtype=torch.float64))

    def test_missing_csv_columns_exclude_the_utterance(self):
        del self.rows[0]["duration_vowel"]
        self.write_csv()

        self.assertEqual(data.load_duration_csv(self.csv, ["first"])[0], {})

    def test_invalid_or_repeated_levels_are_rejected(self):
        for sources in ((), ("word",), ("vowel", "vowel")):
            with self.subTest(sources=sources), self.assertRaisesRegex(ValueError, "rhythm_sources"):
                data.load_duration_csv(self.csv, ["first"], sources)

    def test_duplicate_utterance_rows_are_excluded_instead_of_overwritten(self):
        self.rows.append(dict(self.rows[0]))
        self.write_csv()

        self.assertEqual(data.load_duration_csv(self.csv, ["first"]), ({}, {"first": "Duplicate rhythm rows"}))


class RhythmDatasetTests(unittest.TestCase):
    write_csv = RhythmCsvTests.write_csv

    def setUp(self):
        RhythmCsvTests.setUp(self)
        rng = patch("random.randint", return_value=0)
        rng.start()
        self.addCleanup(rng.stop)

    def dataset(self, **kwargs):
        self.speaker = self.root / "speaker.txt"
        self.speaker.write_text("s1 " + " ".join(["0.25"] * 192) + "\n")
        self.prosody = self.root / "prosody.txt"
        self.prosody.write_text("first\t" + "|".join([",".join(["0.5"] * 128)] * 299) + "\n")
        sf.write(self.root / "first.wav", np.ones(96000, dtype=np.float32) * 0.1, 16000)
        options = dict(
            utt_ids=["first"], spk_ids=["s1"], labels=[1], wav_dir=self.root,
            spkmean_txt=self.speaker, prosody_model=CropTeacher(), duration_csv=self.csv,
            audio_ext=".wav", max_len=0,
        )
        options.update(kwargs)
        return data.ProSDDStage2RhythmDataset(**options)

    def test_sample_preserves_original_stage2_return_order(self):
        dataset = self.dataset()
        original = ProSDDStage2Dataset(
            utt_ids=["first"], spk_ids=["s1"], labels=[1], wav_dir=self.root,
            spkmean_txt=self.speaker, prosody_txt=self.prosody,
            audio_ext=".wav", max_len=0,
        )

        torch.testing.assert_close(dataset[0][:5], original[0])

    def cropped_dataset(self, **kwargs):
        self.rows[0].update({
            "starttime_syllable": "0.2, 1.0, 3.0, 4.8",
            "endtime_syllable": "0.3, 1.2, 3.4, 5.6",
            "starttime_word": "0.2, 0.6001, 1.8, 2.15, 4.4001",
            "endtime_word": "0.6, 1.8, 2.1, 4.4, 5.6",
            "duration_syllable": "0.1, 0.2, 0.4, 0.8",
            "duration_vowel": "0.08, 0.1, 0.15, 0.3",
            "duration_consonant": "0.02, 0.1, 0.25, 0.5",
        })
        for source in data.RHYTHM_SOURCES:
            for stat in ("devi_mu", "mu_diff"):
                self.rows[0][f"{stat}_{source}"] = "99, 99, 99, 99"
        self.write_csv()
        return self.dataset(**{"max_len": 4, **kwargs})

    def test_cropped_duration_rows_match_the_pause_aligned_audio_segment(self):
        dataset = self.cropped_dataset()
        with patch("random.randint", return_value=8000):
            item = dataset[0]

        torch.testing.assert_close(item[5][:, ::3], self.expected[:, ::3])

    def test_cropped_statistics_are_recomputed_using_only_retained_durations(self):
        dataset = self.cropped_dataset()
        with patch("random.randint", return_value=8000):
            features = dataset[0][5]

        torch.testing.assert_close(features, self.expected)

    def test_each_access_selects_new_matching_audio_and_duration_from_cached_csv(self):
        dataset = self.cropped_dataset()
        self.csv.unlink()
        samples = np.linspace(-0.5, 0.5, 96000, dtype=np.float32)
        sf.write(self.root / "first.wav", samples, 16000, subtype="FLOAT")
        with patch("random.randint", side_effect=[8000, 0]):
            first, second = dataset[0], dataset[0]

        torch.testing.assert_close(
            (first[0], first[5], second[0], second[5][:, 0]),
            (torch.from_numpy(samples[9600:70401]), self.expected,
             torch.from_numpy(samples[:70400]), torch.tensor([0.1, 0.2, 0.4])),
        )

    def test_each_crop_extracts_prosody_from_its_own_selected_audio(self):
        dataset = self.cropped_dataset(prosody_model=CropTeacher(use_waveform=True))
        samples = np.arange(96000, dtype=np.float32) / 16000
        sf.write(self.root / "first.wav", samples, 16000, subtype="FLOAT")
        with patch("random.randint", side_effect=[8000, 0]):
            first, second = dataset[0], dataset[0]

        torch.testing.assert_close(
            torch.stack([first[2][0, 0], second[2][0, 0]]),
            torch.tensor([0.61246875, 0.01246875]),
        )

    def test_teacher_targets_describe_clean_audio_before_augmentation(self):
        dataset = self.dataset(
            prosody_model=CropTeacher(use_waveform=True), max_len=4,
            augment_fn=lambda wav, sr, args, algo: wav + 0.5,
            augment_algo=2, augment_prob=1.0,
        )
        sf.write(self.root / "first.wav", np.full(96000, 0.125), 16000, subtype="FLOAT")
        item = dataset[0]

        torch.testing.assert_close(
            torch.stack([item[0][0], item[2][0, 0]]), torch.tensor([0.625, 0.125]),
        )

    def test_crop_mode_excludes_missing_or_invalid_alignment_before_sampling(self):
        good = dict(self.rows[0])
        for field, value in (
            ("starttime_syllable", ""), ("endtime_word", ""),
            ("starttime_syllable", "0.2"), ("endtime_word", "0.4"),
            ("starttime_syllable", "nan, 0.6"), ("endtime_word", "0.4, inf"),
            ("starttime_syllable", "-0.2, 0.6"), ("endtime_syllable", "0.1, 1.0"),
            ("starttime_syllable", "0.7, 0.2"), ("starttime_word", "0.7, 0.2"),
            ("endtime_word", "0.1, 1.0"),
        ):
            with self.subTest(field=field, value=value):
                self.rows = [{**good, field: value}]
                self.write_csv()

                self.assertEqual(len(self.dataset(max_len=4)), 0)

    def test_crop_without_a_complete_syllable_is_skipped(self):
        self.rows[0].update(starttime_syllable="0.1, 5.8", endtime_syllable="5.8, 6.0")
        self.write_csv()
        dataset = self.dataset(max_len=0.5)

        self.assertIsNone(dataset[0])

    def test_padding_cannot_make_a_sub_frame_recording_valid_for_ssl(self):
        self.rows[0].update(
            starttime_word="0.002, 0.011", endtime_word="0.006, 0.019",
            starttime_syllable="0.002, 0.011", endtime_syllable="0.006, 0.019",
            duration_syllable="0.004, 0.008", duration_vowel="0.002, 0.004",
            duration_consonant="0.002, 0.004",
        )
        self.write_csv()
        dataset = self.dataset(max_len=4)
        sf.write(self.root / "first.wav", np.ones(320, dtype=np.float32) * 0.1, 16000)

        self.assertIsNone(dataset[0])

    def test_crop_statistics_do_not_require_utterance_statistic_columns(self):
        self.rows[0] = {key: value for key, value in self.rows[0].items()
                        if not key.startswith(("devi_mu_", "mu_diff_"))}
        self.write_csv()

        features = self.dataset(max_len=0.5)[0][5]

        torch.testing.assert_close(features, torch.tensor([[0.2, 0, 0, 0.1, 0, 0, 0.1, 0, 0]]))

    def test_crop_recomputes_zero_denominators_and_final_npvi_in_selected_source_order(self):
        self.cropped_dataset()
        self.rows[0]["duration_consonant"] = "0, 0, 0.25, 0.5"
        self.write_csv()

        features = self.dataset(max_len=4, rhythm_sources=("consonant", "syllable"))[0][5]

        torch.testing.assert_close(features, torch.tensor([
            [0, -0.0833, 0, 0.1, -0.1333, -0.6667],
            [0, -0.0833, -2, 0.2, -0.0333, -0.6667],
            [0.25, 0.1667, 1, 0.4, 0.1667, 0.6667],
        ]))

    def test_variable_pause_aligned_crops_keep_audio_and_rhythm_padding_masks(self):
        dataset = self.cropped_dataset()
        with patch("random.randint", side_effect=[8000, 0]):
            batch = data.collate_stage2_rhythm([dataset[0], dataset[0]])

        self.assertEqual(
            ((~batch["frame_padding_mask"]).sum(1).tolist(), batch["rhythm_padding_mask"].tolist()),
            ([189, 219], [[False, False, True], [False, False, False]]),
        )

    def test_augmentation_uses_cropped_audio_like_original_stage2(self):
        augment_fn = lambda wav, sr, args, algo: wav + np.linspace(0, 1, wav.size, dtype=np.float32)
        dataset = self.dataset(
            max_len=2.5, augment_fn=augment_fn, augment_algo=2, augment_prob=1.0,
        )
        original = ProSDDStage2Dataset(
            utt_ids=["first"], spk_ids=["s1"], labels=[1], wav_dir=self.root,
            spkmean_txt=self.speaker, prosody_txt=self.prosody,
            audio_ext=".wav", max_len=40000,
            augment_fn=augment_fn, augment_algo=2, augment_prob=1.0,
        )

        torch.testing.assert_close(dataset[0][0], original[0][0])

    def test_positional_dataset_arguments_default_to_four_seconds(self):
        self.dataset()
        dataset = data.ProSDDStage2RhythmDataset(
            ["first"],
            ["s1"],
            [1],
            self.root,
            self.speaker,
            CropTeacher(),
            self.csv,
            audio_ext=".wav",
        )

        self.assertEqual(dataset[0][0].numel(), 64000)

    def test_dataset_max_len_controls_audio_length_in_seconds(self):
        for seconds, count in ((0, 96000), (2.5, 40000), (4, 64000), (8, 128000)):
            with self.subTest(seconds=seconds):
                self.assertEqual(self.dataset(max_len=seconds)[0][0].numel(), count)

    def test_seconds_are_converted_using_the_requested_sampling_rate(self):
        self.assertEqual(self.dataset(sr=8000, max_len=2.5)[0][0].shape, (20000,))

    def test_resampling_preserves_a_pause_with_only_one_original_audio_sample(self):
        self.cropped_dataset()
        self.rows[0]["starttime_word"] = "0.2, 0.60004, 1.8, 2.15, 4.4001"
        self.rows[0]["endtime_word"] = "0.60002, 1.8, 2.1, 4.4, 5.6"
        self.write_csv()
        dataset = self.dataset(max_len=4)
        sf.write(self.root / "first.wav", np.zeros(288000, dtype=np.float32), 48000)
        with patch("random.randint", side_effect=lambda low, high: high // 4):
            item = dataset[0]

        torch.testing.assert_close((item[0], item[5]), (torch.zeros(60801), self.expected))

    def test_resampled_silence_retains_exact_valid_bounds_after_padding(self):
        dataset = self.dataset(sr=8000, max_len=2.5)
        sf.write(self.root / "first.wav", np.zeros(16001, dtype=np.float32), 16000)

        self.assertEqual(dataset[0][6], (5999, 14000))

    def test_invalid_seconds_do_not_silently_produce_empty_audio(self):
        for seconds in (-1, float("nan"), float("inf"), 0.000001):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "max_len"):
                self.dataset(max_len=seconds)

    def test_dataset_level_selection_preserves_complete_vowel_sequence(self):
        item = self.dataset(max_len=0, rhythm_sources=("vowel",))[0]

        torch.testing.assert_close(item[5], self.expected[:, 3:6])

    def test_augmentation_preserves_csv_values_and_artificial_padding(self):
        item = self.dataset(
            max_len=8, augment_fn=lambda wav, sr, args, algo: wav + 0.5,
            augment_algo=2, augment_prob=1.0,
        )[0]

        torch.testing.assert_close(item[5], self.expected)
        self.assertEqual(item[6], (16000, 112000))
        torch.testing.assert_close(item[0][:16000], torch.zeros(16000))
        self.assertGreater(item[0][16000].item(), 0.59)

    def test_protocol_labels_take_precedence_over_csv_labels(self):
        self.assertEqual(self.dataset()[0][4], 1)

    def test_dataset_filters_missing_rhythm_before_joining_speakers_and_labels(self):
        self.rows.insert(0, {**self.rows[0], "flac_file_name": "bad.flac", "duration_vowel": ""})
        self.write_csv()
        dataset = self.dataset(
            utt_ids=["bad", "absent", "first"], spk_ids=["s2", "s3", "s1"], labels=[0, 0, 1],
        )

        self.assertEqual((dataset.utt_ids, dataset.spk_ids, dataset.labels), (["first"], ["s1"], [1]))

    def batch_samples(self):
        dataset = self.dataset()
        first = dataset[0]
        wav, spk_emb, pros_emb, spk_idx, label, duration_features, _, _ = first
        second = (
            wav[:16000], spk_emb, pros_emb[:49], spk_idx, label,
            duration_features[:1], (0, 16000), "short",
        )
        return [first, second]

    def test_full_utterance_batch_keeps_audio_and_right_pads_shorter_members(self):
        batch = data.collate_stage2_rhythm(
            self.batch_samples(), T_target=None, conv_kernel=(400,), conv_stride=(320,),
        )

        self.assertEqual(batch["wav"].shape, (2, 96000))
        torch.testing.assert_close(batch["wav"][1, 16000:], torch.zeros(80000))

    def test_padding_masks_mark_artificial_padding_including_silent_audio(self):
        samples = self.batch_samples()
        samples[1] = (torch.zeros(16000), *samples[1][1:])
        batch = data.collate_stage2_rhythm(samples)

        self.assertEqual((~batch["frame_padding_mask"]).sum(1).tolist(), [299, 49])
        self.assertEqual(batch["rhythm_padding_mask"].tolist(), [[False, False], [False, True]])
        torch.testing.assert_close(batch["duration_features"][1, 1], torch.full((9,), -100.0))

    def test_empty_or_skipped_batches_can_be_ignored_by_training(self):
        for batch in ([], [None]):
            with self.subTest(batch=batch):
                self.assertEqual(data.collate_stage2_rhythm(batch), {"skipped_samples": []})

    def test_strict_collation_reports_failed_samples(self):
        with self.assertRaisesRegex(ValueError, "failed sample"):
            data.collate_stage2_rhythm([self.dataset()[0], None], skip_bad_samples=False)

    def test_missing_audio_is_skipped_without_discarding_the_valid_batch_member(self):
        dataset = self.dataset()
        valid = dataset[0]
        (self.root / "first.wav").unlink()

        batch = data.collate_stage2_rhythm([dataset[0], valid])

        self.assertEqual(batch["utt_ids"], ["first"])

    def test_missing_audio_returns_none(self):
        dataset = self.dataset()
        (self.root / "first.wav").unlink()

        self.assertIsNone(dataset[0])

    def test_batch_rejects_prosody_with_the_wrong_audio_frame_count(self):
        samples = self.batch_samples()
        wav, spk_emb, pros_emb, *rest = samples[0]
        samples[0] = (wav, spk_emb, pros_emb[:200], *rest)

        with self.assertRaisesRegex(AssertionError, "prosody.*frame"):
            data.collate_stage2_rhythm(samples)

    def test_explicit_frame_target_matches_model_truncation_and_padding(self):
        for frames in (100, 305):
            with self.subTest(frames=frames):
                batch = data.collate_stage2_rhythm(self.batch_samples(), T_target=frames)
                expected = torch.full((2, frames, 128), 0.5)
                expected[0, 299:] = 0
                expected[1, 49:] = 0

                torch.testing.assert_close(batch["prosody_emb"], expected)

    def test_center_padding_is_zeroed_in_prosody_targets(self):
        batch = data.collate_stage2_rhythm([self.dataset(max_len=8)[0]])
        expected = torch.full((1, 399, 128), 0.5)
        expected[:, :50] = 0
        expected[:, 349:] = 0

        torch.testing.assert_close(batch["prosody_emb"], expected)

    def model(self, *, frames=None, dim=128):
        """建立具相同 frame 幾何的小型真實模型，供公開訓練介面驗收。"""
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        from model_stage2realfake_rhythm import ProSDDStage2Rhythm

        backbone = Wav2Vec2Model(Wav2Vec2Config(
            hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
            conv_dim=(8,), conv_kernel=(400,), conv_stride=(320,),
            num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
            feat_extract_norm="layer", hidden_dropout=0.0, attention_dropout=0.0,
            feat_proj_dropout=0.0, layerdrop=0.0,
        ))
        with patch("transformers.Wav2Vec2Model.from_pretrained", return_value=backbone):
            model = ProSDDStage2Rhythm(
                model_name="test", prosody_dim=dim, T_target=frames, nhead=2,
                n_rhythm_encoder_layers=1, n_cls_encoder_layers=1, dropout=0.0,
                num_time_neg=2, num_spk_neg=1,
            ).eval()
        return model

    def test_cropped_dataloader_batches_support_ssl_backpropagation(self):
        dataset = self.cropped_dataset(prosody_model=CropTeacher(dim=256))
        for frames in (None, 200):
            with self.subTest(frames=frames):
                model = self.model(frames=frames, dim=dataset.prosody_dim)
                loader = DataLoader(
                    torch.utils.data.Subset(dataset, [0, 0]), batch_size=2,
                    collate_fn=partial(data.collate_stage2_rhythm, T_target=frames),
                )
                with patch("random.randint", side_effect=[8000, 0]):
                    batch = next(iter(loader))
                inputs = {key: value for key, value in batch.items()
                          if key not in ("utt_ids", "labels", "skipped_samples")}

                loss = model(**inputs)["ssl_loss"]
                loss.backward()

                self.assertTrue(torch.isfinite(torch.cat([
                    loss.reshape(1), model.final_proj.weight.grad.flatten(),
                ])).all())

    def test_full_batches_backpropagate_through_audio_beyond_four_seconds(self):
        model = self.model()
        loader = DataLoader(self.batch_samples(), batch_size=2, collate_fn=data.collate_stage2_rhythm)
        batch = next(iter(loader))
        batch["wav"].requires_grad_()
        inputs = {key: value for key, value in batch.items()
                  if key not in ("utt_ids", "labels", "skipped_samples")}

        output = model(**inputs)
        loss = torch.nn.functional.cross_entropy(output["logits"], batch["labels"]) + output["ssl_loss"]
        loss.backward()

        self.assertGreater(batch["wav"].grad[0, 64000:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
