"""Rhythm 評估入口暫停使用，待 data_utils_rhythm.py 完成串接。

保留 checkpoint 載入與命令列參數；舊資料載入與推論流程已移除。
"""

import argparse
import json
from pathlib import Path

import torch

from model_stage2realfake_rhythm import ProSDDStage2Rhythm
from prosody_utils import infer_checkpoint_prosody_dim


def load_model(model_path, config_path=None):
    config_path = Path(config_path) if config_path else Path(model_path).with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_class") != "ProSDDStage2Rhythm":
        raise ValueError("Expected a ProSDDStage2Rhythm training config")
    full_utterance = config.get("audio_mode") == "full_utterance"
    if full_utterance:
        if config.get("sample_rate") != 16000 or config.get("T_target") is not None or config.get("target_samples") is not None:
            raise ValueError("Full-utterance checkpoint must use dynamic audio/frame lengths")
    elif (config.get("sample_rate"), config.get("target_samples")) != (16000, 64000):
        raise ValueError("Expected the training audio policy: 16 kHz, center crop/pad to 4 s")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.get("state_dict", state).items()}
    if infer_checkpoint_prosody_dim(state) != config["prosody_dim"]:
        raise ValueError("Checkpoint and training config disagree on prosody_dim")
    options = (
        "model_name", "prosody_dim", "T_target", "rhythm_sources", "nhead",
        "n_rhythm_encoder_layers", "n_cls_encoder_layers", "dropout", "max_position_embeddings",
    )
    model = ProSDDStage2Rhythm(**{key: config[key] for key in options})
    # A partial load could silently leave an untrained classifier in place.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list_path", required=True, help="Labeled ASVspoof protocol")
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--duration_csv", required=True, help="Syllable/vowel/consonant duration cache")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", help="Defaults to config.json beside the checkpoint")
    parser.add_argument("--save_scores_to", required=True)
    parser.add_argument("--save_metrics_to")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batch_samples", type=int, default=640000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--audio_ext", default=".flac")
    parser.add_argument("--skip_bad_samples", action="store_true",
                        help="Record excluded IDs; report metrics only for successfully scored samples")
    return parser


def main(argv=None):
    """保留 argv 介面；新版資料流程完成前以錯誤狀態結束，不產生評估輸出。"""
    raise SystemExit("Rhythm 評估暫停使用：舊資料流程已移除，待 data_utils_rhythm.py 完成串接。")


if __name__ == "__main__":
    main()
