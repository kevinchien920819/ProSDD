#!/usr/bin/env bash
# ProSDD pipeline
#   Step 0: extract frame-level prosody embeddings (MPM, CPU) for 4 sets
#   Step 1: Stage 1 contrastive pre-training on bonafide speech (LibriSpeech)
# 已跑完的步驟可直接註解掉。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p prosody_txt logs_stage1contrastive

########## Step 0: extract prosody ##########
# ASVspoof2019 LA (protocol 第 1 欄是 utt ID)
uv run --locked python extract_Prosody.py \
  --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --audio_dir    dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
  --out_txt      prosody_txt/asvspoof2019_train_prosody.txt \
  --utt_col 1 --ext .flac --layer 7

uv run --locked python extract_Prosody.py \
  --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --audio_dir    dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
  --out_txt      prosody_txt/asvspoof2019_dev_prosody.txt \
  --utt_col 1 --ext .flac --layer 7

# LibriSpeech (清單每行只有 utt ID；扁平 symlink 目錄由 protocols/ 與 dataset/LibriSpeech_flat/ 提供)
uv run --locked python extract_Prosody.py \
  --protocol_txt protocols/librispeech_train-clean-100.txt \
  --audio_dir    dataset/LibriSpeech_flat/train-clean-100 \
  --out_txt      prosody_txt/librispeech_train_prosody.txt \
  --utt_col 0 --ext .flac --layer 7

uv run --locked python extract_Prosody.py \
  --protocol_txt protocols/librispeech_dev-clean.txt \
  --audio_dir    dataset/LibriSpeech_flat/dev-clean \
  --out_txt      prosody_txt/librispeech_dev_prosody.txt \
  --utt_col 0 --ext .flac --layer 7

########## Step 1: Stage 1 (LibriSpeech train-clean-100 / dev-clean) ##########
export WANDB_PROJECT="${WANDB_PROJECT:-ProSDD}"
export WANDB_NAME="${WANDB_NAME:-stage1-librispeech}"
# 無網路時改成: export WANDB_MODE=offline

# batch_size 32 在 RTX 5090 32GB 上峰值約 22GB；64 會 OOM
uv run --locked python main_stage1real.py \
  --train_prosody_txt prosody_txt/librispeech_train_prosody.txt \
  --dev_prosody_txt   prosody_txt/librispeech_dev_prosody.txt \
  --train_spkmean_txt spk_text/librispeech_train_spkmean.txt \
  --dev_spkmean_txt   spk_text/librispeech_dev_spkmean.txt \
  --wav_dir_train     dataset/LibriSpeech_flat/train-clean-100 \
  --wav_dir_dev       dataset/LibriSpeech_flat/dev-clean \
  --audio_ext .flac \
  --epochs 50 --batch_size 32 --num_workers 8 \
  --ssl_lr 1e-6 --head_lr 1e-4 --weight_decay 1e-4 \
  --mask_prob 0.25 --mask_span_len 8 --tau 0.07 \
  --seed 1234 \
  --log_dir logs_stage1contrastived
