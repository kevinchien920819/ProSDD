#!/usr/bin/env bash
# ProSDD pipeline
#   Step 0: extract frame-level prosody embeddings (MPM, CPU) for 4 sets
#   Step 1: Stage 1 contrastive pre-training on bonafide speech (LibriSpeech)
# 已跑完的步驟可直接註解掉。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p prosody_txt output/logs_stage1contrastived output/logs_stage2realfake_batch_8_sec_10_asvspoof5

########## Step 0: extract prosody ##########
# ASVspoof5（utt ID 在第 2 欄；--utt_col 使用從 0 開始的索引）
# uv run --locked python extract_Prosody.py \
#   --protocol_txt dataset/ASVspoof5/ASVspoof5.train.tsv \
#   --audio_dir    dataset/ASVspoof5/flac_T \
#   --out_txt      prosody_txt/asvspoof5_train_prosody.txt \
#   --utt_col 1 --ext .flac --layer 7

# uv run --locked python extract_Prosody.py \
#   --protocol_txt dataset/ASVspoof5/ASVspoof5.dev.track_1.tsv \
#   --audio_dir    dataset/ASVspoof5/flac_D \
#   --out_txt      prosody_txt/asvspoof5_dev_prosody.txt \
#   --utt_col 1 --ext .flac --layer 7

# # LibriSpeech (清單每行只有 utt ID；扁平 symlink 目錄由 protocols/ 與 dataset/LibriSpeech_flat/ 提供)
# uv run --locked python extract_Prosody.py \
#   --protocol_txt protocols/librispeech_train-clean-100.txt \
#   --audio_dir    dataset/LibriSpeech_flat/train-clean-100 \
#   --out_txt      prosody_txt/librispeech_train_prosody.txt \
#   --utt_col 0 --ext .flac --layer 7

# uv run --locked python extract_Prosody.py \
#   --protocol_txt protocols/librispeech_dev-clean.txt \
#   --audio_dir    dataset/LibriSpeech_flat/dev-clean \
#   --out_txt      prosody_txt/librispeech_dev_prosody.txt \
#   --utt_col 0 --ext .flac --layer 7

########## Step 1: Stage 1 (LibriSpeech train-clean-100 / dev-clean) ##########
export WANDB_PROJECT="${WANDB_PROJECT:-ProSDD}"
export WANDB_NAME="${WANDB_NAME:-stage1-librispeech}"
# 無網路時改成: export WANDB_MODE=offline

# batch_size 32 在 RTX 5090 32GB 上峰值約 22GB；64 會 OOM
# uv run --locked python main_stage1real.py \
#   --train_prosody_txt prosody_txt/librispeech_train_prosody.txt \
#   --dev_prosody_txt   prosody_txt/librispeech_dev_prosody.txt \
#   --train_spkmean_txt spk_text/librispeech_train_spkmean.txt \
#   --dev_spkmean_txt   spk_text/librispeech_dev_spkmean.txt \
#   --wav_dir_train     dataset/LibriSpeech_flat/train-clean-100 \
#   --wav_dir_dev       dataset/LibriSpeech_flat/dev-clean \
#   --audio_ext .flac \
#   --epochs 50 --batch_size 32 --num_workers 8 \
#   --ssl_lr 1e-6 --head_lr 1e-4 --weight_decay 1e-4 \
#   --mask_prob 0.25 --mask_span_len 8 --tau 0.07 \
#   --seed 1234 \
#   --wandb_tags prosdd stage1 librispeech \
#   --log_dir output/logs_stage1contrastived

########## Step 2: Stage 2 (ASVspoof5 train / dev Track 1) ##########
export WANDB_NAME="${WANDB_STAGE2_NAME:-stage2-asvspoof5}"

# 使用指定的 Stage 1 第 50 個 epoch checkpoint。
# `asvspoof2024_*_spkmean.txt` 是 ASVspoof5 train/dev 的 speaker embedding 快取。
uv run --locked python main_stage2realfake.py \
  --train_list dataset/ASVspoof5/ASVspoof5.train.tsv \
  --dev_list   dataset/ASVspoof5/ASVspoof5.dev.track_1.tsv \
  --wav_dir_train dataset/ASVspoof5/flac_T \
  --wav_dir_dev   dataset/ASVspoof5/flac_D \
  --spkmean_txt_train spk_text/asvspoof2024_train_spkmean.txt \
  --spkmean_txt_dev   spk_text/asvspoof2024_dev_spkmean.txt \
  --prosody_txt_train prosody_txt/asvspoof5_train_prosody.txt \
  --prosody_txt_dev   prosody_txt/asvspoof5_dev_prosody.txt \
  --stage1_ckpt output/logs_stage1contrastived/model_epoch_50.pth \
  --audio_ext .flac \
  --audio_seconds 10 \
  --epochs 50 --batch_size 8 --num_workers 8 \
  --lr_ssl_backbone 1e-6 --lr_ssl_head 1e-4 --lr_cls 1e-5 \
  --weight_decay 1e-4 \
  --seed 1234 \
  --wandb_tags prosdd stage2 asvspoof5 batch_8 second_10\
  --log_dir output/logs_stage2realfake_batch_8_sec_10_asvspoof5

########## Step 3: Evaluation (ASVspoof5 Track 1) ##########
# 使用 Stage 2 最低 val loss 的 checkpoint，輸出逐音檔分數與 CM 指標。
eval_ckpt="output/logs_stage2realfake_batch_8_sec_10_asvspoof5/model_best.pth"
eval_score_dir="output/eval_stage2_best"
mkdir -p "$eval_score_dir"

uv run --locked python main_eval.py \
  --list_path dataset/ASVspoof5/ASVspoof5.eval.track_1.tsv \
  --wav_dir dataset/ASVspoof5/flac_E_eval \
  --model_path "$eval_ckpt" \
  --save_scores_to "$eval_score_dir/asvspoof5_eval.txt" \
  --save_metrics_to "$eval_score_dir/asvspoof5_eval.metrics.json" \
  --batch_size 16 \
  --classifier_pool mean

# ASVspoof2021 LA：選取 eval 子集，轉成五欄 CM protocol 供推論與指標計算使用。
# eval_2021_la_protocol="$eval_score_dir/asvspoof2021_la_eval.protocol.txt"
# awk '$8 == "eval" {print $1, $2, "-", $5, $6}' \
#   dataset/ASVspoof2021/keys/LA/CM/trial_metadata.txt > "$eval_2021_la_protocol"

# uv run --locked python main_eval.py \
#   --list_path "$eval_2021_la_protocol" \
#   --wav_dir dataset/ASVspoof2021/ASVspoof2021_LA_eval/flac \
#   --model_path "$eval_ckpt" \
#   --save_scores_to "$eval_score_dir/asvspoof2021_la_eval.txt" \
#   --save_metrics_to "$eval_score_dir/asvspoof2021_la_eval.metrics.json" \
#   --batch_size 16 \
#   --classifier_pool mean
