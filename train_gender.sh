#!/usr/bin/env bash
# ProSDD gender pretraining pipeline
#   Step 0: extract frame-level prosody embeddings (MPM, CPU) for 4 sets
#   Step 1: Stage 1 contrastive pre-training on bonafide speech (LibriSpeech)
#   Step 2: Stage 2 真假分類與完整 450 維 embedding 監督
# 已跑完的步驟可直接註解掉。
set -euo pipefail
# 忽略 torchaudio 2.8 的 TorchCodec 遷移警告（StreamReader／load 即將改版），不影響執行。
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore::UserWarning:torchaudio._backend.utils,ignore::UserWarning:torchaudio._backend.ffmpeg"
cd "$(dirname "$0")"
stage1_log_dir="output/logs_stage1contrastived_gender_rotation_45"
stage2_log_dir="output/logs_stage2realfake_batch_8_sec_4_2019LA_gender_rotation_45"
mkdir -p prosody_txt "$stage1_log_dir" "$stage2_log_dir"

########## Step 0: extract prosody ##########
# ASVspoof2019 LA (protocol 第 1 欄是 utt ID)
# uv run --locked python extract_Prosody.py \
#   --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
#   --audio_dir    dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
#   --out_txt      prosody_txt/asvspoof2019_train_prosody.txt \
#   --utt_col 1 --ext .flac --layer 7

# uv run --locked python extract_Prosody.py \
#   --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
#   --audio_dir    dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
#   --out_txt      prosody_txt/asvspoof2019_dev_prosody.txt \
#   --utt_col 1 --ext .flac --layer 7

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
export WANDB_NAME="${WANDB_NAME:-stage1-librispeech-gender-rotation45}"
# 無網路時改成: export WANDB_MODE=offline

# batch_size 32 在 RTX 5090 32GB 上峰值約 22GB；64 會 OOM
uv run --locked python main_stage1real.py \
  --train_prosody_txt prosody_txt/librispeech_train_prosody.txt \
  --dev_prosody_txt prosody_txt/librispeech_dev_prosody.txt \
  --train_spkmean_txt spk_text/librispeech_train_spkmean.txt \
  --dev_spkmean_txt spk_text/librispeech_dev_spkmean.txt \
  --wav_dir_train dataset/LibriSpeech_flat/train-clean-100 \
  --wav_dir_dev dataset/LibriSpeech_flat/dev-clean \
  --audio_ext .flac \
  --epochs 50 --batch_size 32 --num_workers 8 \
  --ssl_lr 1e-6 --head_lr 1e-4 --weight_decay 1e-4 \
  --mask_prob 0.25 --mask_span_len 8 --tau 0.07 \
  --seed 1234 \
  --use_gender \
  --gender_txt dataset/LibriSpeech/SPEAKERS.TXT \
  --gender_theta 45 \
  --wandb_tags prosdd stage1 librispeech gender rotation45 \
  --log_dir "$stage1_log_dir"

########## Step 2: Stage 2 (ASVspoof2019 LA train / dev) ##########
export WANDB_NAME="${WANDB_STAGE2_NAME:-stage2-asvspoof2019-gender-rotation45}"

# 保留 Stage 1 完整 450 維 projection，兩階段使用相同的 gender 旋轉角度。
uv run --locked python main_stage2realfake.py \
  --train_list dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --dev_list   dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --wav_dir_train dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
  --wav_dir_dev   dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
  --spkmean_txt_train spk_text/asvspoof2019_train_spkmean.txt \
  --spkmean_txt_dev   spk_text/asvspoof2019_dev_spkmean.txt \
  --prosody_txt_train prosody_txt/asvspoof2019_train_prosody.txt \
  --prosody_txt_dev   prosody_txt/asvspoof2019_dev_prosody.txt \
  --prosody_dim 256 \
  --stage1_ckpt "$stage1_log_dir/model_last.pth" \
  --use_gender \
  --gender_txt spk_text/asvspoof2019_gender.txt \
  --gender_theta 45 \
  --audio_ext .flac \
  --audio_seconds 4 \
  --epochs 50 --batch_size 8 --num_workers 8 \
  --lr_ssl_backbone 1e-6 --lr_ssl_head 1e-4 --lr_cls 1e-5 \
  --weight_decay 1e-4 \
  --seed 1234 \
  --wandb_tags prosdd stage2 asvspoof2019 batch_8 second_4 gender rotation45 \
  --log_dir "$stage2_log_dir"

########## Step 3: Evaluation (ASVspoof2019 LA / ASVspoof5 Track 1 / ASVspoof2021 LA) ##########
# 使用 Stage 2 最低 val loss 的 checkpoint，輸出逐音檔分數與 CM 指標。
eval_ckpt="$stage2_log_dir/model_best.pth"
eval_score_dir="$stage2_log_dir/eval_stage2_best"
mkdir -p "$eval_score_dir"

uv run --locked python main_eval.py \
  --list_path dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
  --wav_dir dataset/ASVspoof2019/ASVspoof2019_LA_eval/flac \
  --model_path "$eval_ckpt" \
  --save_scores_to "$eval_score_dir/asvspoof2019_la_eval.txt" \
  --save_metrics_to "$eval_score_dir/asvspoof2019_la_eval.metrics.json" \
  --batch_size 16 \
  --classifier_pool mean

uv run --locked python main_eval.py \
  --list_path dataset/ASVspoof5/ASVspoof5.eval.track_1.tsv \
  --wav_dir dataset/ASVspoof5/flac_E_eval \
  --model_path "$eval_ckpt" \
  --save_scores_to "$eval_score_dir/asvspoof5_eval.txt" \
  --save_metrics_to "$eval_score_dir/asvspoof5_eval.metrics.json" \
  --batch_size 16 \
  --classifier_pool mean
