#!/usr/bin/env bash
# 暫停執行；下方保留既有命令，待新版資料介面完成後再調整。
printf '%s\n' 'Rhythm pipeline 暫停使用：舊資料流程已移除，待 data_utils_rhythm.py 完成串接。' >&2
exit 1

# ProSDD + Rhythm pipeline (VAD prosody)
#   Step 0: extract frame-level prosody embeddings (MPM, CPU) for 4 sets
#   Step 1: Stage 1 contrastive pre-training on bonafide speech (LibriSpeech)
#   Step 2: Stage 2 real/fake training with Rhythm duration fusion (ASVspoof2019 LA)
# 已跑完的步驟可直接註解掉；預設僅啟用 Step 2。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Step 0 使用的 VAD MPM teacher checkpoint；可直接修改為自己的目錄。
checkpoint_dir="${VAD_TEACHER_CHECKPOINT:-prosody_checkpoint}"
mkdir -p prosody_vad_txt output/logs_stage1contrastived_vadv2 output/logs_stage2realfake_rhythm_vad_full

########## Step 0: extract prosody ##########
# ASVspoof2019 LA（utt ID 在第 2 欄，utt_col 使用從 0 開始的索引）
# uv run --locked python extract_Prosody_vad.py \
#   --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
#   --audio_dir    dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
#   --out_txt      prosody_vad_txt/asvspoof2019_train_prosody.txt \
#   --utt_col 1 --ext .flac --layer 7 \
#   --checkpoint_dir "$checkpoint_dir"

# uv run --locked python extract_Prosody_vad.py \
#   --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
#   --audio_dir    dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
#   --out_txt      prosody_vad_txt/asvspoof2019_dev_prosody.txt \
#   --utt_col 1 --ext .flac --layer 7 \
#   --checkpoint_dir "$checkpoint_dir"

# LibriSpeech（清單每行只有 utt ID；音檔位於扁平 symlink 目錄）
# uv run --locked python extract_Prosody_vad.py \
#   --protocol_txt protocols/librispeech_train-clean-100.txt \
#   --audio_dir    dataset/LibriSpeech_flat/train-clean-100 \
#   --out_txt      prosody_vad_txt/librispeech_train_prosody.txt \
#   --utt_col 0 --ext .flac --layer 7 \
#   --checkpoint_dir "$checkpoint_dir"

# uv run --locked python extract_Prosody_vad.py \
#   --protocol_txt protocols/librispeech_dev-clean.txt \
#   --audio_dir    dataset/LibriSpeech_flat/dev-clean \
#   --out_txt      prosody_vad_txt/librispeech_dev_prosody.txt \
#   --utt_col 0 --ext .flac --layer 7 \
#   --checkpoint_dir "$checkpoint_dir"

########## Step 1: Stage 1 (LibriSpeech train-clean-100 / dev-clean) ##########
export WANDB_PROJECT="${WANDB_PROJECT:-ProSDD}"
export WANDB_NAME="${WANDB_NAME:-stage1-librispeech-vad}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
# 如需 W&B 記錄，可改成 export WANDB_MODE=offline 或 online。

# uv run --locked python main_stage1real.py \
#   --train_prosody_txt prosody_vad_txt/librispeech_train_prosody.txt \
#   --dev_prosody_txt   prosody_vad_txt/librispeech_dev_prosody.txt \
#   --train_spkmean_txt spk_text/librispeech_train_spkmean.txt \
#   --dev_spkmean_txt   spk_text/librispeech_dev_spkmean.txt \
#   --wav_dir_train    dataset/LibriSpeech_flat/train-clean-100 \
#   --wav_dir_dev      dataset/LibriSpeech_flat/dev-clean \
#   --audio_ext .flac \
#   --epochs 50 --batch_size 32 --num_workers 8 \
#   --ssl_lr 1e-6 --head_lr 1e-4 --weight_decay 1e-4 \
#   --mask_prob 0.25 --mask_span_len 8 --tau 0.07 \
#   --seed 1234 \
#   --wandb_tags prosdd vad stage1 librispeech \
#   --log_dir output/logs_stage1contrastived_vadv2

########## Step 2: Stage 2 + Rhythm (ASVspoof2019 LA train / dev) ##########
export WANDB_NAME="${WANDB_STAGE2_NAME:-stage2-rhythm-vad-full-asvspoof2019}"

# 使用指定的 Stage 1 第 50 個 epoch checkpoint，使用 VAD frame-level prosody targets。
# 請將下方兩個 --duration_csv 路徑改成同一份 train/dev protocol 對應的 syllabification CSV。
# CSV 需包含音節起訖時間及每個音節的 duration；兩個 prosody 版本可共用這兩份 CSV。
# Loss 對齊 baseline：epoch 1–4 使用 cls + 0.2 * ssl，第 5 個 epoch 起使用 cls + 0.05 * ssl。
# 省略 --beta 才會套用此排程；明確指定 --beta 會改用固定權重。
# 完整語音：首次執行先建立 train/dev targets 與 .meta.json，後續沿用。
# Teacher 逐段處理全部內容；Stage II 本身一次接收整段音訊，沒有裁切。
mkdir -p prosody_vad_full_txt
for prosody_split in train dev; do
  full_cache="prosody_vad_full_txt/asvspoof2019_${prosody_split}_prosody.txt"
  protocol_suffix="trl"
  if [[ "$prosody_split" == "train" ]]; then protocol_suffix="trn"; fi
  if [[ ! -f "${full_cache}.meta.json" ]]; then
    uv run --locked python extract_full_prosody.py \
      --protocol_txt "dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.${prosody_split}.${protocol_suffix}.txt" \
      --audio_dir "dataset/ASVspoof2019/ASVspoof2019_LA_${prosody_split}/flac" \
      --out_txt "$full_cache" --utt_col 1 --ext .flac --layer 7 --teacher_kind vad --teacher_checkpoint "$checkpoint_dir" \
      --skip_bad_samples
  fi
done

uv run --locked python main_stage2realfake_rhythm.py \
  --train_list dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --dev_list   dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --wav_dir_train dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
  --wav_dir_dev   dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
  --spkmean_txt_train spk_text/asvspoof2019_train_spkmean.txt \
  --spkmean_txt_dev   spk_text/asvspoof2019_dev_spkmean.txt \
  --prosody_txt_train prosody_vad_full_txt/asvspoof2019_train_prosody.txt \
  --prosody_txt_dev   prosody_vad_full_txt/asvspoof2019_dev_prosody.txt \
  --duration_csv_train dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_train.csv \
  --duration_csv_dev   dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_dev.csv  \
  --stage1_ckpt output/logs_stage1contrastived_vadv2/model_epoch_50.pth \
  --audio_ext .flac \
  --epochs 50 --batch_size 32 --max_batch_samples 0 --num_workers 8 \
  --lr_ssl_backbone 1e-6 --lr_ssl_head 1e-4 --lr_rhythm 1e-5 --lr_cls 1e-5 \
  --weight_decay 1e-4 \
  --alpha 1.0 \
  --mask_prob 0.15 --mask_span_len 8 --tau 0.1 \
  --num_time_neg 50 --num_spk_neg 50 \
  --rhythm_sources syllable vowel consonant \
  --nhead 4 --n_rhythm_encoder_layers 2 --n_cls_encoder_layers 4 \
  --dropout 0.1 --max_position_embeddings 5000 \
  --algo 3 --augment_prob 0.5 \
  --wandb_tags prosdd vad rhythm stage2 full-utterance asvspoof2019 \
  --seed 1234 \
  --log_dir output/logs_stage2realfake_rhythm_vad_full

# Stage 2 的 main 每個 epoch 會執行 dev 驗證並記錄 loss、accuracy、EER。
# 輸出包含 config.json、metrics.jsonl、最低 val loss 的 model_best.pth。
