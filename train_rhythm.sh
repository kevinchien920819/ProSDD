#!/usr/bin/env bash
# ProSDD + Rhythm pipeline
#   Step 0: extract frame-level prosody embeddings (MPM, CPU) for 4 sets
#   Step 1: Stage 1 contrastive pre-training on bonafide speech (LibriSpeech)
#   Step 2: Stage 2 real/fake training with Rhythm duration fusion (ASVspoof2019 LA)
#   Step 3: Evaluate the Rhythm checkpoint on ASVspoof2019 LA / ASVspoof5
# 預設執行 Step 2 + Step 3；已有 checkpoint 時使用 EVAL_ONLY=1 bash train_rhythm.sh。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p prosody_txt output/logs_stage1contrastived output/logs_stage2realfake_rhythm

########## Step 0: extract prosody ##########
# ASVspoof2019 LA（utt ID 在第 2 欄，utt_col 使用從 0 開始的索引）
# 完整語音 train/dev targets：需要抽取時，手動取消註解或單獨執行下方指令。
# 輸出包含 .meta.json；不可沿用舊的四秒／200-frame cache。
uv run --locked python extract_full_prosody.py \
  --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --audio_dir    dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
  --out_txt      prosody_full_txt/asvspoof2019_train_prosody.txt \
  --utt_col 1 --ext .flac --layer 7 --skip_bad_samples

uv run --locked python extract_full_prosody.py \
  --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --audio_dir    dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
  --out_txt      prosody_full_txt/asvspoof2019_dev_prosody.txt \
  --utt_col 1 --ext .flac --layer 7 --skip_bad_samples

# LibriSpeech（清單每行只有 utt ID；音檔位於扁平 symlink 目錄）
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
export WANDB_MODE="${WANDB_MODE:-disabled}"
# 如需 W&B 記錄，可改成 export WANDB_MODE=offline 或 online。

# uv run --locked python main_stage1real.py \
#   --train_prosody_txt prosody_txt/librispeech_train_prosody.txt \
#   --dev_prosody_txt   prosody_txt/librispeech_dev_prosody.txt \
#   --train_spkmean_txt spk_text/librispeech_train_spkmean.txt \
#   --dev_spkmean_txt   spk_text/librispeech_dev_spkmean.txt \
#   --wav_dir_train    dataset/LibriSpeech_flat/train-clean-100 \
#   --wav_dir_dev      dataset/LibriSpeech_flat/dev-clean \
#   --audio_ext .flac \
#   --epochs 50 --batch_size 32 --num_workers 8 \
#   --ssl_lr 1e-6 --head_lr 1e-4 --weight_decay 1e-4 \
#   --mask_prob 0.25 --mask_span_len 8 --tau 0.07 \
#   --seed 1234 \
#   --wandb_tags prosdd stage1 librispeech \
#   --log_dir output/logs_stage1contrastived

########## Step 2: Stage 2 + Rhythm (ASVspoof2019 LA train / dev) ##########
export WANDB_NAME="${WANDB_STAGE2_NAME:-stage2-rhythm-full-asvspoof2019}"

# 接續上方 Stage 1 第 50 個 epoch 的 checkpoint。
# protocol、音訊、speaker、prosody 與 duration CSV 全部對應 ASVspoof2019 LA train/dev。
# duration 使用 TextGrid 產出的 ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_*.csv。
# 舊的 ASVspoof2019_LA_csv/*.csv 缺少音節與 duration 欄位，無法供 Rhythm loader 使用。
# prosody_full_txt/asvspoof2019_*_prosody.txt 須先以 Step 0 的完整語音抽取指令產生，不可沿用四秒 cache。
# prosody 啟動時串流掃描建立位置索引，訓練時按需讀取，不將整份特徵載入 RAM。
# 單筆樣本缺少或損壞、音檔讀取失敗、音節時間超出音檔時，自動記錄原因並略過。
# 初始化篩選記錄於 duration_filter.json；完整略過紀錄見 skipped_samples.jsonl。
# Loss 對齊 baseline：epoch 1–4 使用 cls + 0.2 * ssl，第 5 個 epoch 起使用 cls + 0.05 * ssl。
# 省略 --beta 才會套用此排程；明確指定 --beta 會改用固定權重。
if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
export PYTHONWARNINGS='ignore::UserWarning:torchaudio._backend.utils,ignore::UserWarning:torchaudio._backend.ffmpeg'
uv run --locked python main_stage2realfake_rhythm.py \
  --train_list dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --dev_list   dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --wav_dir_train dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
  --wav_dir_dev   dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
  --spkmean_txt_train spk_text/asvspoof2019_train_spkmean.txt \
  --spkmean_txt_dev   spk_text/asvspoof2019_dev_spkmean.txt \
  --prosody_txt_train prosody_full_txt/asvspoof2019_train_prosody.txt \
  --prosody_txt_dev   prosody_full_txt/asvspoof2019_dev_prosody.txt \
  --duration_csv_train dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_train.csv \
  --duration_csv_dev   dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_dev.csv \
  --skip_missing_duration \
  --skip_bad_samples \
  --stage1_ckpt output/logs_stage1contrastived/model_epoch_50.pth \
  --audio_ext .flac \
  --epochs 50 --batch_size 8 --max_batch_samples 0 --num_workers 8 \
  --lr_ssl_backbone 1e-6 --lr_ssl_head 1e-4 --lr_cls 1e-5 \
  --weight_decay 1e-4 \
  --alpha 1.0 \
  --mask_prob 0.15 --mask_span_len 8 --tau 0.1 \
  --num_time_neg 50 --num_spk_neg 50 \
  --rhythm_sources syllable \
  --d_model 256 --nhead 4 --n_rhythm_encoder_layers 2 --n_cls_encoder_layers 4 \
  --dropout 0.1 --max_position_embeddings 5000 \
  --algo 3 --augment_prob 0.5 \
  --wandb_tags prosdd rhythm stage2 full-utterance asvspoof2019 \
  --seed 1234 \
  --log_dir output/logs_stage2realfake_rhythm_syllable_full
fi

# Stage 2 的 main 每個 epoch 會執行 dev 驗證並記錄 loss、accuracy、EER。
# 輸出包含 config.json、duration_filter.json、skipped_samples.jsonl、metrics.jsonl、
# model_epoch_*.pth 與最低 dev EER 的 model_best.pth。

########## Step 3: Rhythm Evaluation (ASVspoof2019 LA / ASVspoof5 Track 1) ##########
# 使用第 50 個 epoch；可用 EVAL_CKPT / EVAL_SCORE_DIR 指定其他 checkpoint 與輸出目錄。
# 例如評估 model_best.pth 時，也將 EVAL_SCORE_DIR 設為 output/eval_stage2_best_rhythm。
eval_ckpt="${EVAL_CKPT:-output/logs_stage2realfake_rhythm_syllable_full/model_epoch_50.pth}"
eval_score_dir="${EVAL_SCORE_DIR:-output/eval_stage2_epoch50_rhythm_syllable_full}"
mkdir -p "$eval_score_dir"

# Rhythm 推論需要 duration CSV，不使用一般 main_eval.py 的 --classifier_pool。
# 從 checkpoint 同目錄的 config.json 還原模型，無需 speaker/prosody targets。
# 缺漏或損壞樣本會記錄於 *.coverage.json；CM 指標只針對成功評分的子集。
uv run --locked python main__eval_rhythm.py \
  --list_path dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
  --wav_dir dataset/ASVspoof2019/ASVspoof2019_LA_eval/flac \
  --duration_csv dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_eval.csv \
  --model_path "$eval_ckpt" \
  --save_scores_to "$eval_score_dir/asvspoof2019_la_eval.txt" \
  --save_metrics_to "$eval_score_dir/asvspoof2019_la_eval.metrics.json" \
  --batch_size 16 --max_batch_samples 640000 --num_workers 4 \
  --skip_bad_samples

uv run --locked python main__eval_rhythm.py \
  --list_path dataset/ASVspoof5/ASVspoof5.eval.track_1.tsv \
  --wav_dir dataset/ASVspoof5/flac_E_eval \
  --duration_csv dataset/ASVspoof5/ASVspoof5_cache_csv/cache_ASVspoof5_eval.csv \
  --model_path "$eval_ckpt" \
  --save_scores_to "$eval_score_dir/asvspoof5_eval.txt" \
  --save_metrics_to "$eval_score_dir/asvspoof5_eval.metrics.json" \
  --batch_size 16 --max_batch_samples 640000 --num_workers 4 \
  --skip_bad_samples

# ASVspoof2021 LA：取得對應的新版 duration CSV 後，設定 EVAL_2021_LA_DURATION_CSV，
# 再取消下方註解。其 CSV 必須包含 2021 utterance ID 對應的完整音節與 duration 欄位。
# eval_2021_la_protocol="$eval_score_dir/asvspoof2021_la_eval.protocol.txt"
# awk '$8 == "eval" {print $1, $2, "-", $5, $6}' \
#   dataset/ASVspoof2021/keys/LA/CM/trial_metadata.txt > "$eval_2021_la_protocol"
# uv run --locked python main__eval_rhythm.py \
#   --list_path "$eval_2021_la_protocol" \
#   --wav_dir dataset/ASVspoof2021/ASVspoof2021_LA_eval/flac \
#   --duration_csv "${EVAL_2021_LA_DURATION_CSV:?請指定 ASVspoof2021 LA duration CSV}" \
#   --model_path "$eval_ckpt" \
#   --save_scores_to "$eval_score_dir/asvspoof2021_la_eval.txt" \
#   --save_metrics_to "$eval_score_dir/asvspoof2021_la_eval.metrics.json" \
#   --batch_size 16 --num_workers 4 \
#   --skip_bad_samples
