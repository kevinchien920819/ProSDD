#!/usr/bin/env bash
# ASVspoof2019 LA 的 ProSDD Stage 2 Rhythm 訓練與評估入口。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

log_dir="${RHYTHM_LOG_DIR:-output/logs_stage2realfake_rhythm_crop}"
mkdir -p "$log_dir"

uv run --locked python main_stage2realfake_rhythm.py \
  --train_list dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --dev_list dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --wav_dir_train dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
  --wav_dir_dev dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
  --spkmean_txt_train spk_text/asvspoof2019_train_spkmean.txt \
  --spkmean_txt_dev spk_text/asvspoof2019_dev_spkmean.txt \
  --duration_csv_train dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_train.csv \
  --duration_csv_dev dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_dev.csv \
  --stage1_ckpt output/logs_stage1contrastived/model_epoch_50.pth \
  --teacher_kind mpm --prosody_layer 7 \
  --audio_seconds 4 \
  --epochs 50 --batch_size 8 --num_workers 8 \
  --lr_ssl_backbone 1e-6 --lr_ssl_head 1e-4 \
  --lr_rhythm 1e-5 --lr_cls 1e-5 \
  --weight_decay 1e-4 --seed 1234 \
  --rhythm_sources syllable --mask_prob 0.15 --tau 0.1 \
  --algo 3 --augment_prob 0.5 --skip_bad_samples \
  --wandb_mode online \
  --wandb_tags prosdd rhythm stage2 asvspoof2019 \
  --log_dir "$log_dir"

# 訓練成功後，以最低 dev EER 的 checkpoint 評估 eval split。
eval_dir="$log_dir/eval"
uv run --locked python main__eval_rhythm.py \
  --list_path dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
  --wav_dir dataset/ASVspoof2019/ASVspoof2019_LA_eval/flac \
  --duration_csv dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_eval.csv \
  --model_path "$log_dir/model_best.pth" \
  --config_path "$log_dir/config.json" \
  --save_scores_to "$eval_dir/asvspoof2019_la_eval.txt" \
  --save_metrics_to "$eval_dir/asvspoof2019_la_eval.metrics.json" \
  --batch_size 16 --max_batch_samples 640000 --num_workers 4 \
  --seed 1234 --skip_bad_samples
