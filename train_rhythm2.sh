#!/usr/bin/env bash
# ASVspoof2019 LA 的 ProSDD Stage 2 Rhythm2 訓練與評估入口，使用 VAD teacher。
# Rhythm2 = ProSDDStage2 backbone + rhythm-transformer 融合層（model_stage2realfake_rhythm2.py）；
# 參數與 train_rhythm.sh 相同，只換訓練入口與輸出目錄，方便對照實驗。
set -euo pipefail
# 忽略 torchaudio 2.8 的 TorchCodec 遷移警告（StreamReader／load 即將改版），不影響執行。
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore::UserWarning:torchaudio._backend.utils,ignore::UserWarning:torchaudio._backend.ffmpeg"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 預設 teacher 對應既有 VAD Stage 1 的 128 維、layer 7 targets。
log_dir="${RHYTHM2_LOG_DIR:-output/logs_stage2realfake_rhythm2_syllable_batch_8_sec_all_2019LA}"
mkdir -p "$log_dir"

uv run --locked python main_stage2realfake_rhythm2.py \
  --train_list dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --dev_list dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --wav_dir_train dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
  --wav_dir_dev dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac \
  --spkmean_txt_train spk_text/asvspoof2019_train_spkmean.txt \
  --spkmean_txt_dev spk_text/asvspoof2019_dev_spkmean.txt \
  --duration_csv_train dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_train.csv \
  --duration_csv_dev dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_dev.csv \
  --stage1_ckpt output/logs_stage1contrastived_vadv2/model_epoch_50.pth \
  --teacher_kind vad --teacher_checkpoint "prosody_checkpoint/mpm_vad_random_1_128_v4_dim64_layer8/step_10000" --prosody_layer 7 \
  --audio_seconds 0 \
  --epochs 50 --batch_size 8 --num_workers 8 \
  --lr_ssl_backbone 1e-6 --lr_ssl_head 1e-4 \
  --lr_rhythm 1e-5 --lr_cls 1e-5 \
  --weight_decay 1e-4 --seed 1234 \
  --rhythm_sources syllable --mask_prob 0.15 --tau 0.1 \
  --n_rhythm_encoder_layers 2 --n_cls_encoder_layers 4 \
  --algo 3 --augment_prob 0.5 --skip_bad_samples \
  --wandb_mode online \
  --wandb_tags prosdd vad stage2 asvspoof2019 batch_8 second_4 rhythm2 syllable \
  --wandb_name "stage2-asvspoof2019-rhythm2" \
  --log_dir "$log_dir"

# 訓練成功後，以最低 dev EER 的 checkpoint 評估 eval split；config.json 的 model_class 會還原 Rhythm2。
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

# 使用同一個 checkpoint 評估 ASVspoof5 Track 1。
uv run --locked python main__eval_rhythm.py \
  --list_path dataset/ASVspoof5/ASVspoof5.eval.track_1.tsv \
  --wav_dir dataset/ASVspoof5/flac_E_eval \
  --duration_csv dataset/ASVspoof5/ASVspoof5_cache_csv/cache_ASVspoof5_eval.csv \
  --model_path "$log_dir/model_best.pth" \
  --config_path "$log_dir/config.json" \
  --save_scores_to "$eval_dir/asvspoof5_eval.txt" \
  --save_metrics_to "$eval_dir/asvspoof5_eval.metrics.json" \
  --batch_size 16 --max_batch_samples 640000 --num_workers 0 \
  --seed 1234 --skip_bad_samples
