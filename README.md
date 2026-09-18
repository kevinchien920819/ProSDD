# ProSDD

Official implementation of the paper:

**ProSDD: Learning Prosodic Representations for Speech Deepfake Detection against Expressive and Emotional Attacks**

[arXiv Paper](https://arxiv.org/abs/2604.13229) | [Project Website](https://prosdd.github.io/ProSDD_website/)

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/ProSDD/codes.git
cd codes
```

### Setup Environment

使用 `uv` 管理 Python 3.10 與專案依賴。訓練、評估及韻律特徵擷取共用
`pyproject.toml` 與 `uv.lock`：

```bash
uv sync --locked
source .venv/bin/activate
```

PyTorch 相關套件已納入 `uv` 管理，固定使用以下相容版本：

- `torch==2.8.0`
- `torchaudio==2.8.0`
- `torchvision==0.23.0`
- `triton==3.4.0`（Linux x86_64）

`torch`、`torchaudio` 與 `torchvision` 使用 PyTorch 官方 CUDA 12.8
索引，其餘依賴由 PyPI 解析。`uv sync --locked` 會一併安裝，
不需要另外手動安裝 PyTorch。GPU 運算需有可正常運作的相容 NVIDIA 驅動程式。
版本對應可參考 [PyTorch 官方版本說明](https://docs.pytorch.org/get-started/previous-versions/)
與 [PyTorch 2.8.0 的 Triton 版本設定](https://github.com/pytorch/pytorch/blob/v2.8.0/.ci/docker/triton_version.txt)。

其他執行前提：

- 系統需有 `ffmpeg` 指令，供音訊解碼使用；它不由 `uv` 安裝。
- 目前儲存庫未包含 `RawBoost` 與 `core_scripts.startup_config`。
  請補入原始實作或讓它們可由 Python 匯入；Stage 1／Stage 2 使用
  `set_random_seed`，Stage 2 另外使用 RawBoost 音訊增強。

原本的 `environment.yml` 與 `environment_prosody.yml` 保留作為歷史環境參考，
使用 `uv` 時不需要建立 Conda 環境。
### Pre-trained SSL Backbone

ProSDD uses [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m) as the pre-trained speech encoder for both Stage 1 and Stage 2.

The model is downloaded automatically through the Hugging Face `transformers` library when the training or evaluation scripts are run for the first time.

## Implementation Guidelines

### Pre-trained Checkpoints and Supervised Targets

Pre-trained checkpoints and supervised targets (speaker embeddings only) are available at the following link. Since the prosody embedding files are large, we provide the code used to extract the frame-level prosody embeddings.

Link:  
https://drive.google.com/drive/folders/1h250-Um5qWo-rpeOE6K_Gdeygy46k3xg?usp=sharing

### Available Checkpoints
1. ProSDD trained on ASVspoof 2019
2. ProSDD trained on ASVspoof 2024
3. Baselines trained on ASVspoof 2024: RawNet2; AASIST; XLSR-SLS  

These baselines are provided to help the community efficiently use the ASVspoof 2024 dataset.

### Evaluation Scores
We also release the evaluation scores for all provided checkpoints.

### 計算 EER、Cllr、minDCF、actDCF

`evaluation_metric/` 已從 `rhythm-transformer/src/evaluation_metric` 移入，
提供 CM 指標計算。既有分數檔可以直接評估，不需要重新推論或使用 GPU：

```bash
uv run --locked python -m evaluation_metric \
  --score_path output/eval_stage2_best/asvspoof2019_la_eval.txt \
  --protocol_path dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
  --save_metrics_to output/eval_stage2_best/asvspoof2019_la_eval.metrics.json
```

若要在推論後直接計算，在 `main_eval.py` 命令加上
`--save_metrics_to /path/to/metrics.json`。標籤預設取自 `--list_path`；
如果音檔清單與標籤分開保存，另外指定 `--metrics_protocol /path/to/keys.tsv`。
`train.sh` 的兩個 evaluation 步驟都已啟用此選項。

支援 ASVspoof 2019 五欄 protocol、ASVspoof5 Track 1 十欄 protocol，
以及兩欄 ID／標籤檔和具名 TSV。分數依 utterance ID 對齊，缺漏或重複 ID
會回報錯誤。JSON 包含四項指標、樣本數及 cost model；`eer` 是 0–1 比例，
`eer_percent` 是百分比，`cllr` 單位為 bits。

沿用來源的 CM cost model：`Pspoof=0.05`、`Cmiss=1`、`Cfa=10`。
`main_eval.py` 的分數仍是 `logits[:, 1]`；Cllr／actDCF 直接以這組分數作為
LLR 輸入計算，未做校正。原始 a-DCF／t-DCF／t-EER 函式也已移入，但需要
額外 ASV／SASV 資料，目前 CM 入口不會計算這些指標。詳細格式、來源與
分數語意見 [evaluation_metric/README.md](evaluation_metric/README.md)。

### Prosody Extraction Environment

Frame-level prosody embeddings are extracted using the [Masked Prosody Model](https://huggingface.co/cdminix/masked_prosody_model).

韻律特徵擷取所需的 `masked-prosody-model` 已納入共用的 `uv` 環境。
完成上述環境設定後，可執行：

```bash
uv run --locked python extract_Prosody.py
```

The speaker embeddings used as supervised targets are available in the Google Drive folder linked above.

### Running the Code

The repository provides separate scripts for each stage of the ProSDD pipeline:

- `extract_Prosody.py`: Extract frame-level prosody embeddings
- `main_stage1real.py`: Train Stage 1 using bonafide speech
- `main_stage2realfake.py`: Train Stage 2 using bonafide and spoofed speech
- `main_eval.py`: Evaluate a trained ProSDD checkpoint

一般 ProSDD Stage 2 可用 `--audio_seconds` 控制固定輸入長度，預設為 4 秒。
程式會以 16 kHz 自動換算 `max_len`，並按 XLS-R 每 20 ms 一個 frame 自動設定
`T_target`（例如 6 秒為 96,000 samples 與 300 frames）。非 4 秒實驗必須使用
相同長度重新抽取 prosody targets；否則音訊與 prosody 的時間對齊會失效。
`--T_target` 僅保留給重現舊實驗時手動覆寫。

### Prosody 維度（128／256）

Stage 1 與 Stage 2 預設從訓練 prosody 檔案自動判定維度，支援 128 與 256。
也可以在原本的訓練命令加上 `--prosody_dim 128` 或 `--prosody_dim 256`
明確指定；若與資料不符，會在載入時回報錯誤並指出檔案與 utterance。
同一份檔案內的所有 utterance，以及同次訓練的 train／dev，必須使用相同維度。

Prosody targets 使用檔案位置索引：每次啟動會串流掃描文字檔，檢查各筆
frame 數與維度，記憶體只保留 utterance ID、位置與形狀；需要組成 batch 時
才解析對應的 float32 tensor。原本的 `prosody_txt` 可直接使用，不需重新抽取。
ASVspoof5 的 train／dev targets 若全部載入約需 61.7 GiB RAM，索引方式可避免
在 Dataset 初始化時耗盡記憶體。每次啟動的掃描仍需要磁碟讀取時間，
程式會顯示索引進度。各個 DataLoader worker 分別開檔讀取；訓練期間請保持
來源檔案不變，若檔案被覆寫，程式會要求重新建立 Dataset。

`train_rhythm.sh` 已啟用 `--skip_bad_samples`：單筆 duration 缺少或損壞、
speaker／prosody 特徵缺少或形狀錯誤、音檔讀取失敗、中央 4 秒內沒有完整音節、
特徵含非有限值，或音訊太短而無法產生有效 CNN frame 時，會自動略過並記錄原因。
剩餘樣本照常組成 batch；整批都不可用時繼續下一批。沒有啟用此參數時維持嚴格檢查。
OOM、模型運算失敗、整份 CSV 欄位錯誤或訓練期間 prosody 檔案被覆寫，仍會回報錯誤。

`config.json` 的 `dataset_counts` 記錄初始化篩選後的筆數；`duration_filter.json`
保存初始化時略過的 ID 與理由。`skipped_samples.jsonl` 記錄所有略過項目的
`split`、`epoch`、`utt_id`、`reason`（`epoch=0` 表示初始化篩選）。
`metrics.jsonl` 的 `train/samples`、`val/samples` 是該 epoch 實際處理的筆數，
`train/skipped_samples`、`val/skipped_samples` 則是讀取及組 batch 時略過的筆數。
loss、accuracy 與 EER 以成功載入的樣本計算；有效評估集合可能小於原始 protocol。
整個 epoch 無有效樣本，或 dev 剩下單一類別而無法計算 EER 時，仍會停止並回報原因。

兩階段模型的 `pros_ln` 與 `final_proj` 都會使用判定後的維度：
speaker 固定為 192 維，因此 prosody 為 128／256 維時，投影輸出分別為 320／448 維。
啟動時會印出 `Prosody dim: ...`，W&B config 也會記錄實際的 `prosody_dim`。
直接在 Python 建立模型時，請傳入 `prosody_dim=dataset.prosody_dim`；
模型建構子的預設值仍為 128。

Stage 2 使用的 Stage 1 checkpoint 必須與 Stage 2 資料維度一致。
`main_eval.py` 會從 checkpoint 權重判定維度，評估時不需要另外指定。

### Rhythm Stage 2：完整語音訓練

`main_stage2realfake_rhythm.py` 預設保留整段音訊與全部音節，不再裁成四秒。
XLS-R 的 frame 數與音節數皆可變；同一個 batch 只在右側補零，attention、
SSL anchors 與負樣本取樣都排除 padding。音節 duration、deviation、相鄰差異
與 nPVI 使用完整音檔的音節重新計算，cross-attention 融合方式維持不變。

完整語音必須搭配重新抽取的 prosody targets，不能沿用原本四秒／200-frame cache。
以下以 train 為例；dev 使用相同 teacher 與設定另行抽取：

```bash
uv run --locked python extract_full_prosody.py \
  --protocol_txt dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --audio_dir dataset/ASVspoof2019/ASVspoof2019_LA_train/flac \
  --out_txt prosody_full_txt/asvspoof2019_train_prosody.txt \
  --utt_col 1 --ext .flac --layer 7
```

VAD teacher 加上 `--teacher_kind vad --teacher_checkpoint <實際 checkpoint 目錄>`；
該目錄必須包含 `model_config.yml` 與 `pytorch_model.bin`，並與 Stage 1 的 teacher 選擇相符。
抽取工具沿用 teacher 原生的六秒窗口，逐段處理完整音檔（含最後不足六秒的尾段），
再以實際時間戳插值到 XLS-R CNN receptive field 的中心。Teacher 的每段上下文有限，
但沒有丟棄任何音訊區段；Stage 2 的 XLS-R 與融合模型仍一次接收完整語音。

工具會同時寫入 `<prosody.txt>.meta.json`，記錄完整音長、CNN frame 幾何與 teacher 設定。
訓練會驗證此 metadata、target frame 數及實際音長；缺少 metadata 的舊 cache 會直接報錯，
不會自動拉長四秒 target。`--skip_bad_samples` 也不會隱藏音長／target 對應錯誤。

`train_rhythm.sh` 的 Step 0 以註解列出完整語音 train/dev targets 的抽取指令，
需手動單獨執行以準備 `prosody_full_txt/`。腳本不會自動抽取。
準備完成後，執行 `bash train_rhythm.sh` 進行訓練與評估，輸出至
`output/logs_stage2realfake_rhythm_syllable_full/` 與 `output/eval_stage2_best_rhythm_syllable_full/`；
`EVAL_ONLY=1` 可只評估，亦可透過原有的 `EVAL_CKPT`／`EVAL_SCORE_DIR` 指定輸入輸出。
VAD 版本使用 `train_rhythm_vad.sh`，可用 `VAD_TEACHER_CHECKPOINT` 指定 teacher checkpoint。

- 訓練預設 `--batch_size 32 --num_workers 8`，對齊原先 baseline 的實際 Stage 2 run。
  訓練隨機打亂，dev 不打亂；最後一批保留，略過無效資料時實際筆數可能較少。
- 訓練 `--max_batch_samples` 預設 `0`，停用音長預算，維持固定 batch 大小。
  完整語音比四秒裁切需要更多顯存；不會自動縮小 batch 或裁切語音。
  明確指定正值才啟用依音長分組的可變 batch，例如 `640000` 是每批補零後合計 40 秒的樣本數預算。
  單一超長音檔仍完整保留並獨立成批；此預算不是顯存上限。
  可變 batch 會改變每次更新的樣本數與跨 speaker 負樣本候選，不能視為和 baseline 相同的訓練設定。
- eval 預設 `--batch_size 16 --max_batch_samples 640000 --num_workers 4`，依音長分組，
  分數以 utterance ID 對應；這些是推論設定。
- `beta` 維持 baseline 排程：epoch 1–4 為 `0.2`，之後為 `0.05`；明確指定 `--beta` 才使用固定值。
- 新訓練的 Rhythm/fusion `dropout` 為 `0.1`，對齊 baseline 分類器的機率；兩種架構套用 dropout 的位置與次數不同。
- 新訓練省略 `--T_target`。數值型 `--T_target` 僅供重現舊四秒流程；eval 依 checkpoint 的
  `config.json` 自動選擇完整語音或舊四秒模式，不會改變既有 checkpoint 的評估語意。

### 使用 W&B 紀錄訓練

Stage 1 與 Stage 2 使用 Weights & Biases（W&B）保存原本由
TensorBoardX 紀錄的指標。每次執行會建立獨立 run，保存命令列訓練參數，
並以 `job_type` 的 `stage1`／`stage2` 區分階段。所有指標每個 epoch
寫入一次，圖表橫軸為從 1 開始的 `epoch`，指標名稱維持不變：

- Stage 1：`loss/train_contrastive`、`loss/val_contrastive`。
- Stage 2：`loss/train_total`、`loss/train_ssl`、`loss/train_cls`、
  `loss/val_total`、`loss/val_ssl`、`loss/val_cls`、`cos/train_spk`、
  `cos/train_pros`、`cos/val_spk`、`cos/val_pros`、`acc/val`、
  `acc/val_bonafide`、`acc/val_spoof`、`acc/val_balanced`。

`wandb` 已包含在 `uv` 依賴中；依上述步驟建立環境後可執行：

```bash
source .venv/bin/activate
wandb login
export WANDB_PROJECT='your-project'
export WANDB_ENTITY='your-team-or-username'
# 選用：指定本次 run 的顯示名稱
export WANDB_NAME='stage1-experiment'
```

可在訓練命令末尾以 `--wandb_tags` 指定一或多個標籤，例如
`--wandb_tags prosdd stage1 baseline`。此選項適用於 Stage 1、一般 Stage 2
與 Rhythm Stage 2。

接著以原本的資料路徑與訓練參數執行 `main_stage1real.py` 或
`main_stage2realfake.py`，預設會同步至 W&B。無互動環境可透過
`WANDB_API_KEY` 提供憑證；請勿將 API key 寫入程式或版本控制。
`WANDB_ENTITY` 可省略以使用帳號預設位置；建議明確設定 `WANDB_PROJECT`。

`--log_dir` 的 checkpoint 儲存規則如下：

- Stage 1：只在最後一個 epoch 完成後儲存 `model_last.pth`。
- Stage 2（一般版與 Rhythm 版）：只保存最低 val loss 的 `model_best.pth`；
  只有 loss 嚴格下降時才覆寫，平手或退步時不存檔。

兩個階段都不再產生逐 epoch checkpoint。
Stage 2 的 val loss 是當前 epoch 的 `alpha * cls_loss + beta * ssl_loss`，
其中 `beta` 仍沿用既有排程。已存在的舊 checkpoint 不會自動刪除。
此目錄也作為 W&B 本機紀錄的根目錄（其下的 `wandb/`）。模型 checkpoint 不會自動上傳。
run 會在訓練正常結束或拋出例外時關閉。

無網路時可先在本機保存數據，再於恢復連線後上傳：

```bash
export WANDB_MODE=offline
# 執行原本的 Stage 1 或 Stage 2 訓練命令
# 訓練完成並恢復連線後：
unset WANDB_MODE
wandb sync --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
  /path/to/log_dir/wandb/offline-run-<timestamp>-<run_id>
```

此變更適用於後續訓練；既有 TensorBoard event 檔不會自動匯入。
W&B 的初始化、環境變數與同步方式可參考
[Python SDK 文件](https://docs.wandb.ai/models/ref/python/functions/init)、
[環境變數文件](https://docs.wandb.ai/models/track/environment-variables)與
[sync 指令文件](https://docs.wandb.ai/models/ref/cli/wandb-sync)。

### 多顯卡運算（Stage 1／Stage 2／評估）

三個入口 `main_stage1real.py`、`main_stage2realfake.py`、`main_eval.py`
統一使用 `CUDA_VISIBLE_DEVICES` 指定顯卡，不需要額外的命令列 GPU 參數。
請在啟動 Python 程序前設定環境變數：

- 未設定：預設只使用第一張可用 GPU。
- `CUDA_VISIBLE_DEVICES=2`：只使用實體 GPU 2。
- `CUDA_VISIBLE_DEVICES=2,3`：使用實體 GPU 2、3，啟用模型平行。
- `CUDA_VISIBLE_DEVICES=0,1,2,3`：使用四張 GPU。
- 沒有可用 CUDA GPU（包含設為空字串或 `-1`）：訓練使用 CPU；評估需要 CUDA，會回報錯誤。

使用一般 `python` 單一程序啟動，不需使用 `torchrun`。
如果執行環境已預先設定 `CUDA_VISIBLE_DEVICES`，程式會使用其中所有可見 GPU；
只想用單卡時，請在啟動命令明確指定一張卡。

例如，Stage 1 的完整命令範本（請替換資料路徑）：

```bash
CUDA_VISIBLE_DEVICES=2,3 uv run --locked python main_stage1real.py \
  --train_prosody_txt /data/train_prosody.txt \
  --dev_prosody_txt /data/dev_prosody.txt \
  --train_spkmean_txt /data/train_spkmean.txt \
  --dev_spkmean_txt /data/dev_spkmean.txt \
  --wav_dir_train /data/train_audio \
  --wav_dir_dev /data/dev_audio \
  --batch_size 64
```

CUDA 會將選定的 GPU 重新編號為程序內的 `cuda:0`、`cuda:1` 等。
例如 `CUDA_VISIBLE_DEVICES=2,3` 對應實體 GPU 2、3，程式顯示為 `cuda:0`、`cuda:1`。
第一個指定的 GPU 為主裝置，負責特徵擷取、投影、遮罩、損失與分類頭；
Transformer 層則依序平均分配到指定的 GPU。每層輸出會回到主裝置，
讓原本 encoder 的 LayerDrop 與最終 LayerNorm 流程保持相容。

此實作採用模型平行，完整 batch 會依序通過各層。
`--batch_size` 仍是每次 optimizer 更新的完整 batch 大小，不需乘上顯卡數。
跨說話者負樣本候選集合、遮罩與負樣本抽樣規則、損失權重、
學習率分組、凍結排程、資料順序、W&B 紀錄及 checkpoint 欄位名稱均維持原有設計。
舊 checkpoint 可以直接用於多卡評估，多卡訓練輸出的 checkpoint
也可用於原本單卡流程。

這種配置分散 Transformer 參數、梯度、optimizer 狀態與部分 activation
的記憶體需求，但各層仍依序運算，且有跨卡傳輸成本，**不保證加速或平均分配顯存**。
主裝置仍需容納完整 batch 的特徵與對比損失，增加顯卡不保證能消除所有 OOM。
訓練中的 dropout 使用各裝置的隨機數產生器，加上浮點運算差異，
即使 seed 相同，多卡與單卡也不保證逐位元相同或產生完全相同的訓練軌跡；
這裡保留的是模型與訓練演算法的邏輯。
韻律特徵擷取腳本 `extract_Prosody.py` 不適用此參數。

驗證方式（使用小型隨機初始化 Wav2Vec2，不下載預訓練權重）：

```bash
uv run --locked python -m unittest discover -s tests -v
```

測試涵蓋兩個訓練階段、兩種分類 pooling、輸出與梯度、一次 AdamW 更新、
LayerDrop、分類頭凍結、評估入口及 checkpoint 嚴格載入。
CPU 測試驗證裝置轉移 hook 不改變計算；實際跨 GPU 的輸出、反向傳播與更新測試
需要至少兩張可用 CUDA 顯卡，否則會標示為 skipped。
