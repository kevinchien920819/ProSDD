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

接著以原本的資料路徑與訓練參數執行 `main_stage1real.py` 或
`main_stage2realfake.py`，預設會同步至 W&B。無互動環境可透過
`WANDB_API_KEY` 提供憑證；請勿將 API key 寫入程式或版本控制。
`WANDB_ENTITY` 可省略以使用帳號預設位置；建議明確設定 `WANDB_PROJECT`。

`--log_dir` 繼續保存每個 epoch 的 `model_epoch_<epoch>.pth`，並作為
W&B 本機紀錄的根目錄（其下的 `wandb/`）。模型 checkpoint 不會自動上傳。
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
