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

兩階段模型的 `pros_ln` 與 `final_proj` 都會使用判定後的維度：
speaker 固定為 192 維，因此 prosody 為 128／256 維時，投影輸出分別為 320／448 維。
啟動時會印出 `Prosody dim: ...`，W&B config 也會記錄實際的 `prosody_dim`。
直接在 Python 建立模型時，請傳入 `prosody_dim=dataset.prosody_dim`；
模型建構子的預設值仍為 128。

Stage 2 使用的 Stage 1 checkpoint 必須與 Stage 2 資料維度一致。
`main_eval.py` 會從 checkpoint 權重判定維度，評估時不需要另外指定。

### Rhythm 資料介面（crop 後即時抽取）

可設定目標秒數並對齊停頓的新模組為
[`data_utils_rhythm.py`](data_utils_rhythm.py)。使用 `max_len=0` 保留完整音訊，
或指定正秒數，每次 `dataset[idx]` 重新選取附近 word 之間的停頓作為起訖點。
CSV 在初始化時讀取一次，`select_duration` 回傳片段特徵與起訖秒數，
再由 `load_audio` 讀取該區間。短音訊置中補零後，呼叫
[`extract_Prosody_rhythm.py`](extract_Prosody_rhythm.py) 的 `extract_prosody_rhythm`
重新抽取該片段的 prosody，最後才對有效音訊套用 RawBoost。
Teacher 接收乾淨音訊，權重共用且固定為 CPU 推論模式；每次取樣都重新計算 targets。
`rhythm_sources` 選取層級；
特徵只使用片段內完整音節的 duration，並重新計算 `devi_mu`、`mu_diff`。

Dataset 建構子以 `prosody_model` 接收已載入的 MPM／VAD teacher，取代 `prosody_txt`。
特徵維度從 teacher 判定；Dataset、collate 與學生模型須使用相同的 CNN kernel／stride。
單筆仍回傳 `wav, spk_emb, pros_emb, spk_idx, label, duration_features, valid_samples, utt_id`。

```python
from functools import partial
from torch.utils.data import DataLoader
from data_utils_rhythm import ProSDDStage2RhythmDataset, collate_stage2_rhythm
from extract_full_prosody import load_prosody_teacher

teacher = load_prosody_teacher()  # MPM；VAD 使用 load_prosody_teacher("vad", checkpoint)
dataset = ProSDDStage2RhythmDataset(
    utt_ids, spk_ids, labels, wav_dir, spkmean_txt, teacher, duration_csv,
    max_len=4.0,
)
loader = DataLoader(
    dataset, batch_size=2, num_workers=0,
    collate_fn=partial(collate_stage2_rhythm, T_target=None),
)
```

預設 CNN 為 XLS-R。擷取器使用實際時間中心產生 targets：16 kHz、四秒音訊為
199 個有效 CNN frames。`collate_stage2_rhythm` 會 assert 單筆 prosody 與音訊的
CNN frame 數相同，再統一 batch 的長度；`T_target=None` 保留所有 frame，
指定數值則配合模型截斷或補零。音訊與 prosody 補零，rhythm 補 `-100`，
兩種 mask 的 `True` 表示人工 padding。Prosody 在人工 padding 的位置清零，
真實靜音的有效性由音訊範圍判定。`None` 樣本預設略過，整批無有效樣本回傳
`{"skipped_samples": []}`；`skip_bad_samples=False` 遇到 `None` 會報錯。
訓練入口另外開啟 `report_bad_samples=True`，讓 Dataset 將失敗的 ID／原因交給
collate 的 `skipped_samples`，再由主程序統一寫入紀錄，支援多 worker。

小型抽取驗證（本機已有 MPM 與 XLS-R 快取）：

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --locked python extract_Prosody_rhythm.py \
  --audio_path dataset/ASVspoof2019/ASVspoof2019_LA_dev/flac/LA_D_1047731.flac \
  --start 0.6 --end 4.4
```

此例輸出 `samples=60800`、`prosody=(189, 256)`。加上 `--out_pt <新檔案.pt>`
可保存 tensor；已存在的檔案不會覆寫。VAD 使用 `--teacher_kind vad --teacher_checkpoint <目錄>`。
一般機器首次下載模型時省略 `HF_HUB_OFFLINE=1`。

### Rhythm 完整訓練

[`main_stage2realfake_rhythm.py`](main_stage2realfake_rhythm.py) 已串接 teacher、
Stage 1 checkpoint、train/dev Dataset、SSL／分類聯合訓練、驗證、W&B 與模型儲存。
不再接受 `--prosody_txt_train`／`--prosody_txt_dev`，也不需要先重抽完整 prosody 快取。
Teacher 的種類、checkpoint、layer 與特徵維度應和 Stage 1 targets 一致；
本機標準 MPM 與 `output/logs_stage1contrastived/model_epoch_50.pth` 都使用 256 維。

在專案根目錄執行 `bash train_rhythm.sh` 即可開始 ASVspoof2019 LA 訓練，成功後自動以
`model_best.pth` 執行 eval split 的獨立評估。可用 `RHYTHM_LOG_DIR=<路徑>` 指定訓練
輸出目錄；評估分數、指標、protocol 與 coverage 存於該目錄下的 `eval/`。
訓練使用 W&B `online` 模式，並附上 `prosdd`、`rhythm`、`stage2`、`asvspoof2019` tags。

腳本中的 Stage 2 訓練命令如下，可依需要調整參數：

```bash
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
  --audio_seconds 4 --epochs 50 --batch_size 8 --num_workers 8 \
  --lr_ssl_backbone 1e-6 --lr_ssl_head 1e-4 \
  --lr_rhythm 1e-5 --lr_cls 1e-5 \
  --weight_decay 1e-4 --seed 1234 \
  --rhythm_sources syllable --mask_prob 0.15 --tau 0.1 \
  --algo 3 --augment_prob 0.5 --skip_bad_samples \
  --wandb_mode online \
  --wandb_tags prosdd rhythm stage2 asvspoof2019 \
  --log_dir output/logs_stage2realfake_rhythm_crop
```

- `--audio_seconds 4` 每次重新選 crop，邊界移至停頓，實際長度可能超過四秒；
  `--audio_seconds 0` 保留整句。Train／dev 都重新抽取 targets，只有 train 套用 RawBoost。
- `--T_target` 預設不指定，保留實際 CNN frames；不會自動把四秒設成 200。
- `--num_workers 8` 使用 `spawn` 與常駐 CPU teacher workers，共用凍結的 teacher 權重，
  每次取樣仍重新推論；設為 `0` 則在主程序執行。
- `--max_batch_samples` 可設定 batch 補零後的 sample 預算，以整句長度作 crop 上界；
  單筆超出預算時獨立成批。預設 0 表示只依 `batch_size` 分批。
- 聯合 loss 為 `alpha * cls_loss + beta * ssl_loss`，分類權重 `[0.1, 0.9]`。
  預設前四輪 beta=0.2，第五輪起 0.05；`--beta` 可固定覆寫。沒有分類器凍結期。
- VAD 改用 `--teacher_kind vad --teacher_checkpoint <VAD 目錄>`，並提供相符的 Stage 1 checkpoint。
- W&B 可改成 `--wandb_mode offline`／`online`；省略時遵循 `WANDB_MODE`，未設定則為 online。

`log_dir` 保存 `config.json`、`metrics.jsonl`、`duration_filter.json`、
`skipped_samples.jsonl`、每輪 `model_epoch_<epoch>.pth`，以及最低 dev EER 的
`model_best.pth`（相同 EER 保留較早的一輪）。Checkpoint 格式為模型 `state_dict`，
不含 optimizer 續訓狀態。EER 在 JSON／W&B 中以 0–1 儲存。
初始化排除記為 epoch 0；各輪略過筆數只計當輪讀取失敗的樣本。
Dev 成功讀取的資料必須同時包含 bonafide 與 spoof，才能計算 EER。

獨立評估入口為 `main__eval_rhythm.py`；`train_rhythm.sh` 會依序執行 Stage 2 訓練與
eval 評估，`train_rhythm_vad.sh` 仍暫停使用。每輪訓練已包含 dev 驗證。
訓練失敗時腳本立即結束，不執行 eval；評估失敗也會回傳非零狀態。

與 baseline 訓練入口的八類流程差異及小型驗證結果，見
[Stage 2 Rhythm 流程檢驗](docs/stage2-rhythm-flow-review.md)。

### Rhythm 獨立評估

`main__eval_rhythm.py` 從 checkpoint 同目錄的 `config.json` 還原模型、
`rhythm_sources`、`audio_seconds` 與 `T_target`；也可用 `--config_path` 指定設定檔。
支援目前訓練產生的 `pause_crop` 與 `full_utterance` 模式，包括非四秒裁切及指定
frame 數。推論只需音訊與 duration CSV，不需要 speaker embeddings、prosody targets
或 MPM／VAD teacher。Protocol 必須含 bonafide／spoof 標籤。

```bash
uv run --locked python main__eval_rhythm.py \
  --list_path dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
  --wav_dir dataset/ASVspoof2019/ASVspoof2019_LA_eval/flac \
  --duration_csv dataset/ASVspoof2019/ASVspoof2019_LA_cache_csv/cache_ASVspoof2019.LA_eval.csv \
  --model_path output/logs_stage2realfake_rhythm_crop/model_best.pth \
  --save_scores_to output/eval_rhythm/asvspoof2019_la_eval.txt \
  --save_metrics_to output/eval_rhythm/asvspoof2019_la_eval.metrics.json \
  --batch_size 16 --max_batch_samples 640000 --num_workers 4 \
  --seed 1234 --skip_bad_samples
```

- 裁切、短錄音置中補零、rhythm 統計及 padding masks 與訓練共用處理函式。
  評估以 `seed` 和 utterance ID 固定裁切位置，不因 worker 數或 batch 排序改變；
  訓練的 dev 每輪仍重新取樣，因此兩者的 EER 不一定相同。
- 分數為 `bonafide logit - spoof logit`，越高越接近真實語音，與 Rhythm dev EER 一致。
  `--save_metrics_to` 選用，可輸出 EER、Cllr、minDCF、actDCF；分數未經校準。
- 同時輸出 `*.protocol.txt`（成功評分的 ID／標籤）及 `*.coverage.json`
  （涵蓋率、排除 ID／原因與評估設定）。`--skip_bad_samples` 允許略過缺漏或損壞樣本；
  指標只計算成功評分子集。全部失敗，或要求計算指標時僅剩一種類別，會回報錯誤。
- `--max_batch_samples 0` 只依 batch size 分批；正數以整句／補零長度上界控制預算，
  單筆超出預算時獨立成批。既有輸出檔不會覆寫，重跑時請使用新的輸出名稱。

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

### 單顯卡運算（Stage 1／Stage 2／評估）

Stage 1、一般與 Rhythm Stage 2 的訓練及評估皆將完整模型放在第一張
可見 GPU（`cuda:0`）。可在啟動 Python 前使用 `CUDA_VISIBLE_DEVICES`
選擇顯卡，例如 `CUDA_VISIBLE_DEVICES=2` 使用實體 GPU 2。
即使環境中有多張可見 GPU，也只會使用第一張。

沒有可用 CUDA GPU（包含設為空字串或 `-1`）時，訓練與 Rhythm 獨立評估使用 CPU；
一般評估入口 `main_eval.py` 仍需要 CUDA，否則會回報錯誤。

例如，Stage 1 的完整命令範本（請替換資料路徑）：

```bash
CUDA_VISIBLE_DEVICES=2 uv run --locked python main_stage1real.py \
  --train_prosody_txt /data/train_prosody.txt \
  --dev_prosody_txt /data/dev_prosody.txt \
  --train_spkmean_txt /data/train_spkmean.txt \
  --dev_spkmean_txt /data/dev_spkmean.txt \
  --wav_dir_train /data/train_audio \
  --wav_dir_dev /data/dev_audio \
  --batch_size 64
```

CUDA 會將選定的 GPU 編號為程序內的 `cuda:0`。
模型 checkpoint 的權重名稱與格式維持相容，既有 checkpoint 仍可直接載入。
