# Stage 2 Rhythm 流程檢驗

`main_stage2realfake_rhythm.py` 現在可以直接執行完整訓練：載入 teacher 與
Stage 1 權重、建立 train/dev 資料流程、逐輪最佳化聯合 loss、驗證及儲存結果。
以下比較工作目錄中的目前版本，將差異分成八類。

| 類別 | `main_stage2realfake.py` | `main_stage2realfake_rhythm.py` |
| --- | --- | --- |
| 1. 啟動與設定 | 在 `__main__` 解析參數、設定 seed、選 device | 可透過 `main(argv)` 或 CLI 啟動，使用相同 seed/device 工具；新增 teacher 設定及 JSON 設定紀錄 |
| 2. 資料與 batch | 置中裁切／補零、讀取離線 prosody，回傳五項 tuple | 先選停頓 crop、短音訊置中補零、每次從乾淨音訊重新抽 prosody，之後才套用 RawBoost；dict batch 含 rhythm 與兩種 padding mask，可用長度預算分批 |
| 3. 模型與 criterion | 一般 Stage 2，Stage 1 選填，分類權重 `[0.1, 0.9]` | Rhythm 模型要求 Stage 1，還原 backbone、mask、projection、prosody LayerNorm；teacher 維度與 checkpoint 必須一致；分類權重相同 |
| 4. Optimizer | backbone、SSL head、classifier 三組 AdamW learning rate | 增加獨立 rhythm/fusion 組，共四組 learning rate |
| 5. Epoch 與凍結 | `train_epoch`／`validate` 分開；`epoch < freeze_epochs` 時凍結分類器，只最佳化 SSL | 共用 `run_epoch`；train 更新權重，dev 關閉梯度；從第一輪使用分類＋SSL loss，沒有凍結期 |
| 6. SSL loss 權重 | 前四輪 0.2、之後 0.05；目前實際忽略 `--beta` | 相同預設排程，並允許 `--beta` 固定覆寫；同輪 train/dev 使用相同 beta |
| 7. 指標與異常 | train 有 loss/cosine；dev 另算 accuracy、各類別及 balanced accuracy | train 也有 accuracy；dev 增加 EER；本機另存 batch 數，並記錄處理／略過樣本數；拒絕非有限 loss、空 epoch 或缺少某一類的 dev |
| 8. 紀錄與 checkpoint | W&B 與逐輪模型權重 | 沿用 W&B 的 loss/cos/acc 命名，另存設定、JSONL 指標、排除 ID／原因、逐輪權重及最低 dev EER 的 `model_best.pth` |

## 會影響實驗比較的設定

| 參數 | Baseline 預設 | Rhythm 預設 |
| --- | --- | --- |
| `batch_size` | 64 | 32 |
| `mask_prob` | 0.15 | 0.25 |
| `tau` | 0.1 | 0.07 |
| `num_workers` | 4 | 0；正數時使用 `spawn`、常駐 worker 與共用 CPU teacher 權重 |
| `audio_seconds` | 4；固定置中裁切 | 4；停頓邊界裁切，0 表示整句 |
| `T_target` | 依秒數推算，四秒 200 | `None`，使用實際 CNN 輸出長度 |
| `stage1_ckpt` | 選填 | 必填 |
| `rhythm_sources` | 無 | `syllable`；同一設定傳給 Dataset 與模型 |

標準 XLS-R 的四秒音訊有 199 個 CNN frames。Rhythm extractor 依 CNN 時間中心
產生對齊 targets，collate assert 每筆 frame 數一致；若明確指定 `T_target=200`，
則補一列零並遮罩。這與 baseline 的四秒／200-frame 慣例不同。

Train 與 dev 的每次取樣都重新選 crop、重新計算 targets；dev 不做 RawBoost。
因此 crop 模式的 dev EER 也受當輪取樣影響。整句模式可用 `--audio_seconds 0`。
每個 epoch 的指標只涵蓋成功處理的樣本；初始化排除另記於 `duration_filter.json`
與 `skipped_samples.jsonl` 的 epoch 0。

Teacher 在主程序載入一次並凍結，資料集共享該模型；多 worker 共享 CPU 權重。
`--prosody_txt_train`／`--prosody_txt_dev` 已移除，改用 `--teacher_kind`、
`--teacher_checkpoint`、`--prosody_layer`。Layer 與 teacher 來源必須和 Stage 1 一致；
程式會檢查維度，無法僅從原始 `state_dict` 識別當初 teacher 的來源。

## 驗證結果

- 全專案 125 項測試通過，其中獨立評估包含 17 項測試。
- 本次獨立評估入口覆蓋率 92%、共用 `data_utils_rhythm.py` 97%，兩個模組合計 95%。
  評估測試包含本機小型 Wav2Vec2 checkpoint 的真正 CLI 執行、CPU worker、分數與指標輸出，
  以及裁切重現性、動態／指定 frame 數、排除樣本紀錄與輸出檔保護。
- 訓練串接時，`main_stage2realfake_rhythm.py` 覆蓋率 96%、`data_utils_rhythm.py` 97%、
  `extract_Prosody_rhythm.py` 94%、`extract_full_prosody.py` 79%；四個模組合計 93%。
- 端到端測試使用真實 Dataset、collate、Wav2Vec2、Rhythm 模型、optimizer 與檔案輸出；
  外部 pretrained 下載與 teacher 以小型替身取代。涵蓋 SSL backward、checkpoint 重載、
  最佳 EER 選模、固定 beta、RawBoost、crop/frame override、長度預算、失敗 batch 及多 worker。
- 額外以真正的命令列入口執行兩個 epoch：真實 MPM teacher（256 維、layer 7）、
  ASVspoof dev 的 `LA_D_1008730`／`LA_D_1047731`、一秒目標停頓 crop、
  一個 CPU worker，以及保存成 Hugging Face 格式的小型 Wav2Vec2／Stage 1 權重。
  此次 CLI 執行未使用 mock，每輪 train/dev 各處理兩筆，沒有略過樣本。
  SSL loss 第一輪 train/dev 約 1.2866／1.2020，第二輪約 1.1662／0.9726，均為有限值。
  W&B offline 正常結束，逐輪 checkpoint、最佳權重及 JSON 紀錄均成功寫入。
- 上述小型驗證確認流程可執行；尚未啟動完整 XLS-R 300M 的全資料集訓練，也未評估分類品質。

## 命令與範圍

正式訓練參數與 ASVspoof2019 LA 完整命令見 [README 的 Rhythm 完整訓練](../README.md#rhythm-完整訓練)。
在專案根目錄執行測試：

```bash
uv run --locked python -m unittest discover -s tests
uv run --locked python -m coverage run \
  --source=main__eval_rhythm,main_stage2realfake_rhythm,data_utils_rhythm,extract_Prosody_rhythm,extract_full_prosody \
  -m unittest discover -s tests
uv run --locked python -m coverage report --fail-under=70
```

獨立 eval 入口 `main__eval_rhythm.py` 已串接相同的停頓裁切、音訊與 rhythm padding，
由訓練設定還原音訊模式與 frame 數，只執行分類推論。每個 ID 的裁切由 seed 固定，
不需要 SSL targets。輸出分數、成功評分 protocol、coverage 與選用的 CM 指標；
用法見 [Rhythm 獨立評估](../README.md#rhythm-獨立評估)。完整訓練可直接執行
`main_stage2realfake_rhythm.py`，或執行 `bash train_rhythm.sh` 使用標準 ASVspoof2019
LA 設定；兩者每輪都包含 dev 驗證。
`train_rhythm.sh` 訓練成功後，會用同一輸出目錄的 `model_best.pth` 與 `config.json`
評估 ASVspoof2019 LA eval，評估產物存於該目錄的 `eval/`；任一步驟失敗即結束腳本。
Checkpoint 保存 `state_dict`，沿用 baseline 格式，不包含 optimizer 續訓狀態。
