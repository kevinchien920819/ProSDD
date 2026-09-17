# evaluation_metric

此目錄的 `__init__.py`、`calculate_metrics.py`、`calculate_modules.py`、
`a_dcf.py`、`util.py` 移植自本機
`/home/icebird/01_proj/rhythm-transformer/src/evaluation_metric`。
來源儲存庫 commit：`7524506746dddc606466c8adf7cae378c0997888`，
複製時這五個檔案沒有未提交變更。`a_dcf.py` 原有的 MIT 授權與作者資訊完整保留。

移植時保留來源的 EER、DCF 定義與函式介面，只做兩項調整：

- Cllr 使用數學等價的 `np.logaddexp(0, -lodds)`，避免大幅度分數讓 `exp` 溢位。
- 結果文字改由 Python 讀取並印出，支援含空白的檔案路徑。

`prosdd.py` 與 `__main__.py` 是 ProSDD 的銜接入口，提供分數／protocol
載入、依 utterance ID 對齊、輸入驗證與 JSON 報告。

## 使用方式

在專案根目錄執行：

```bash
uv run --locked python -m evaluation_metric \
  --score_path output/eval_stage2_epoch50/asvspoof2019_la_eval.txt \
  --protocol_path dataset/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
  --save_metrics_to output/eval_stage2_epoch50/asvspoof2019_la_eval.metrics.json
```

省略 `--save_metrics_to` 時，只印出四項指標。此入口不需要 GPU、音訊或模型權重。

支援的分數檔是 `utterance_id score` 兩欄，空白／Tab 分隔；也接受
`filename cm-score` 或 `trial_anon cm-score` 標頭。
支援的標籤檔如下（欄位位置由 1 起算）：

| 格式 | 音檔 ID | bonafide／spoof 標籤 |
|---|---|---|
| ASVspoof 2019，無標頭、5 欄 | 第 2 欄 | 第 5 欄 |
| ASVspoof5 Track 1，無標頭、10 欄 | 第 2 欄 | 第 9 欄 |
| 兩欄 key 檔，無標頭 | 第 1 欄 | 第 2 欄 |
| 有標頭的 TSV | `filename` 或 `trial_anon` | `cm-label` 或 `cm_label` |

分數檔與 protocol 的 ID 集合必須完全相同，檔案順序可以不同。
重複 ID、缺漏分數、未知標籤、NaN／Inf、空檔與只有單一類別的資料都會回報錯誤。

## 指標與分數語意

- `eer` 是 0–1 比例；`eer_percent` 是百分比；`cllr` 單位為 bits。
- `min_dcf` 與 `act_dcf` 使用來源的 CM cost model：
  `Pspoof=0.05`、`Cmiss=1`、`Cfa=10`。即使輸入 ASVspoof 2019 protocol，
  這裡仍計算這套 CM DCF，不代表 ASVspoof 2019 的 min t-DCF。
- 分數方向是越大越支持 bonafide。評估器直接使用輸入分數，不會自動翻轉、
  softmax 或校正。Cllr／actDCF 將輸入視為 log-likelihood ratio（LLR）。
- ProSDD 的 `main_eval.py` 目前輸出 `logits[:, 1]`，是 bonafide 原始 logit；
  因此其 Cllr／actDCF 反映這組未校正分數，不能視為已完成 LLR 校正的結果。
- 原始 a-DCF、t-DCF、t-EER 函式也已移植；它們需要額外的 ASV／SASV 分數、
  標籤或 ASV 錯誤率，ProSDD 的 CM 分數入口只計算上述四項指標。
