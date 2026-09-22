# 代理工作指引

所有面向使用者的回覆與報告均使用繁體中文（臺灣，`zh-TW`）。程式識別字、指令、路徑與技能辨識用的固定名稱保留原文。

## Agent skills

### Issue tracker

議題與產品需求文件（PRD）使用 `ProSDD/codes` 的 GitHub Issues 管理，透過 `gh` CLI 操作。詳見 `docs/agents/issue-tracker.md`。

### Triage labels

採用五個標準標籤：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。詳見 `docs/agents/triage-labels.md`。

### 領域文件

採用單一領域配置（`single-context`）：根目錄的 `CONTEXT.md` 與 `docs/adr/`。探索程式碼前，依 [領域文件設定](docs/agents/domain.md) 閱讀相關術語與決策；文件尚未存在時照常繼續。

# 程式碼風格

1. 不要重複造輪子，同樣的功能寫在單一function 進行引用，
2. 不要在function 內寫過多檢查，轉為在測式程式內進行測試，
3. 少量中文註解 大型或是主要的function需要有doc 說明 功能、定義輸入、輸出參數。
4. 變數命名簡短但需要有意義

# 安全規範

## 絕對禁止
- 在程式碼中寫死 API key、密碼、token

## 必須遵守
- 所有敏感資訊從環境變數讀取（.env）


# 測試規範
## 測試涵蓋要求
- TDD 方式開發
- 所有新功能都要有對應測試
- API 路由要有整合測試
- 工具函數要有單元測試
- 最低測試覆蓋率：70%

## 測試原則
- 每個測試只驗證一件事
- 使用描述性的測試名稱
- 測試要能獨立執行

# 特殊規則

## Git Commit 規範
遵循 Conventional Commits：
- `feat:` 新功能
- `fix:` 修復 bug
- `refactor:` 重構
- `test:` 測試相關
- `docs:` 文件更新
