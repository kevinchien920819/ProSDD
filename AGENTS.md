# 代理工作指引

所有面向使用者的回覆與報告均使用繁體中文（臺灣，`zh-TW`）。程式識別字、指令、路徑與技能辨識用的固定名稱保留原文。

## Agent skills

### Issue tracker

議題與產品需求文件（PRD）使用 `ProSDD/codes` 的 GitHub Issues 管理，透過 `gh` CLI 操作。詳見 `docs/agents/issue-tracker.md`。

### Triage labels

採用五個標準標籤：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。詳見 `docs/agents/triage-labels.md`。

### Domain docs

採用單一領域（`single-context`）配置：根目錄 `CONTEXT.md` 與 `docs/adr/`。詳見 `docs/agents/domain.md`。
