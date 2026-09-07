# 議題追蹤：GitHub

本儲存庫的議題與產品需求文件（PRD）存放於 [ProSDD/codes 的 GitHub Issues](https://github.com/ProSDD/codes/issues)。所有操作使用 `gh` CLI。

## 操作慣例

在本儲存庫中執行時，`gh` 會依 Git 遠端判定儲存庫；必要時明確指定 `--repo ProSDD/codes`。

- 建立議題：`gh issue create --title "..." --body-file <檔案路徑>`。多行內容先寫入暫存檔，保留實際換行。
- 讀取議題及留言：`gh issue view <number> --comments`；需要結構化資料與標籤時，使用 `gh issue view <number> --json number,title,body,labels,comments`，再以 `jq` 篩選。
- 列出議題：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`；依需求調整 `--label` 與 `--state`。
- 新增留言：`gh issue comment <number> --body-file <檔案路徑>`。
- 新增或移除標籤：`gh issue edit <number> --add-label "..."` 或 `--remove-label "..."`。
- 關閉議題：`gh issue close <number> --comment "..."`。

## Pull request 作為需求入口

**PRs as a request surface: no.**

若日後將此值改為 `yes`，外部 PR 會使用與議題相同的分流標籤與狀態：

- 讀取 PR：`gh pr view <number> --comments`，並以 `gh pr diff <number>` 讀取差異。
- 列出待分流的外部 PR：`gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`，保留 `authorAssociation` 為 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR` 或 `NONE` 的項目，排除 `OWNER`、`MEMBER` 與 `COLLABORATOR`。
- 留言、標籤與關閉操作分別使用 `gh pr comment`、`gh pr edit --add-label`／`--remove-label` 與 `gh pr close`。

GitHub 的議題與 PR 共用編號。若只有 `#42` 這類編號，先用 `gh pr view 42` 判定，必要時再用 `gh issue view 42`。

## 技能指令的對應操作

技能要求「發布至議題追蹤系統」時，建立 GitHub issue。要求「取得相關工作單」時，執行 `gh issue view <number> --comments`。

## Wayfinder 操作

`/wayfinder` 使用一個總覽議題（map）與多個子議題管理探索工作。

- 總覽議題：使用 `wayfinder:map` 標籤，內文保留筆記（Notes）、目前決策（Decisions-so-far）與待釐清事項（Fog）。以 `gh issue create --label wayfinder:map` 建立。
- 子工作單：使用 `gh api` 的 sub-issues 端點連結至總覽議題。若不支援子議題，改在總覽內文使用工作清單，並在子議題頂端寫入 `Part of #<map>`。標籤為 `wayfinder:<type>`，類型使用 `research`、`prototype`、`grilling` 或 `task`；認領後指派給負責開發者。
- 阻擋關係：優先使用 GitHub 原生議題相依關係。以 `gh api --method POST repos/ProSDD/codes/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>` 新增。資料庫 ID 透過 `gh api repos/ProSDD/codes/issues/<n> --jq .id` 取得，不使用議題編號或 `node_id`。`issue_dependencies_summary.blocked_by` 代表尚未關閉的阻擋項目數。若不支援此功能，在子議題頂端寫入 `Blocked by: #<n>, #<n>`；所有阻擋議題關閉後才能開始。
- 下一個可處理項目：列出總覽所屬的未關閉子議題，排除仍被阻擋或已有負責人的項目，再依總覽順序選取第一個。
- 認領：在工作階段首次寫入時，執行 `gh issue edit <n> --add-assignee @me`。
- 完成：先以 `gh issue comment <n> --body-file <檔案路徑>` 留下結果，再以 `gh issue close <n>` 關閉，最後將摘要與連結加入總覽的目前決策區。
