# BFCL v3 — Multi-Turn（範例資料）

> **尚未支援評測**：multi-turn 格式與 v1/v2 有根本性差異，需要有狀態的多輪對話模擬，
> 目前 twinkle-eval 尚未實作此路徑。

## 格式差異

| 欄位 | v1/v2 | v3 (multi-turn) |
|------|-------|-----------------|
| `question` | `[[turn0_messages]]` | `[[turn0], [turn1], ...]` |
| `function` | function 定義列表 | 無（改用 `involved_classes`）|
| `initial_config` | 無 | 系統初始狀態（dict）|
| `path` | 無 | ground truth 執行路徑（步驟列表）|

## 追蹤 Issue

BFCL v3 multi-turn 評測將在 Milestone #8 實作。
