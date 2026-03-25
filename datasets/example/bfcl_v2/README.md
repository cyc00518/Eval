# BFCL v2 — Live 範例資料集

來源：[gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)

| 檔案 | Category | 筆數 | 說明 |
|------|----------|------|------|
| `live_simple.jsonl` | live_simple | 5 | 真實世界 API 的單一 function call |

Live 類別使用真實 API function 定義（非合成），難度更接近實際使用情境。

## 使用方式

Config 設定：
```yaml
evaluation:
  dataset_paths:
    - "datasets/example/bfcl_v2/"
  evaluation_method: "bfcl_fc"   # 或 "bfcl_prompt"
```

## 格式說明

格式與 v1 相同（`id`、`question`、`functions`、`answer`），由
`twinkle_eval.datasets.bfcl.merge_bfcl_directory()` 轉換產生。
