# BFCL v1 — 範例資料集

來源：[gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)

| 檔案 | Category | 筆數 | 說明 |
|------|----------|------|------|
| `simple.jsonl` | simple | 5 | 單一 function call，單一 function 定義 |
| `parallel.jsonl` | parallel | 5 | 多個 function call，同一個 function 定義 |
| `multiple.jsonl` | multiple | 5 | 單一 function call，多個 function 定義（需選對） |

## 使用方式

Config 設定：
```yaml
evaluation:
  dataset_paths:
    - "datasets/example/bfcl_v1/"
  evaluation_method: "bfcl_fc"   # 或 "bfcl_prompt"
```

## 格式說明

每行 JSON 包含：
- `id`: 題目 ID（如 `simple_0`）
- `question`: JSON string，messages 列表（`[{"role": "user", "content": "..."}]`）
- `functions`: JSON string，function 定義列表
- `answer`: JSON string，`{"category": "simple", "ground_truth": [...]}`

原始資料由 `twinkle_eval.datasets.bfcl.merge_bfcl_directory()` 轉換產生。
