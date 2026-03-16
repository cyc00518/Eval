"""
建立 Twinkle Eval 範例評測資料集（用於除錯）

從 HuggingFace 下載各資料集的子集，存放於 datasets/example/ 目錄。

用法：
    python scripts/create_example_datasets.py

產出：
    datasets/example/gsm8k/test.jsonl          (20 題，數學)
    datasets/example/aime/test.jsonl           (20 題，數學競賽)
    datasets/example/tmmluplus/test.jsonl      (20 題，繁中選擇題)
    datasets/example/mmlu/test.jsonl           (20 題，英文選擇題)
    datasets/example/mmlu_pro/test.jsonl       (20 題，英文多選項選擇題)
"""

import json
import os
import re
import string

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "example")
N = 20  # 每個資料集取 N 題


def _index_to_label(idx: int) -> str:
    letters = []
    while True:
        idx, rem = divmod(idx, 26)
        letters.append(string.ascii_uppercase[rem])
        if idx == 0:
            break
        idx -= 1
    return "".join(reversed(letters))


def save_jsonl(path: str, records: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✅ 已儲存 {len(records)} 題 → {path}")


# ──────────────────────────────────────────────
# 1. GSM8K
# ──────────────────────────────────────────────
def create_gsm8k():
    print("\n[1/5] GSM8K (openai/gsm8k)...")
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    records = []
    for row in ds.select(range(N)):
        # 從 "...計算過程\n#### 42" 提取最終數字答案
        answer_raw = row["answer"]
        match = re.search(r"####\s*(.+)$", answer_raw, re.MULTILINE)
        answer = match.group(1).strip().replace(",", "") if match else answer_raw.strip()
        records.append({"question": row["question"], "answer": answer})
    save_jsonl(os.path.join(EXAMPLE_DIR, "gsm8k", "test.jsonl"), records)


# ──────────────────────────────────────────────
# 2. AIME 2025 (共 30 題，全取)
# ──────────────────────────────────────────────
def create_aime():
    print("\n[2/5] AIME 2025 (MathArena/aime_2025)...")
    from datasets import load_dataset

    ds = load_dataset("MathArena/aime_2025", split="train")
    records = []
    for row in ds:
        records.append({
            "question": row["problem"],
            "answer": str(row["answer"]).strip(),
        })
    save_jsonl(os.path.join(EXAMPLE_DIR, "aime2025", "test.jsonl"), records)


# ──────────────────────────────────────────────
# 3. TMMLU+ (取 economics 10 題 + basic_medical_science 10 題)
# ──────────────────────────────────────────────
def create_tmmluplus():
    print("\n[3/5] TMMLU+ (ikala/tmmluplus)...")
    from datasets import load_dataset

    subjects = [("economics", 10), ("basic_medical_science", 10)]
    records = []
    for subject, n in subjects:
        ds = load_dataset("ikala/tmmluplus", subject, split="test")
        for row in ds.select(range(min(n, len(ds)))):
            records.append({
                "question": row["question"],
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
                "answer": row["answer"].strip().upper(),
            })
    save_jsonl(os.path.join(EXAMPLE_DIR, "tmmluplus", "test.jsonl"), records)


# ──────────────────────────────────────────────
# 4. MMLU (取 high_school_mathematics 10 題 + computer_science 10 題)
#    choices-list 格式，_normalize_record 會自動處理
# ──────────────────────────────────────────────
def create_mmlu():
    print("\n[4/5] MMLU (cais/mmlu)...")
    from datasets import load_dataset

    subjects = [("high_school_mathematics", 10), ("high_school_computer_science", 10)]
    records = []
    for subject, n in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in ds.select(range(min(n, len(ds)))):
            # 保留原始 choices-list + int answer，
            # twinkle_eval.dataset._normalize_record 評測時自動轉換
            records.append({
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
            })
    save_jsonl(os.path.join(EXAMPLE_DIR, "mmlu", "test.jsonl"), records)


# ──────────────────────────────────────────────
# 5. MMLU Pro (最多 10 個選項，options 欄位)
#    直接展開為 A–J 具名欄位
# ──────────────────────────────────────────────
def create_mmlu_pro():
    print("\n[5/5] MMLU Pro (TIGER-Lab/MMLU-Pro)...")
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    records = []
    for row in ds.select(range(N)):
        options = list(row["options"])
        labels = [_index_to_label(i) for i in range(len(options))]
        rec = {"question": row["question"]}
        for label, text in zip(labels, options):
            rec[label] = text
        rec["answer"] = str(row["answer"]).strip().upper()
        records.append(rec)
    save_jsonl(os.path.join(EXAMPLE_DIR, "mmlu_pro", "test.jsonl"), records)


if __name__ == "__main__":
    print(f"建立範例資料集 → {os.path.abspath(EXAMPLE_DIR)}")
    create_gsm8k()
    create_aime()
    create_tmmluplus()
    create_mmlu()
    create_mmlu_pro()
    print("\n✅ 全部完成！")
