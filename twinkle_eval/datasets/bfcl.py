"""BFCL 資料集轉換工具。

將 BFCL 原始格式（分離的 questions.jsonl + possible_answer/answers.jsonl）
合併為 twinkle-eval 評測格式的單一 JSONL 檔案。

BFCL 原始格式（questions.jsonl 每行）：
    {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "..."}]],  # 雙層 list，外層 = turns
        "function": [{...}]   # function 定義列表
    }

BFCL 原始格式（answers.jsonl 每行）：
    {
        "id": "simple_0",
        "ground_truth": [{"func_name": {"param": [acceptable_values]}}]
    }

twinkle-eval 評測格式（輸出每行）：
    {
        "id": "simple_0",
        "question": "[{\"role\": \"user\", \"content\": \"...\"}]",  # JSON string，單 turn 的 messages
        "functions": "[{...}]",  # JSON string
        "answer": "{\"category\": \"simple\", \"ground_truth\": [...]}"  # JSON string
    }
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


def _infer_category(file_path: str) -> str:
    """從檔案路徑推斷 BFCL category 名稱。"""
    path = os.path.normpath(file_path)
    parts = path.split(os.sep)
    # 取倒數第二段（questions.jsonl 的上層目錄）
    for part in reversed(parts):
        if part and part not in ("questions.jsonl", "possible_answer", "answers.jsonl"):
            return part
    return "unknown"


def merge_bfcl_files(
    questions_path: str,
    answers_path: str,
    output_path: str,
    category: Optional[str] = None,
) -> int:
    """合併 BFCL questions 和 answers 為單一 JSONL 評測格式。

    Args:
        questions_path: BFCL questions.jsonl 路徑
        answers_path:   BFCL possible_answer/answers.jsonl 路徑
        output_path:    輸出的 JSONL 路徑（目錄若不存在會自動建立）
        category:       BFCL category 名稱（如 "simple"）；若為 None 則從路徑推斷

    Returns:
        成功合併的題目數量。
    """
    if category is None:
        category = _infer_category(questions_path)

    # 讀取 answers，建立 id → ground_truth 對應表
    answers: Dict[str, list] = {}
    with open(answers_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answers[obj["id"]] = obj["ground_truth"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = 0
    with (
        open(questions_path, encoding="utf-8") as qf,
        open(output_path, "w", encoding="utf-8") as of,
    ):
        for line in qf:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qid = q["id"]

            if qid not in answers:
                continue

            # question 取第一個 turn（BFCL 格式外層是 turns 列表）
            turns: List[List[dict]] = q["question"]
            messages: List[dict] = turns[0] if turns else []

            row = {
                "id": qid,
                "question": json.dumps(messages, ensure_ascii=False),
                "functions": json.dumps(q.get("function", []), ensure_ascii=False),
                "answer": json.dumps(
                    {"category": category, "ground_truth": answers[qid]},
                    ensure_ascii=False,
                ),
            }
            of.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def merge_bfcl_directory(
    questions_path: str,
    output_path: str,
    category: Optional[str] = None,
) -> int:
    """從含有 questions.jsonl 與 possible_answer/answers.jsonl 的目錄自動合併。

    Args:
        questions_path: 包含 questions.jsonl 的目錄路徑
        output_path:    輸出的 JSONL 檔案路徑
        category:       BFCL category；若為 None 則從目錄名稱推斷

    Returns:
        成功合併的題目數量。

    Raises:
        FileNotFoundError: 若 questions.jsonl 或 answers.jsonl 不存在。
    """
    q_file = os.path.join(questions_path, "questions.jsonl")
    a_file = os.path.join(questions_path, "possible_answer", "answers.jsonl")

    if not os.path.exists(q_file):
        raise FileNotFoundError(f"找不到 questions.jsonl: {q_file}")
    if not os.path.exists(a_file):
        raise FileNotFoundError(f"找不到 answers.jsonl: {a_file}")

    if category is None:
        category = os.path.basename(os.path.normpath(questions_path))

    return merge_bfcl_files(q_file, a_file, output_path, category=category)
