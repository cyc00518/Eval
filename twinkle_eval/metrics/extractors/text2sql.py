"""Text-to-SQL Extractor。

從 LLM 回應中提取 SQL 查詢語句。
支援：純 SQL 回應、```sql ... ``` 包裹、混雜文字中的 SELECT 語句。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Extractor


def extract_sql(text: str) -> Optional[str]:
    """從 LLM 回應中提取 SQL 查詢。

    依序嘗試：
    1. ```sql ... ``` markdown code block
    2. ``` ... ``` 通用 code block（內容看起來像 SQL）
    3. 以 SELECT / WITH / INSERT / UPDATE / DELETE 開頭的語句
    4. 整段文字視為 SQL（fallback）
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 1. ```sql ... ``` block
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        if sql:
            return _clean_sql(sql)

    # 2. ``` ... ``` block（內容含 SQL 關鍵字）
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if _looks_like_sql(candidate):
            return _clean_sql(candidate)

    # 3. 找以 SQL 關鍵字開頭的語句
    match = re.search(
        r"((?:SELECT|WITH|INSERT|UPDATE|DELETE)\b.*?)(?:;|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        sql = match.group(1).strip()
        if sql:
            return _clean_sql(sql)

    # 4. Fallback：整段文字
    if _looks_like_sql(text):
        return _clean_sql(text)

    return None


def _looks_like_sql(text: str) -> bool:
    """簡易判斷文字是否像 SQL 查詢。"""
    upper = text.upper().strip()
    sql_keywords = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP")
    return any(upper.startswith(kw) for kw in sql_keywords)


def _clean_sql(sql: str) -> str:
    """清理 SQL：移除結尾分號、多餘空白。"""
    sql = sql.strip().rstrip(";").strip()
    # 合併連續空白為單一空格
    sql = re.sub(r"\s+", " ", sql)
    return sql


class Text2SQLExtractor(Extractor):
    """Text-to-SQL Extractor — 從 LLM 回應中提取 SQL 查詢語句。"""

    def get_name(self) -> str:
        return "text2sql"

    def extract(self, raw: Optional[Any]) -> Optional[str]:
        """提取 SQL 查詢。

        Returns:
            提取到的 SQL 字串，或 None（無法提取時）。
        """
        if raw is None:
            return None
        return extract_sql(str(raw))
