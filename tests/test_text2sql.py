"""tests/test_text2sql.py — Text-to-SQL 評測方法的單元測試。

涵蓋：
- SQL Extractor（各種 LLM 回應格式）
- SQL Scorer（EM 模式、EX 模式、結果集比較）
- SQL 正規化
- PRESETS 註冊
- Example datasets（Spider 1.0、BIRD、Spider 2.0-lite）
"""

import json
import os
import sqlite3
import tempfile

import pytest

from twinkle_eval.metrics import PRESETS, create_metric_pair
from twinkle_eval.metrics.extractors.text2sql import (
    Text2SQLExtractor,
    _clean_sql,
    _looks_like_sql,
    extract_sql,
)
from twinkle_eval.metrics.scorers.text2sql import (
    Text2SQLScorer,
    _parse_gold,
    execute_sql,
    normalize_sql,
    result_sets_match,
)


# ──────────────────────────────────────────────────────────────
# SQL Extraction
# ──────────────────────────────────────────────────────────────


class TestExtractSQL:
    """從 LLM 回應中提取 SQL。"""

    def test_plain_select(self) -> None:
        assert extract_sql("SELECT * FROM users") == "SELECT * FROM users"

    def test_sql_code_block(self) -> None:
        text = "Here's the query:\n```sql\nSELECT count(*) FROM orders;\n```"
        assert extract_sql(text) == "SELECT count(*) FROM orders"

    def test_generic_code_block(self) -> None:
        text = "```\nSELECT name FROM students WHERE age > 20\n```"
        assert extract_sql(text) == "SELECT name FROM students WHERE age > 20"

    def test_mixed_text_with_select(self) -> None:
        text = "The answer is:\nSELECT id, name FROM products WHERE price > 100;"
        assert extract_sql(text) == "SELECT id, name FROM products WHERE price > 100"

    def test_with_clause(self) -> None:
        text = "WITH cte AS (SELECT * FROM t) SELECT * FROM cte"
        result = extract_sql(text)
        assert result is not None
        assert result.startswith("WITH cte AS")

    def test_none_input(self) -> None:
        assert extract_sql("") is None
        assert extract_sql("   ") is None

    def test_no_sql_content(self) -> None:
        assert extract_sql("I don't know how to write SQL for that.") is None

    def test_removes_trailing_semicolon(self) -> None:
        assert extract_sql("SELECT 1;") == "SELECT 1"

    def test_multiline_sql(self) -> None:
        text = """```sql
SELECT s.name, COUNT(*) AS cnt
FROM students s
JOIN enrollments e ON s.id = e.student_id
GROUP BY s.name
ORDER BY cnt DESC
```"""
        result = extract_sql(text)
        assert result is not None
        assert "SELECT s.name" in result
        assert "ORDER BY cnt DESC" in result

    def test_insert_statement(self) -> None:
        text = "INSERT INTO users (name) VALUES ('test')"
        assert extract_sql(text) is not None


class TestText2SQLExtractor:
    """Text2SQLExtractor 類別測試。"""

    def test_get_name(self) -> None:
        ext = Text2SQLExtractor()
        assert ext.get_name() == "text2sql"

    def test_extract_basic(self) -> None:
        ext = Text2SQLExtractor()
        result = ext.extract("SELECT count(*) FROM users")
        assert result == "SELECT count(*) FROM users"

    def test_extract_none(self) -> None:
        ext = Text2SQLExtractor()
        assert ext.extract(None) is None

    def test_extract_code_block(self) -> None:
        ext = Text2SQLExtractor()
        result = ext.extract("```sql\nSELECT 1\n```")
        assert result == "SELECT 1"


# ──────────────────────────────────────────────────────────────
# SQL Normalization
# ──────────────────────────────────────────────────────────────


class TestNormalizeSQL:
    """SQL 正規化。"""

    def test_lowercase(self) -> None:
        assert normalize_sql("SELECT * FROM Users") == "select * from users"

    def test_strip_whitespace(self) -> None:
        assert normalize_sql("  SELECT  *  FROM  t  ; ") == "select * from t"

    def test_remove_semicolon(self) -> None:
        assert normalize_sql("SELECT 1;") == "select 1"

    def test_collapse_newlines(self) -> None:
        sql = "SELECT\n  name\nFROM\n  t"
        assert normalize_sql(sql) == "select name from t"


# ──────────────────────────────────────────────────────────────
# Gold Answer Parsing
# ──────────────────────────────────────────────────────────────


class TestParseGold:
    """解析 gold answer。"""

    def test_json_format(self) -> None:
        gold = json.dumps({"sql": "SELECT 1", "db_id": "test_db"})
        sql, db_id = _parse_gold(gold)
        assert sql == "SELECT 1"
        assert db_id == "test_db"

    def test_plain_sql(self) -> None:
        sql, db_id = _parse_gold("SELECT * FROM users")
        assert sql == "SELECT * FROM users"
        assert db_id is None

    def test_none(self) -> None:
        sql, db_id = _parse_gold(None)
        assert sql == ""
        assert db_id is None


# ──────────────────────────────────────────────────────────────
# Result Set Matching
# ──────────────────────────────────────────────────────────────


class TestResultSetsMatch:
    """結果集比較。"""

    def test_identical(self) -> None:
        a = [(1, "alice"), (2, "bob")]
        b = [(1, "alice"), (2, "bob")]
        assert result_sets_match(a, b) is True

    def test_different_order(self) -> None:
        a = [(2, "bob"), (1, "alice")]
        b = [(1, "alice"), (2, "bob")]
        assert result_sets_match(a, b) is True

    def test_different_content(self) -> None:
        a = [(1, "alice")]
        b = [(1, "bob")]
        assert result_sets_match(a, b) is False

    def test_different_length(self) -> None:
        a = [(1,), (2,)]
        b = [(1,)]
        assert result_sets_match(a, b) is False

    def test_none_results(self) -> None:
        assert result_sets_match(None, [(1,)]) is False
        assert result_sets_match([(1,)], None) is False

    def test_float_precision(self) -> None:
        a = [(1.0000001,)]
        b = [(1.0000002,)]
        assert result_sets_match(a, b) is True

    def test_string_case_insensitive(self) -> None:
        a = [("Alice",)]
        b = [("alice",)]
        assert result_sets_match(a, b) is True

    def test_empty_results(self) -> None:
        assert result_sets_match([], []) is True


# ──────────────────────────────────────────────────────────────
# SQL Execution
# ──────────────────────────────────────────────────────────────


class TestExecuteSQL:
    """SQLite 執行。"""

    @pytest.fixture()
    def test_db(self, tmp_path: str) -> str:
        db_path = os.path.join(str(tmp_path), "test.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'alice')")
        conn.execute("INSERT INTO t VALUES (2, 'bob')")
        conn.commit()
        conn.close()
        return db_path

    def test_basic_query(self, test_db: str) -> None:
        result = execute_sql(test_db, "SELECT count(*) FROM t")
        assert result == [(2,)]

    def test_invalid_sql(self, test_db: str) -> None:
        result = execute_sql(test_db, "INVALID SQL QUERY")
        assert result is None

    def test_nonexistent_db(self) -> None:
        result = execute_sql("/nonexistent/path.sqlite", "SELECT 1")
        assert result is None


# ──────────────────────────────────────────────────────────────
# Scorer
# ──────────────────────────────────────────────────────────────


class TestText2SQLScorer:
    """Text2SQL Scorer 測試。"""

    def test_get_name(self) -> None:
        scorer = Text2SQLScorer()
        assert scorer.get_name() == "text2sql"

    def test_normalize_passthrough(self) -> None:
        scorer = Text2SQLScorer()
        gold = '{"sql": "SELECT 1", "db_id": "test"}'
        assert scorer.normalize(gold) == gold

    def test_em_mode_match(self) -> None:
        scorer = Text2SQLScorer({"text2sql_scoring_mode": "em"})
        gold = json.dumps({"sql": "SELECT count(*) FROM singer", "db_id": "test"})
        assert scorer.score("select count(*) from singer", gold) is True

    def test_em_mode_mismatch(self) -> None:
        scorer = Text2SQLScorer({"text2sql_scoring_mode": "em"})
        gold = json.dumps({"sql": "SELECT count(*) FROM singer", "db_id": "test"})
        assert scorer.score("SELECT name FROM singer", gold) is False

    def test_em_mode_whitespace_insensitive(self) -> None:
        scorer = Text2SQLScorer({"text2sql_scoring_mode": "em"})
        gold = json.dumps({"sql": "SELECT  count(*)  FROM  singer", "db_id": "test"})
        assert scorer.score("SELECT count(*) FROM singer", gold) is True

    def test_em_mode_case_insensitive(self) -> None:
        scorer = Text2SQLScorer({"text2sql_scoring_mode": "em"})
        gold = json.dumps({"sql": "SELECT COUNT(*) FROM Singer", "db_id": "test"})
        assert scorer.score("select count(*) from singer", gold) is True

    def test_score_empty_predicted(self) -> None:
        scorer = Text2SQLScorer()
        assert scorer.score("", '{"sql": "SELECT 1", "db_id": "t"}') is False
        assert scorer.score(None, '{"sql": "SELECT 1", "db_id": "t"}') is False

    @pytest.fixture()
    def db_setup(self, tmp_path: str) -> tuple:
        """建立測試用 SQLite 資料庫。"""
        db_dir = os.path.join(str(tmp_path), "test_db")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "test_db.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE singer (id INTEGER, name TEXT, country TEXT)")
        conn.execute("INSERT INTO singer VALUES (1, 'Adele', 'UK')")
        conn.execute("INSERT INTO singer VALUES (2, 'Ed Sheeran', 'UK')")
        conn.execute("INSERT INTO singer VALUES (3, 'Taylor Swift', 'US')")
        conn.commit()
        conn.close()
        return str(tmp_path), db_path

    def test_exec_mode_correct(self, db_setup: tuple) -> None:
        base_path, _ = db_setup
        scorer = Text2SQLScorer({
            "text2sql_scoring_mode": "exec",
            "text2sql_db_base_path": base_path,
        })
        gold = json.dumps({"sql": "SELECT count(*) FROM singer", "db_id": "test_db"})
        # Semantically equivalent SQL
        assert scorer.score("SELECT COUNT(*) FROM singer", gold) is True

    def test_exec_mode_wrong_result(self, db_setup: tuple) -> None:
        base_path, _ = db_setup
        scorer = Text2SQLScorer({
            "text2sql_scoring_mode": "exec",
            "text2sql_db_base_path": base_path,
        })
        gold = json.dumps({"sql": "SELECT count(*) FROM singer", "db_id": "test_db"})
        assert scorer.score("SELECT count(*) FROM singer WHERE country = 'US'", gold) is False

    def test_exec_mode_different_sql_same_result(self, db_setup: tuple) -> None:
        base_path, _ = db_setup
        scorer = Text2SQLScorer({
            "text2sql_scoring_mode": "exec",
            "text2sql_db_base_path": base_path,
        })
        gold = json.dumps({
            "sql": "SELECT name FROM singer WHERE country = 'UK' ORDER BY name",
            "db_id": "test_db",
        })
        # Different SQL, same result
        pred = "SELECT name FROM singer WHERE country = 'UK' ORDER BY name ASC"
        assert scorer.score(pred, gold) is True

    def test_exec_fallback_to_em_no_db(self) -> None:
        scorer = Text2SQLScorer({
            "text2sql_scoring_mode": "exec",
            "text2sql_db_base_path": "/nonexistent/path",
        })
        gold = json.dumps({"sql": "SELECT count(*) FROM t", "db_id": "missing"})
        # Falls back to EM — same normalized SQL should match
        assert scorer.score("select count(*) from t", gold) is True

    def test_exec_invalid_predicted_sql(self, db_setup: tuple) -> None:
        base_path, _ = db_setup
        scorer = Text2SQLScorer({
            "text2sql_scoring_mode": "exec",
            "text2sql_db_base_path": base_path,
        })
        gold = json.dumps({"sql": "SELECT count(*) FROM singer", "db_id": "test_db"})
        assert scorer.score("INVALID SQL!!!", gold) is False


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────


class TestHelpers:
    """輔助函式測試。"""

    def test_looks_like_sql(self) -> None:
        assert _looks_like_sql("SELECT * FROM t") is True
        assert _looks_like_sql("WITH cte AS ...") is True
        assert _looks_like_sql("Hello world") is False
        assert _looks_like_sql("CREATE TABLE t (id INT)") is True

    def test_clean_sql(self) -> None:
        assert _clean_sql("  SELECT  1  ;  ") == "SELECT 1"
        assert _clean_sql("SELECT\n  *\n  FROM\n  t") == "SELECT * FROM t"


# ──────────────────────────────────────────────────────────────
# PRESETS Registration
# ──────────────────────────────────────────────────────────────


class TestPresets:
    """PRESETS 註冊驗證。"""

    def test_text2sql_in_presets(self) -> None:
        assert "text2sql" in PRESETS

    def test_preset_classes(self) -> None:
        ext_cls, scorer_cls = PRESETS["text2sql"]
        assert ext_cls is Text2SQLExtractor
        assert scorer_cls is Text2SQLScorer

    def test_create_metric_pair(self) -> None:
        ext, scorer = create_metric_pair("text2sql")
        assert isinstance(ext, Text2SQLExtractor)
        assert isinstance(scorer, Text2SQLScorer)

    def test_create_metric_pair_with_config(self) -> None:
        cfg = {"text2sql_scoring_mode": "em"}
        ext, scorer = create_metric_pair("text2sql", cfg)
        assert isinstance(scorer, Text2SQLScorer)
        assert scorer.scoring_mode == "em"


# ──────────────────────────────────────────────────────────────
# Example Datasets
# ──────────────────────────────────────────────────────────────


DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "example")


class TestExampleDatasetSpider:
    """Spider 1.0 example dataset 驗證。"""

    SPIDER_DIR = os.path.join(DATASETS_DIR, "spider")
    JSONL_PATH = os.path.join(SPIDER_DIR, "spider_dev.jsonl")

    def test_jsonl_exists(self) -> None:
        assert os.path.isfile(self.JSONL_PATH)

    def test_row_count(self) -> None:
        with open(self.JSONL_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 10

    def test_required_fields(self) -> None:
        with open(self.JSONL_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            assert "id" in row
            assert "question" in row
            assert "answer" in row
            assert "db_id" in row
            # answer 應為 JSON 字串
            parsed = json.loads(row["answer"])
            assert "sql" in parsed
            assert "db_id" in parsed

    def test_databases_exist(self) -> None:
        db_dir = os.path.join(self.SPIDER_DIR, "databases")
        assert os.path.isdir(os.path.join(db_dir, "concert_singer"))
        assert os.path.isdir(os.path.join(db_dir, "pets_1"))
        assert os.path.isfile(os.path.join(db_dir, "concert_singer", "concert_singer.sqlite"))
        assert os.path.isfile(os.path.join(db_dir, "pets_1", "pets_1.sqlite"))

    def test_db_ids_match(self) -> None:
        with open(self.JSONL_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        db_ids = {row["db_id"] for row in rows}
        assert db_ids == {"concert_singer", "pets_1"}


class TestExampleDatasetBIRD:
    """BIRD example dataset 驗證。"""

    BIRD_DIR = os.path.join(DATASETS_DIR, "bird")
    JSONL_PATH = os.path.join(BIRD_DIR, "bird_dev.jsonl")

    def test_jsonl_exists(self) -> None:
        assert os.path.isfile(self.JSONL_PATH)

    def test_row_count(self) -> None:
        with open(self.JSONL_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 10

    def test_has_evidence_field(self) -> None:
        with open(self.JSONL_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            assert "evidence" in row

    def test_databases_exist(self) -> None:
        db_dir = os.path.join(self.BIRD_DIR, "databases")
        assert os.path.isfile(
            os.path.join(db_dir, "california_schools", "california_schools.sqlite")
        )
        assert os.path.isfile(os.path.join(db_dir, "financial", "financial.sqlite"))

    def test_db_ids_match(self) -> None:
        with open(self.JSONL_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        db_ids = {row["db_id"] for row in rows}
        assert db_ids == {"california_schools", "financial"}


class TestExampleDatasetSpider2Lite:
    """Spider 2.0-lite example dataset 驗證。"""

    S2_DIR = os.path.join(DATASETS_DIR, "spider2_lite")
    JSONL_PATH = os.path.join(S2_DIR, "spider2_lite_dev.jsonl")

    def test_jsonl_exists(self) -> None:
        assert os.path.isfile(self.JSONL_PATH)

    def test_row_count(self) -> None:
        with open(self.JSONL_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 10

    def test_databases_exist(self) -> None:
        db_dir = os.path.join(self.S2_DIR, "databases")
        assert os.path.isfile(os.path.join(db_dir, "book_store", "book_store.sqlite"))

    def test_required_fields(self) -> None:
        with open(self.JSONL_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            assert "id" in row
            assert "question" in row
            assert "answer" in row
            assert "db_id" in row


# ──────────────────────────────────────────────────────────────
# Integration: Extractor + Scorer
# ──────────────────────────────────────────────────────────────


class TestIntegration:
    """Extractor + Scorer 整合測試。"""

    def test_em_pipeline(self) -> None:
        ext = Text2SQLExtractor()
        scorer = Text2SQLScorer({"text2sql_scoring_mode": "em"})

        llm_output = "```sql\nSELECT count(*) FROM singer\n```"
        gold = json.dumps({"sql": "select count(*) from singer", "db_id": "test"})

        predicted = ext.extract(llm_output)
        assert predicted is not None
        assert scorer.score(predicted, gold) is True

    def test_exec_pipeline_with_spider_db(self) -> None:
        """使用 Spider example 資料庫進行 EX 模式整合測試。"""
        db_base = os.path.join(DATASETS_DIR, "spider", "databases")
        if not os.path.isdir(db_base):
            pytest.skip("Spider example databases not found")

        ext = Text2SQLExtractor()
        scorer = Text2SQLScorer({
            "text2sql_scoring_mode": "exec",
            "text2sql_db_base_path": db_base,
        })

        # 模擬 LLM 回應
        llm_output = "SELECT COUNT(*) FROM singer"
        gold = json.dumps({"sql": "SELECT count(*) FROM singer", "db_id": "concert_singer"})

        predicted = ext.extract(llm_output)
        assert predicted is not None
        assert scorer.score(predicted, gold) is True

    def test_exec_pipeline_wrong_answer(self) -> None:
        db_base = os.path.join(DATASETS_DIR, "spider", "databases")
        if not os.path.isdir(db_base):
            pytest.skip("Spider example databases not found")

        ext = Text2SQLExtractor()
        scorer = Text2SQLScorer({
            "text2sql_scoring_mode": "exec",
            "text2sql_db_base_path": db_base,
        })

        llm_output = "SELECT COUNT(*) FROM concert"
        gold = json.dumps({"sql": "SELECT count(*) FROM singer", "db_id": "concert_singer"})

        predicted = ext.extract(llm_output)
        assert predicted is not None
        assert scorer.score(predicted, gold) is False
