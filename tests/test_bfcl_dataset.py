"""BFCL 資料集轉換工具測試 (twinkle_eval.datasets.bfcl)。"""

import json
import os
import tempfile

import pytest

from twinkle_eval.datasets.bfcl import merge_bfcl_directory, merge_bfcl_files


# ── 共用 fixtures ──────────────────────────────────────────────────────────────


def _write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


SAMPLE_QUESTIONS = [
    {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Find the area of a triangle."}]],
        "function": [
            {
                "name": "calculate_triangle_area",
                "description": "Calculate the area of a triangle.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer", "description": "Base."},
                        "height": {"type": "integer", "description": "Height."},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
    },
    {
        "id": "simple_1",
        "question": [[{"role": "user", "content": "Calculate factorial of 5."}]],
        "function": [
            {
                "name": "math.factorial",
                "description": "Calculate factorial.",
                "parameters": {
                    "type": "dict",
                    "properties": {"number": {"type": "integer", "description": "Number."}},
                    "required": ["number"],
                },
            }
        ],
    },
]

SAMPLE_ANSWERS = [
    {
        "id": "simple_0",
        "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}],
    },
    {
        "id": "simple_1",
        "ground_truth": [{"math.factorial": {"number": [5]}}],
    },
]

SAMPLE_PARALLEL_QUESTIONS = [
    {
        "id": "parallel_0",
        "question": [[{"role": "user", "content": "Play songs from Taylor Swift and Maroon 5."}]],
        "function": [
            {
                "name": "spotify.play",
                "description": "Play tracks.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "artist": {"type": "string", "description": "Artist name."},
                        "duration": {"type": "integer", "description": "Duration."},
                    },
                    "required": ["artist", "duration"],
                },
            }
        ],
    },
]

SAMPLE_PARALLEL_ANSWERS = [
    {
        "id": "parallel_0",
        "ground_truth": [
            {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
            {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
        ],
    },
]


@pytest.fixture()
def simple_dir(tmp_path):
    """建立 simple category 的原始 BFCL 目錄結構。"""
    q_path = str(tmp_path / "questions.jsonl")
    a_path = str(tmp_path / "possible_answer" / "answers.jsonl")
    _write_jsonl(q_path, SAMPLE_QUESTIONS)
    _write_jsonl(a_path, SAMPLE_ANSWERS)
    return str(tmp_path)


@pytest.fixture()
def parallel_dir(tmp_path):
    """建立 parallel category 的原始 BFCL 目錄結構。"""
    q_path = str(tmp_path / "questions.jsonl")
    a_path = str(tmp_path / "possible_answer" / "answers.jsonl")
    _write_jsonl(q_path, SAMPLE_PARALLEL_QUESTIONS)
    _write_jsonl(a_path, SAMPLE_PARALLEL_ANSWERS)
    return str(tmp_path)


# ── merge_bfcl_files ──────────────────────────────────────────────────────────


class TestMergeBfclFiles:
    def test_count_matches_questions_with_answers(self, simple_dir, tmp_path):
        q_path = os.path.join(simple_dir, "questions.jsonl")
        a_path = os.path.join(simple_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "output.jsonl")
        count = merge_bfcl_files(q_path, a_path, out, category="simple")
        assert count == 2

    def test_output_file_created(self, simple_dir, tmp_path):
        q_path = os.path.join(simple_dir, "questions.jsonl")
        a_path = os.path.join(simple_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "out" / "merged.jsonl")
        merge_bfcl_files(q_path, a_path, out, category="simple")
        assert os.path.exists(out)

    def test_row_schema(self, simple_dir, tmp_path):
        q_path = os.path.join(simple_dir, "questions.jsonl")
        a_path = os.path.join(simple_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "out.jsonl")
        merge_bfcl_files(q_path, a_path, out, category="simple")
        with open(out) as f:
            rows = [json.loads(l) for l in f]
        for row in rows:
            assert set(row.keys()) == {"id", "question", "functions", "answer"}

    def test_question_is_json_string_of_messages(self, simple_dir, tmp_path):
        q_path = os.path.join(simple_dir, "questions.jsonl")
        a_path = os.path.join(simple_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "out.jsonl")
        merge_bfcl_files(q_path, a_path, out, category="simple")
        with open(out) as f:
            row = json.loads(f.readline())
        messages = json.loads(row["question"])
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"

    def test_functions_is_json_string(self, simple_dir, tmp_path):
        q_path = os.path.join(simple_dir, "questions.jsonl")
        a_path = os.path.join(simple_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "out.jsonl")
        merge_bfcl_files(q_path, a_path, out, category="simple")
        with open(out) as f:
            row = json.loads(f.readline())
        functions = json.loads(row["functions"])
        assert isinstance(functions, list)
        assert "name" in functions[0]

    def test_answer_contains_category_and_ground_truth(self, simple_dir, tmp_path):
        q_path = os.path.join(simple_dir, "questions.jsonl")
        a_path = os.path.join(simple_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "out.jsonl")
        merge_bfcl_files(q_path, a_path, out, category="simple")
        with open(out) as f:
            row = json.loads(f.readline())
        answer = json.loads(row["answer"])
        assert answer["category"] == "simple"
        assert isinstance(answer["ground_truth"], list)
        assert len(answer["ground_truth"]) > 0

    def test_id_preserved(self, simple_dir, tmp_path):
        q_path = os.path.join(simple_dir, "questions.jsonl")
        a_path = os.path.join(simple_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "out.jsonl")
        merge_bfcl_files(q_path, a_path, out, category="simple")
        with open(out) as f:
            rows = [json.loads(l) for l in f]
        ids = {row["id"] for row in rows}
        assert "simple_0" in ids
        assert "simple_1" in ids

    def test_skips_question_without_answer(self, tmp_path):
        """若 questions 有 answer 裡沒有的 id，應跳過。"""
        q_path = str(tmp_path / "questions.jsonl")
        a_path = str(tmp_path / "possible_answer" / "answers.jsonl")
        _write_jsonl(q_path, SAMPLE_QUESTIONS)
        _write_jsonl(a_path, [SAMPLE_ANSWERS[0]])  # 只有 simple_0 的 answer
        out = str(tmp_path / "out.jsonl")
        count = merge_bfcl_files(q_path, a_path, out, category="simple")
        assert count == 1

    def test_parallel_ground_truth_preserved(self, parallel_dir, tmp_path):
        q_path = os.path.join(parallel_dir, "questions.jsonl")
        a_path = os.path.join(parallel_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "out.jsonl")
        merge_bfcl_files(q_path, a_path, out, category="parallel")
        with open(out) as f:
            row = json.loads(f.readline())
        answer = json.loads(row["answer"])
        assert answer["category"] == "parallel"
        assert len(answer["ground_truth"]) == 2  # parallel has 2 calls

    def test_output_creates_parent_dirs(self, simple_dir, tmp_path):
        q_path = os.path.join(simple_dir, "questions.jsonl")
        a_path = os.path.join(simple_dir, "possible_answer", "answers.jsonl")
        out = str(tmp_path / "deep" / "nested" / "dir" / "out.jsonl")
        merge_bfcl_files(q_path, a_path, out, category="simple")
        assert os.path.exists(out)


# ── merge_bfcl_directory ──────────────────────────────────────────────────────


class TestMergeBfclDirectory:
    def test_basic(self, simple_dir, tmp_path):
        out = str(tmp_path / "out.jsonl")
        count = merge_bfcl_directory(simple_dir, out, category="simple")
        assert count == 2

    def test_category_inferred_from_dir_name(self, tmp_path):
        cat_dir = tmp_path / "my_simple"
        q_path = str(cat_dir / "questions.jsonl")
        a_path = str(cat_dir / "possible_answer" / "answers.jsonl")
        _write_jsonl(q_path, SAMPLE_QUESTIONS)
        _write_jsonl(a_path, SAMPLE_ANSWERS)
        out = str(tmp_path / "out.jsonl")
        merge_bfcl_directory(str(cat_dir), out)
        with open(out) as f:
            row = json.loads(f.readline())
        answer = json.loads(row["answer"])
        assert answer["category"] == "my_simple"

    def test_raises_if_questions_missing(self, tmp_path):
        a_path = str(tmp_path / "possible_answer" / "answers.jsonl")
        _write_jsonl(a_path, SAMPLE_ANSWERS)
        with pytest.raises(FileNotFoundError, match="questions.jsonl"):
            merge_bfcl_directory(str(tmp_path), str(tmp_path / "out.jsonl"))

    def test_raises_if_answers_missing(self, tmp_path):
        q_path = str(tmp_path / "questions.jsonl")
        _write_jsonl(q_path, SAMPLE_QUESTIONS)
        with pytest.raises(FileNotFoundError, match="answers.jsonl"):
            merge_bfcl_directory(str(tmp_path), str(tmp_path / "out.jsonl"))


# ── 整合：驗證範例資料集 ─────────────────────────────────────────────────────────


EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "example")


class TestExampleDatasets:
    """驗證 datasets/example/bfcl_v1/ 和 bfcl_v2/ 的合併資料集格式正確。"""

    @pytest.mark.parametrize("filename,expected_category", [
        ("simple.jsonl", "simple"),
        ("parallel.jsonl", "parallel"),
        ("multiple.jsonl", "multiple"),
    ])
    def test_bfcl_v1_files(self, filename, expected_category):
        path = os.path.join(EXAMPLE_DIR, "bfcl_v1", filename)
        assert os.path.exists(path), f"找不到範例資料集: {path}"
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) > 0
        for row in rows:
            assert set(row.keys()) == {"id", "question", "functions", "answer"}
            msgs = json.loads(row["question"])
            assert isinstance(msgs, list) and msgs
            funcs = json.loads(row["functions"])
            assert isinstance(funcs, list) and funcs
            ans = json.loads(row["answer"])
            assert ans["category"] == expected_category
            assert isinstance(ans["ground_truth"], list) and ans["ground_truth"]

    def test_bfcl_v2_live_simple(self):
        path = os.path.join(EXAMPLE_DIR, "bfcl_v2", "live_simple.jsonl")
        assert os.path.exists(path), f"找不到範例資料集: {path}"
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) > 0
        for row in rows:
            ans = json.loads(row["answer"])
            assert ans["category"] == "live_simple"

    def test_bfcl_v3_raw_file_exists(self):
        path = os.path.join(EXAMPLE_DIR, "bfcl_v3", "multi_turn_questions.jsonl")
        assert os.path.exists(path), f"找不到 v3 原始資料: {path}"
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) > 0
        # v3 格式有 initial_config 和 path 欄位
        assert "initial_config" in rows[0]
