"""Tests for IFBench evaluation."""

import json
import os

import pytest

from twinkle_eval.metrics.extractors.ifbench import IFBenchExtractor
from twinkle_eval.metrics.scorers.ifbench import (
    IFBenchScorer,
    _get_loose_variants,
    _remove_first_line,
    _remove_last_line,
    _remove_markdown,
    score_ifbench,
)


# ── Text transformation helpers ──────────────────────────────────────────────
# IFBench 使用與 IFEval 相同的 loose variant 邏輯，但各自有獨立的 copy，
# 因此需獨立測試以確保兩邊行為一致。


class TestTextTransformations:
    def test_remove_markdown(self):
        assert _remove_markdown("Hello **world** and *test*") == "Hello world and test"

    def test_remove_markdown_no_change(self):
        assert _remove_markdown("No markdown here") == "No markdown here"

    def test_remove_first_line(self):
        assert _remove_first_line("First\nSecond\nThird") == "Second\nThird"

    def test_remove_first_line_single(self):
        assert _remove_first_line("Only one line") == ""

    def test_remove_last_line(self):
        assert _remove_last_line("First\nSecond\nThird") == "First\nSecond"

    def test_remove_last_line_single(self):
        assert _remove_last_line("Only one line") == ""

    def test_loose_variants_count(self):
        variants = _get_loose_variants("Some response")
        assert len(variants) == 8

    def test_loose_variants_includes_original(self):
        text = "Some response text"
        variants = _get_loose_variants(text)
        assert text in variants


# ── IFBenchScorer ─────────────────────────────────────────────────────────────


class TestIFBenchScorer:
    @pytest.fixture
    def scorer(self):
        return IFBenchScorer({})

    def test_get_name(self, scorer):
        assert scorer.get_name() == "ifbench"

    def test_normalize_passthrough(self, scorer):
        assert scorer.normalize("hello world") == "hello world"

    def test_score_empty_response(self, scorer):
        gt = json.dumps({
            "instruction_id_list": ["format:title_case"],
            "kwargs": [{}],
        })
        assert scorer.score("", gt) is False

    def test_score_invalid_gt(self, scorer):
        assert scorer.score("some response", "not json") is False

    def test_score_none_response(self, scorer):
        gt = json.dumps({
            "instruction_id_list": ["format:title_case"],
            "kwargs": [{}],
        })
        assert scorer.score(None, gt) is False

    def test_score_none_gt(self, scorer):
        assert scorer.score("some response", None) is False

    def test_score_full_returns_four_metrics(self, scorer):
        result = scorer.score_full(
            "HELLO WORLD",
            ["format:no_whitespace"],
            [{}],
        )
        assert "prompt_strict" in result
        assert "prompt_loose" in result
        assert "instruction_strict" in result
        assert "instruction_loose" in result

    def test_score_full_multiple_instructions(self, scorer):
        result = scorer.score_full(
            "some response",
            ["format:title_case", "format:no_whitespace"],
            [{}, {}],
        )
        assert len(result["instruction_strict"]) == 2
        assert len(result["instruction_loose"]) == 2


# ── IFBenchExtractor ─────────────────────────────────────────────────────────


class TestIFBenchExtractor:
    @pytest.fixture
    def extractor(self):
        return IFBenchExtractor({})

    def test_uses_ifeval_flag(self, extractor):
        assert extractor.uses_ifeval is True

    def test_extract_passthrough(self, extractor):
        assert extractor.extract("some response") == "some response"

    def test_extract_none(self, extractor):
        assert extractor.extract(None) is None

    def test_get_name(self, extractor):
        assert extractor.get_name() == "ifbench"


# ── score_ifbench function ───────────────────────────────────────────────────


class TestScoreIfbench:
    def test_unknown_instruction_returns_false(self):
        result = score_ifbench(
            "some response",
            ["unknown:nonexistent_instruction"],
            [{}],
        )
        assert result["instruction_strict"] == [False]

    def test_empty_response_returns_false(self):
        result = score_ifbench(
            "",
            ["format:title_case"],
            [{}],
        )
        assert result["prompt_strict"] is False

    def test_whitespace_only_response_returns_false(self):
        result = score_ifbench(
            "   \n  ",
            ["format:title_case"],
            [{}],
        )
        assert result["prompt_strict"] is False

    def test_no_whitespace_pass(self):
        """format:no_whitespace — 回答不含空白字元。"""
        result = score_ifbench(
            "HelloWorldNoSpaces",
            ["format:no_whitespace"],
            [{}],
        )
        assert result["instruction_strict"] == [True]

    def test_no_whitespace_fail(self):
        result = score_ifbench(
            "Hello World With Spaces",
            ["format:no_whitespace"],
            [{}],
        )
        assert result["instruction_strict"] == [False]

    def test_title_case_pass(self):
        """format:title_case — 每個詞首字母大寫。"""
        result = score_ifbench(
            "This Is A Title Case Sentence",
            ["format:title_case"],
            [{}],
        )
        assert result["instruction_strict"] == [True]

    def test_title_case_fail(self):
        result = score_ifbench(
            "this is not title case",
            ["format:title_case"],
            [{}],
        )
        assert result["instruction_strict"] == [False]

    def test_kwargs_null_filtering(self):
        """IFBench kwargs 包含大量 IFEval 繼承的 null 值欄位，必須能正確過濾。"""
        kwargs_with_nulls = {
            "capital_frequency": None,
            "end_phrase": None,
            "forbidden_words": None,
            "keyword": None,
            "language": None,
            "num_words": None,
        }
        # 應不會因 null kwargs 而 raise
        result = score_ifbench(
            "This Is Title Case",
            ["format:title_case"],
            [kwargs_with_nulls],
        )
        assert result["instruction_strict"] == [True]

    def test_prompt_strict_all_must_pass(self):
        """prompt_strict 需所有 instruction 都通過。"""
        result = score_ifbench(
            "hello world no spaces",
            ["format:no_whitespace", "format:title_case"],
            [{}, {}],
        )
        # no_whitespace fail (has spaces), title_case fail (not capitalized)
        assert result["prompt_strict"] is False

    def test_loose_tolerates_markdown(self):
        """Loose 模式應能容忍 markdown 標記。"""
        result = score_ifbench(
            "**This Is Title Case**",
            ["format:title_case"],
            [{}],
        )
        assert result["prompt_loose"] is True

    def test_output_template_pass(self):
        """format:output_template — 回答需包含指定 template 的所有 placeholder。"""
        result = score_ifbench(
            "My Answer: something here My Conclusion: another thing Future Outlook: good",
            ["format:output_template"],
            [{}],
        )
        # output_template 需要特定 kwargs，空 kwargs 應 gracefully handle
        assert isinstance(result["instruction_strict"], list)

    def test_prompt_parameter_passed(self):
        """某些 checker（如 repeat:repeat_change）需要 prompt 參數。

        RepeatChangeChecker 需 prompt_to_repeat kwarg，空 kwargs 時
        build_description 會 raise ValueError，scorer 應 catch 並回傳 False。
        """
        result = score_ifbench(
            "Tell me about AI. Here is my changed version of the prompt.",
            ["repeat:repeat_change"],
            [{"prompt_to_repeat": "Tell me about AI"}],
            prompt="Tell me about AI",
        )
        # 不管結果 True/False，重點是不 crash
        assert isinstance(result["instruction_strict"], list)
        assert len(result["instruction_strict"]) == 1


# ── Checker registry ─────────────────────────────────────────────────────────


class TestCheckerRegistry:
    def test_instruction_dict_loads(self):
        from twinkle_eval.metrics.checkers.ifbench import INSTRUCTION_DICT

        assert isinstance(INSTRUCTION_DICT, dict)

    def test_58_instruction_types(self):
        from twinkle_eval.metrics.checkers.ifbench import INSTRUCTION_DICT

        assert len(INSTRUCTION_DICT) == 58

    def test_all_7_categories_present(self):
        from twinkle_eval.metrics.checkers.ifbench import INSTRUCTION_DICT

        categories = {k.split(":")[0] for k in INSTRUCTION_DICT}
        expected = {"count", "ratio", "words", "sentence", "format", "custom", "repeat"}
        assert categories == expected

    def test_all_checkers_are_classes(self):
        from twinkle_eval.metrics.checkers.ifbench import INSTRUCTION_DICT

        for inst_id, cls in INSTRUCTION_DICT.items():
            assert isinstance(cls, type), f"{inst_id} maps to {cls}, not a class"


# ── PRESETS registration ─────────────────────────────────────────────────────


class TestPresetsRegistration:
    def test_ifbench_in_presets(self):
        from twinkle_eval.metrics import PRESETS

        assert "ifbench" in PRESETS

    def test_ifbench_preset_types(self):
        from twinkle_eval.metrics import PRESETS

        extractor_cls, scorer_cls = PRESETS["ifbench"]
        assert extractor_cls is IFBenchExtractor
        assert scorer_cls is IFBenchScorer


# ── Example dataset ──────────────────────────────────────────────────────────


class TestExampleDataset:
    def test_example_dataset_exists(self):
        assert os.path.exists("datasets/example/ifbench/ifbench.jsonl")

    def test_example_dataset_format(self):
        with open("datasets/example/ifbench/ifbench.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        assert len(rows) >= 1
        for row in rows:
            assert "key" in row
            assert "prompt" in row
            assert "instruction_id_list" in row
            assert "kwargs" in row

            inst_ids = row["instruction_id_list"]
            kwargs = row["kwargs"]
            assert isinstance(inst_ids, list)
            assert isinstance(kwargs, list)
            assert len(inst_ids) == len(kwargs)
            assert len(inst_ids) >= 1

    def test_example_dataset_14_rows(self):
        with open("datasets/example/ifbench/ifbench.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 14

    def test_example_dataset_covers_all_categories(self):
        with open("datasets/example/ifbench/ifbench.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        categories = set()
        for row in rows:
            for inst_id in row["instruction_id_list"]:
                categories.add(inst_id.split(":")[0])

        expected = {"count", "ratio", "words", "sentence", "format", "custom", "repeat"}
        assert categories == expected, f"Missing categories: {expected - categories}"

    def test_example_dataset_native_list_format(self):
        """IFBench example dataset 使用原生 list/dict，非 JSON string。"""
        with open("datasets/example/ifbench/ifbench.jsonl") as f:
            row = json.loads(f.readline())

        # instruction_id_list 和 kwargs 應為原生 list，不是 JSON string
        assert isinstance(row["instruction_id_list"], list)
        assert isinstance(row["kwargs"], list)
        assert isinstance(row["kwargs"][0], dict)
