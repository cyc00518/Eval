"""評測策略模組 - 定義各種從 LLM 輸出中提取答案的策略

包含多種策略：
- PatternMatchingStrategy: 使用正則表達式模式匹配
- BoxExtractionStrategy: 提取 LaTeX 格式的 \\box{} 或 \\boxed{} 中的答案
- CustomRegexStrategy: 使用自定義正則表達式
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from mathruler.grader import grade_answer

class EvaluationStrategy(ABC):
    """評測策略抽象基本類別

    所有評測策略都必須從這個類別繼承，並實現必要的抽象方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer from LLM output."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass

    def validate_output(self, llm_output: Optional[str]) -> bool:
        """Validate the LLM output format."""
        return isinstance(llm_output, str) and llm_output.strip() != ""

    def normalize_answer(self, answer: str) -> str:
        """Normalize 提取答案，預設轉成大寫去除空白。"""
        return str(answer).strip().upper()

    def is_correct(self, predicted: Optional[str], gold: Optional[str]) -> bool:
        """預設直接比較 normalize 後結果。"""
        return predicted == gold


class PatternMatchingStrategy(EvaluationStrategy):
    """模式匹配策略 - 使用正則表達式在 LLM 輸出中尋找答案

    預設包含了多種中文和英文的答案模式，能夠處理大部分常見的答案格式
    """

    # 預設的答案匹配模式，包含中英文各種常見格式
    DEFAULT_PATTERNS = [
        r"correct answer is:\n\n\n([A-D]).",
        r"correct answer is:\n\n([A-D]).",
        r"correct answer is:\n([A-D]).",
        r"正確的答案應該是:.*?\b([A-D])\b",
        r"正确的答案应该是:.*?\b([A-D])\b",
        r"正確的選項應為:.*?\b([A-D])\b",
        r"正确的选项应为:.*?\b([A-D])\b",
        r"正確的答案是（([A-D])）",
        r"正确的答案是（([A-D])）",
        r"答案應該是:\s?選?項?\s?([A-D])",
        r"答案应该是:\s?选?项?\s?([A-D])",
        r"答案是:\s?選?項?\s?([A-D])",
        r"答案是:\s?选?项?\s?([A-D])",
        r"答案應為:\s?選?項?\s?([A-D])",
        r"答案应为:\s?选?项?\s?([A-D])",
        r"答案為:\s?([A-D])",
        r"答案应为：\s?([A-D])",
        r"答案為：\s?([A-D])",
        r"答案應該是:\s?([A-D])",
        r"正確答案為 \*\*([A-D])",
        r"正確答案為\(([A-D])\)",
        r"答案應為:\s?([A-D])",
        r"答案应为:\s?([A-D])",
        r"答案是 \*\*([A-D])",
        r"答案 ([A-D]) 正確",
        r"選項 ([A-D]) 正確",
        r"所以答案為([A-D])",
        r"答案：\(([A-D])\)",
        r"答案:\s?([A-D])",
        r"答案：\s?([A-D])",
        r"答案: ([A-D]) ",
        r"答案([A-D]) ",
        r"^選項([A-D])",
        r"^选项([A-D])",
        r"^選([A-D])",
        r"^选([A-D])",
        r"([A-D]). ",
        r"([A-D]).",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.patterns = self.config.get("patterns", self.DEFAULT_PATTERNS)

    def get_strategy_name(self) -> str:
        return "pattern"

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer using regex patterns."""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return self.normalize_answer(match.group(1))
        return None

    def add_pattern(self, pattern: str):
        """Add a custom pattern to the strategy."""
        if pattern not in self.patterns:
            self.patterns.append(pattern)


class BoxExtractionStrategy(EvaluationStrategy):
    """Strategy that extracts answers from LaTeX-style box formatting."""

    DEFAULT_PATTERNS = [r"\\{1,2}box{([A-D])}", r"\\{1,2}boxed{([A-D])}"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.patterns = self.config.get("patterns", self.DEFAULT_PATTERNS)

    def get_strategy_name(self) -> str:
        return "box"

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer from box/boxed formatting."""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return self.normalize_answer(match.group(1))
        return None

    def add_pattern(self, pattern: str):
        """Add a custom box pattern to the strategy."""
        if pattern not in self.patterns:
            self.patterns.append(pattern)


class CustomRegexStrategy(EvaluationStrategy):
    """Strategy that allows custom regex patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not self.config.get("patterns"):
            raise ValueError("CustomRegexStrategy requires 'patterns' in config")
        self.patterns = self.config["patterns"]

    def get_strategy_name(self) -> str:
        return "custom_regex"

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer using custom regex patterns."""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return match.group(1).strip()
        return None


class MathExtractionStrategy(EvaluationStrategy):
    """數學評測策略：固定以 box 標記抽取答案，並用 MathRuler 進行比對。"""

    def get_strategy_name(self) -> str:
        return "math"

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """只嘗試從 \\boxed{...} 或 \\box{...} 取出最後一個答案。"""
        if not self.validate_output(llm_output):
            return None

        boxed = self._extract_braced_content(llm_output, r"\boxed{")
        if boxed is not None:
            return boxed

        box = self._extract_braced_content(llm_output, r"\box{")
        if box is not None:
            return box

        return None

    def _extract_braced_content(self, text: str, prefix: str) -> Optional[str]:
        """從最後一次 prefix 出現後，擷取對應到 prefix 所開啟之大括號 {...} 內容（支援巢狀）。"""
        start = text.rfind(prefix)
        if start == -1:
            return None

        # 假設 prefix 以 "{" 結尾（例如 r"\boxed{"），因此此處已位於外層括號內
        content = text[start + len(prefix):]

        depth = 1  # 已在外層 { ... } 內
        for idx, ch in enumerate(content):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return content[:idx].strip()

        return None

    def is_correct(self, predicted: Optional[str], gold: Optional[str]) -> bool:
        """使用 MathRuler 的 grade_answer 進行等價判斷。"""
        if predicted is None or gold is None:
            return False
        if grade_answer(predicted, gold):
            return True
        return self._post_check_equivalence(predicted, gold)

    def normalize_answer(self, answer: str) -> str:
        """數學答案維持原格式僅做基礎修剪，避免破壞 LaTeX。"""
        return str(answer).strip()

    def _post_check_equivalence(self, predicted: str, gold: str) -> bool:
        """補強 MathRuler normalize 漏網的大小寫/排列差異。

        例：
        - \\FRAC{11}{36} 等價 \\frac{11}{36}
        - 1,-2 等價 -2, 1（忽略順序）
        - 2516_8 等價 2516（進位註記換算）
        """
        normalized_pred = self._normalize_latex_commands(predicted)
        normalized_gold = self._normalize_latex_commands(gold)

        if grade_answer(normalized_pred, normalized_gold):
            return True

        if self._compare_as_unordered_list(normalized_pred, normalized_gold):
            return True

        return self._check_base_suffix_equivalence(normalized_pred, normalized_gold)

    def _normalize_latex_commands(self, expr: str) -> str:
        """將 LaTeX 指令及環境名稱轉成小寫以利 parser 處理。

        例：\\FRAC{3\\SQRT{3}}{4} -> \\frac{3\\sqrt{3}}{4}
        """
        if expr is None:
            return expr

        lowered = re.sub(r"\\[A-Za-z]+", lambda m: m.group(0).lower(), expr)
        lowered = re.sub(r"\\mbox\{([^}]+)\}", lambda m: m.group(1), lowered)
        lowered = re.sub(
            r"(\\begin|\\end)\{([^}]+)\}",
            lambda m: f"{m.group(1)}{{{m.group(2).lower()}}}",
            lowered,
        )
        return lowered.lower()

    def _compare_as_unordered_list(self, predicted: str, gold: str) -> bool:
        """允許逗號分隔解集合忽略順序的比對。

        例：1,-2 等價 -2, 1
        """
        pred_items = self._split_simple_commas(predicted)
        gold_items = self._split_simple_commas(gold)

        if not pred_items or not gold_items or len(pred_items) != len(gold_items):
            return False

        used = [False] * len(gold_items)
        for pred_item in pred_items:
            matched = False
            for idx, gold_item in enumerate(gold_items):
                if used[idx]:
                    continue
                if grade_answer(pred_item, gold_item):
                    used[idx] = True
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _split_simple_commas(self, expr: str) -> List[str]:
        """拆解簡單逗號分隔的元素（避免矩陣/LaTeX 排版）。

        例：
        - 1,-2 -> ['1', '-2']
        """
        if expr is None:
            return []
        stripped = expr.strip()
        if "\\\\" in stripped or "\\begin" in stripped or "\\end" in stripped:
            return []
        if len(stripped) >= 2 and stripped[0] in "([{<" and stripped[-1] in ")]}>":
            stripped = stripped[1:-1]
        parts = [part.strip() for part in stripped.split(",") if part.strip() != ""]
        return parts if len(parts) > 1 else []

    def _check_base_suffix_equivalence(self, predicted: str, gold: str) -> bool:
        """處理純數字附帶進位註記造成的不等價。

        例：2516_8 等價 1326（十進位），以及 2516_8 等價 2516
        """
        pred_match = re.fullmatch(r"([+-]?[0-9A-Fa-f]+)_([0-9]+)", predicted.strip())
        gold_match = re.fullmatch(r"([+-]?[0-9A-Fa-f]+)_([0-9]+)", gold.strip())

        def _base_value(match: re.Match) -> Optional[int]:
            try:
                return int(match.group(1), int(match.group(2)))
            except ValueError:
                return None

        if pred_match and not gold_match:
            pred_val = _base_value(pred_match)
            if pred_val is not None and gold.strip().lstrip("+-").isdigit():
                if pred_val == int(gold):
                    return True
        if gold_match and not pred_match:
            gold_val = _base_value(gold_match)
            if gold_val is not None and predicted.strip().lstrip("+-").isdigit():
                if gold_val == int(predicted):
                    return True

        if pred_match or gold_match:
            stripped_pred = pred_match.group(1) if pred_match else predicted
            stripped_gold = gold_match.group(1) if gold_match else gold
            return grade_answer(stripped_pred, stripped_gold)

        return False


class EvaluationStrategyFactory:
    """Factory class for creating evaluation strategy instances."""

    _registry: Dict[str, Type[EvaluationStrategy]] = {
        "pattern": PatternMatchingStrategy,
        "box": BoxExtractionStrategy,
        "custom_regex": CustomRegexStrategy,
        "math": MathExtractionStrategy,
    }

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[EvaluationStrategy]):
        """Register a new evaluation strategy."""
        cls._registry[name] = strategy_class

    @classmethod
    def create_strategy(
        cls, strategy_type: str, config: Optional[Dict[str, Any]] = None
    ) -> EvaluationStrategy:
        """Create an evaluation strategy instance based on type."""
        if strategy_type not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported strategy type: {strategy_type}. Available types: {available_types}"
            )

        strategy_class = cls._registry[strategy_type]
        return strategy_class(config)

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available strategy types."""
        return list(cls._registry.keys())
