"""評測策略模組 - 定義各種從 LLM 輸出中提取答案的策略

包含多種策略：
- PatternMatchingStrategy: 使用正則表達式模式匹配
- BoxExtractionStrategy: 提取 LaTeX 格式的 \\box{} 或 \\boxed{} 中的答案
- CustomRegexStrategy: 使用自定義正則表達式
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import sympy as _sp  # type: ignore
from latex2sympy2_extended.latex2sympy2 import NormalizationConfig as _NormConfig  # type: ignore
from latex2sympy2_extended.latex2sympy2 import latex2sympy as _latex2sympy  # type: ignore
from latex2sympy2_extended.latex2sympy2 import normalize_latex as _normalize_latex  # type: ignore
_HAS_LATEX2SYMPY_EXT = True


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

    def is_correct(self, predicted: Optional[str], gold: Optional[str]) -> bool:
        """Determine if a predicted answer matches the gold answer."""
        return predicted == gold
    def normalize_answer(self, answer: str) -> str:
        """Normalize an extracted answer for comparison."""
        text = str(answer).strip().upper()
        # 移除常見前綴 (ANSWER:, FINAL ANSWER:)
        text = re.sub(r"^(FINAL\\s+ANSWER|FINAL\\s*ANS|ANSWER)\\s*[:：]\\s*", "", text)
        return text


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

    DEFAULT_NUMERIC_PATTERNS = [
        r"[Ff]inal\s*[Aa]nswer\s*[:：]?\s*([^\n]+)",
        r"[Aa]nswer\s*[:：]\s*([^\n]+)",
        r"答案\s*[:：]\s*([^\n]+)",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.patterns = self.config.get("patterns", self.DEFAULT_PATTERNS)
        self.numeric_patterns = self.config.get("numeric_patterns", self.DEFAULT_NUMERIC_PATTERNS)
        self.enable_fallback_numeric = self.config.get("enable_fallback_numeric", True)

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

        for pattern in self.numeric_patterns:
            match = re.search(pattern, llm_output, re.IGNORECASE | re.DOTALL)
            if match:
                candidate = match.group(match.lastindex or 1)
                normalized = self.normalize_answer(candidate)
                if normalized:
                    return normalized

        if self.enable_fallback_numeric:
            fallback = self._extract_last_numeric_token(llm_output)
            if fallback:
                return self.normalize_answer(fallback)
        return None

    def add_pattern(self, pattern: str):
        """Add a custom pattern to the strategy."""
        if pattern not in self.patterns:
            self.patterns.append(pattern)

    def _extract_last_numeric_token(self, text: str) -> Optional[str]:
        """抓取最後出現的數值/分數當作備援答案。"""
        matches = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
        if matches:
            return matches[-1]
        return None

class BoxExtractionStrategy(EvaluationStrategy):
    """Strategy that extracts answers only from LaTeX-style box/boxed content."""

    DEFAULT_PATTERNS = [
        r"\\{1,2}box\{(?P<brace>.*)\}",
        r"\\{1,2}boxed\{(?P<brace>.*)\}",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.patterns = self.config.get("patterns", self.DEFAULT_PATTERNS)
        self.case_insensitive = self.config.get("case_insensitive", True)
        self.squeeze_whitespace = self.config.get("squeeze_whitespace", True)

    def get_strategy_name(self) -> str:
        return "box"

    def normalize_answer(self, answer: str) -> str:
        """去除 latex 包裝、空白並可選擇補零。"""
        text = str(answer).strip()
        text = re.sub(r"\\{1,2}boxed\{([^{}]+)\}", r"\1", text)
        text = re.sub(r"\\{1,2}box\{([^{}]+)\}", r"\1", text)
        text = text.strip("$")
        text = text.replace("\n", "")
        text = re.sub(r"\\left|\\right", "", text)
        if self.squeeze_whitespace:
            text = re.sub(r"\s+", "", text)
        text = text.replace("\\,", "")
        text = text.strip(".，。;；")

        return text.upper() if self.case_insensitive else text

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer from box/boxed formatting only."""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            matches = list(re.finditer(pattern, llm_output, re.DOTALL))
            if not matches:
                continue
            raw = matches[-1].groupdict().get("brace") or matches[-1].group(1)
            extracted = self._extract_balanced_brace(raw)
            if extracted is not None:
                return self.normalize_answer(extracted)
        return None

    def _extract_balanced_brace(self, text: str) -> Optional[str]:
        """嘗試從 brace 內容中取出第一層平衡大括號的內文，允許巢狀。"""
        if text is None:
            return None
        depth = 0
        buf = []
        started = False
        for ch in text:
            if ch in ("{", "｛"):
                depth += 1
                started = True
                if depth == 1:
                    continue
            if ch in ("}", "｝"):
                depth -= 1
                if depth == 0:
                    break
            if started:
                buf.append(ch)
            if depth < 0:
                break
        return "".join(buf) if buf else None

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


class MathEvaluationStrategy(EvaluationStrategy):
    """數學專用策略：支援 boxed 或 ANSWER/pattern 模式，不影響原始 pattern/box。"""

    DEFAULT_NUMERIC_PATTERNS = [
        r"[Ff]inal\s*[Aa]nswer\s*[:：]\s*([^\n]+)",
        r"[Aa]nswer\s*[:：]\s*([^\n]+)",
    ]
    BOX_PATTERNS = [
        r"\\{1,2}box\{(?P<brace>.*)\}",
        r"\\{1,2}boxed\{(?P<brace>.*)\}",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = self.config
        self.mode = cfg.get("mode", "pattern")  # pattern | box
        self.numeric_patterns = cfg.get("numeric_patterns", self.DEFAULT_NUMERIC_PATTERNS)
        self.box_patterns = cfg.get("box_patterns", self.BOX_PATTERNS)
        self.enable_fallback_numeric = cfg.get("enable_fallback_numeric", True)
        self.case_insensitive = cfg.get("case_insensitive", True)
        self.squeeze_whitespace = cfg.get("squeeze_whitespace", True)

    def get_strategy_name(self) -> str:
        return "math"

    def normalize_answer(self, answer: str) -> str:
        """對齊 lighteval math_normalizer 的邏輯，轉成可比較的 canonical 字串。"""
        text = self._math_normalize(answer)
        # 保留原有行為：可選大小寫折疊與去空白
        if self.squeeze_whitespace:
            text = re.sub(r"\s+", "", text)
        text = text.replace("\\,", "")
        text = text.strip(".，。;；")
        text = re.sub(r"^(FINAL\\s+ANSWER|FINAL\\s*ANS|ANSWER)\\s*[:：]\\s*", "", text, flags=re.IGNORECASE)
        # 降整體大小寫，但再把 latex 指令轉回小寫，避免 \\FRAC 之類破壞解析
        text = text.upper() if self.case_insensitive else text
        text = self._lower_latex_commands(text)
        text = self._ensure_sqrt_braces(text)
        text = self._strip_degrees(text)
        text = self._strip_text_wrapper(text)
        text = self._strip_parenthesized_single(text)
        text = self._strip_leading_trailing_stars(text)
        return self._maybe_sort_numeric_list(text)

    def _maybe_sort_numeric_list(self, text: str) -> str:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) <= 1:
            return text
        numeric_re = re.compile(r"^-?\\d+(?:/\\d+)?(?:\\.\\d+)?$")
        if all(numeric_re.match(p) for p in parts):
            try:
                parts = sorted(parts, key=lambda x: float(eval(x.replace("/", "/1")) if "/" in x else x))
            except Exception:
                parts = sorted(parts)
            return ",".join(parts)
        return text

    def _extract_box(self, llm_output: str) -> Optional[str]:
        """抽取 box/boxed 內容：先抓，再 normalize。"""
        # 1) \boxed{...}
        boxed = self._extract_braced_content(llm_output, r"\boxed{")
        if boxed is not None and boxed != "None":
            return self.normalize_answer(boxed)

        # 2) \box{...}
        box = self._extract_braced_content(llm_output, r"\box{")
        if box is not None and box != "None":
            return self.normalize_answer(box)

        return None

    def _extract_braced_content(self, text: str, prefix: str) -> Optional[str]:
        """模仿 MathRuler extract_boxed_content：從最後一次前綴開始，掃到成對右括號。"""
        if not text or not prefix:
            return None
        start = text.rfind(prefix)
        if start == -1:
            return None
        content = text[start + len(prefix) :]
        depth = 0
        end_pos = -1
        for i, ch in enumerate(content):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == -1:
                end_pos = i
                break
        if end_pos != -1:
            return content[:end_pos].strip()
        return None

    def _extract_numeric(self, llm_output: str) -> Optional[str]:
        """以最後出現的 ANSWER/Final Answer 為主，並清掉 Markdown 標記。"""
        for pattern in self.numeric_patterns:
            matches = list(re.finditer(pattern, llm_output, re.IGNORECASE | re.DOTALL))
            if not matches:
                continue
            match = matches[-1]
            candidate = match.group(match.lastindex or 1)
            candidate = re.sub(r"^\*+|\*+$", "", candidate).strip()
            normalized = self.normalize_answer(candidate)
            if normalized:
                return normalized
        return None

    def _extract_fallback_number(self, llm_output: str) -> Optional[str]:
        matches = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", llm_output)
        if matches:
            return self.normalize_answer(matches[-1])
        return None

    def extract_answer(self, llm_output: str) -> Optional[str]:
        if not self.validate_output(llm_output):
            return None

        if self.mode == "box":
            return self._extract_box(llm_output)

        # pattern 模式：先 ANSWER 前綴，後 boxed，最後 fallback 數值
        numeric = self._extract_numeric(llm_output)
        if numeric:
            return numeric

        boxed = self._extract_box(llm_output)
        if boxed:
            return boxed

        if self.enable_fallback_numeric:
            return self._extract_fallback_number(llm_output)

        return None

    def add_pattern(self, pattern: str):
        if pattern not in self.numeric_patterns:
            self.numeric_patterns.append(pattern)

    def set_mode(self, mode: str):
        if mode in ("box", "pattern"):
            self.mode = mode

    def is_correct(self, predicted: Optional[str], gold: Optional[str]) -> bool:
        """數學答案等價判斷：優先數學正規化後的數值/結構等價。"""
        if predicted is None or gold is None:
            return False
        norm_pred = self.normalize_answer(predicted)
        norm_gold = self.normalize_answer(gold)
        if norm_pred == norm_gold:
            return True
        if self._sympy_equal(norm_pred, norm_gold):
            return True
        return False

    # --- 內部工具：貼近 lighteval math_normalizer ---
    def _math_normalize(self, text: str) -> str:
        """輕量移植自 lighteval math_normalizer，處理 boxed、分數、sqrt 等格式。"""
        if text is None:
            return ""
        text = str(text)

        def _remove_boxed(txt: Optional[str]) -> str:
            if txt is None:
                return ""
            try:
                if "\\boxed " in txt and txt.startswith("\\boxed "):
                    return txt[len("\\boxed ") :]
                if txt.startswith("\\boxed{") and txt.endswith("}"):
                    return txt[len("\\boxed{") : -1]
                return txt
            except Exception:
                return txt

        def _last_boxed_only_string(txt: str) -> Optional[str]:
            idx = txt.rfind("\\boxed")
            if idx < 0:
                idx = txt.rfind("\\fbox")
                if idx < 0:
                    return None
            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(txt):
                if txt[i] == "{":
                    num_left_braces_open += 1
                if txt[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1
            if right_brace_idx is None:
                return None
            return txt[idx : right_brace_idx + 1]

        def _fix_fracs(txt: str) -> str:
            substrs = txt.split("\\frac")
            new_str = substrs[0]
            if len(substrs) > 1:
                substrs = substrs[1:]
                for substr in substrs:
                    new_str += "\\frac"
                    if not substr:
                        continue
                    if substr[0] == "{":
                        new_str += substr
                    else:
                        if len(substr) < 2:
                            new_str += substr
                            continue
                        a = substr[0]
                        b = substr[1]
                        if b != "{":
                            post_substr = substr[2:] if len(substr) > 2 else ""
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            post_substr = substr[2:] if len(substr) > 2 else ""
                            new_str += "{" + a + "}" + b + post_substr
            return new_str

        def _fix_a_slash_b(txt: str) -> str:
            if len(txt.split("/")) != 2:
                return txt
            a_str, b_str = txt.split("/")
            try:
                a = int(a_str)
                b = int(b_str)
                if txt == f"{a}/{b}":
                    return f"\\frac{{{a}}}{{{b}}}"
            except Exception:
                pass
            return txt

        def _remove_right_units(txt: str) -> str:
            if "\\text{ " in txt:
                return txt.split("\\text{ ")[0].rstrip()
            return txt

        def _fix_sqrt(txt: str) -> str:
            if "\\sqrt" not in txt:
                return txt
            splits = txt.split("\\sqrt")
            new_string = splits[0]
            for split in splits[1:]:
                if not split:
                    continue
                if split[0] != "{":
                    a = split[0]
                    new_substr = "\\sqrt{" + a + "}" + split[1:]
                else:
                    new_substr = "\\sqrt" + split
                new_string += new_substr
            return new_string

        # 取最後 boxed 內容
        boxed_only = _last_boxed_only_string(text)
        text = _remove_boxed(boxed_only) if boxed_only else text

        replacements = [
            ("\n", ""),
            ("\\!", ""),
            ("\\\\", "\\"),
            ("tfrac", "frac"),
            ("dfrac", "frac"),
            ("\\left", ""),
            ("\\right", ""),
            ("^{\\circ}", ""),
            ("^\\circ", ""),
            ("\\$", ""),
            ("\\(", ""),
            ("\\)", ""),
            ("\\[", ""),
            ("\\]", ""),
        ]
        for src, dst in replacements:
            text = text.replace(src, dst)

        text = _remove_right_units(text)

        for src, dst in [("\\%", ""), (r"\%", ""), (" .", " 0."), ("{.", "{0.")]:
            text = text.replace(src, dst)

        if not text:
            return text
        if text[0] == ".":
            text = "0" + text

        if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
            text = text.split("=")[1]

        text = _fix_sqrt(text)
        text = text.replace(" ", "")
        text = _fix_fracs(text)
        if text == "0.5":
            text = "\\frac{1}{2}"
        text = _fix_a_slash_b(text)
        # 補救無花括號分數 e.g. \\FRAC65 -> \\frac{6}{5}, 也處理變數字母
        text = re.sub(r"\\frac\\s*{\\s*([^{}]+)\\s*}\\s*{\\s*([^{}]+)\\s*}", r"\\frac{\1}{\2}", text, flags=re.IGNORECASE)
        text = re.sub(r"\\frac\\s*([0-9A-Za-z]+)\\s*([0-9A-Za-z]+)", r"\\frac{\1}{\2}", text, flags=re.IGNORECASE)
        # 處理 unicode/pi/sqrt 等常見符號與多餘星號
        text = self._normalize_unicode_symbols(text)
        text = self._strip_leading_trailing_stars(text)
        # 把 latex 指令統一小寫，避免 \\FRAC 之類
        text = self._lower_latex_commands(text)
        text = self._normalize_intervals(text)
        text = self._strip_base_suffix(text)
        text = self._strip_mbox_units(text)
        text = self._ensure_sqrt_braces(text)
        text = self._strip_degrees(text)
        return text

    def _latex_frac_to_slash(self, text: str) -> str:
        """將 \\frac{a}{b} 轉成 (a)/(b) 方便解析。"""
        text = re.sub(r"\\frac\s*{\s*([^{}]+)\s*}\s*{\s*([^{}]+)\s*}", r"(\1)/(\2)", text, flags=re.IGNORECASE)
        text = re.sub(r"\\frac\s*([0-9A-Za-z]+)\s*([0-9A-Za-z]+)", r"(\1)/(\2)", text, flags=re.IGNORECASE)
        return text

    def _lower_latex_commands(self, text: str) -> str:
        """將 \\COMMAND 類 latex 指令轉成小寫（不影響後面實際內容與變數）。"""
        return re.sub(r"\\([A-Z]+)", lambda m: "\\" + m.group(1).lower(), text)

    def _strip_degrees(self, text: str) -> str:
        """移除角度符號 ^\\circ（忽略大小寫）。"""
        return re.sub(r"\^\\circ", "", text, flags=re.IGNORECASE)

    def _strip_text_wrapper(self, text: str) -> str:
        """移除 \\text{...} 包裝。"""
        return re.sub(r"\\text\{([^{}]+)\}", r"\1", text, flags=re.IGNORECASE)

    def _strip_parenthesized_single(self, text: str) -> str:
        """將 (X) 簡化為 X（用於 \text{(B)} 類答案）。"""
        return re.sub(r"^\(([^()])\)$", r"\1", text)

    def _strip_leading_trailing_stars(self, text: str) -> str:
        """去掉前後多餘星號（markdown 粗體殘留）。"""
        return re.sub(r"^\*+|\*+$", "", text)

    def _normalize_unicode_symbols(self, text: str) -> str:
        """簡單處理 unicode 常見數學符號。"""
        # π/Π -> \pi
        text = text.replace("π", "\\pi").replace("Π", "\\pi")
        # Unicode sqrt
        text = re.sub(r"√\s*([A-Za-z0-9\+\-]+)", r"\\sqrt{\1}", text)
        # ∞ -> \infty
        text = text.replace("∞", "\\infty")
        return text

    def _ensure_sqrt_braces(self, text: str) -> str:
        """確保 \\sqrt 後面有花括號，便於後續解析。"""
        return re.sub(r"\\sqrt\s*([A-Za-z0-9]+)", r"\\sqrt{\1}", text)

    def _normalize_intervals(self, text: str) -> str:
        """將區間符號的 -∞ / -INFTY 變成 \\infty，便於解析。"""
        text = re.sub(r"-\s*(\\INFTY|INFTY)", r"-\\infty", text, flags=re.IGNORECASE)
        return text

    def _strip_base_suffix(self, text: str) -> str:
        """移除簡單進位後綴，如 2516_8 -> 2516。"""
        return re.sub(r"(_[0-9]+)$", "", text)

    def _strip_mbox_units(self, text: str) -> str:
        """移除 \\mbox{...} 單位及緊跟的冪次，對齊 lighteval 單位清理。"""
        return re.sub(r"\\MBOX\{[^{}]*\}(?:\^\d+)?", "", text, flags=re.IGNORECASE)

    def _safe_eval_numeric(self, text: str):
        """已改用 sympy，比較時直接走 _sympy_equal，不再用簡易 AST。"""
        return None

    def _sympy_equal(self, a: str, b: str) -> bool:
        """若環境裝有 latex2sympy2_extended/sympy，優先用其做等價判斷；否則返回 False。"""
        if _HAS_LATEX2SYMPY_EXT:
            try:
                la = self._to_sympy_expr_ext(a)
                lb = self._to_sympy_expr_ext(b)
                if la is not None and lb is not None:
                    if isinstance(la, _sp.Set) and not isinstance(lb, _sp.Set):
                        return lb in la
                    if isinstance(lb, _sp.Set) and not isinstance(la, _sp.Set):
                        return la in lb
                    if isinstance(la, _sp.Set) and isinstance(lb, _sp.Set):
                        return la == lb
                    # 非集合直接等號或差為零
                    try:
                        if la == lb:
                            return True
                        return _sp.simplify(la - lb) == 0
                    except Exception:
                        return False
                # 若只解析到其中一個，嘗試直接比較轉字串後的簡化
                if la is not None and lb is None:
                    lb = self._to_sympy_expr_ext(str(b))
                if lb is not None and la is None:
                    la = self._to_sympy_expr_ext(str(a))
                if la is not None and lb is not None:
                    return _sp.simplify(la - lb) == 0
            except Exception:
                pass  # fallback below

        try:
            import sympy as sp
        except Exception:
            return False

        def _to_expr(txt: str):
            safe_txt = self._latex_frac_to_slash(txt)
            safe_txt = safe_txt.replace("^", "**")
            safe_txt = safe_txt.replace("\\pi", "pi")
            safe_txt = re.sub(r"\\sqrt\\s*{([^{}]+)}", r"sqrt(\1)", safe_txt)
            safe_txt = re.sub(r"\\sqrt\\s*([A-Za-z0-9]+)", r"sqrt(\1)", safe_txt)
            safe_txt = safe_txt.replace("\\sqrt", "sqrt")
            safe_txt = re.sub(r"sqrt\\s*([A-Za-z0-9]+)", r"sqrt(\\1)", safe_txt)
            safe_txt = re.sub(r"(?<=\d)(?=[A-Za-z\(])", "*", safe_txt)
            safe_txt = re.sub(r"(?<=[A-Za-z\)])(?=\d)", "*", safe_txt)
            safe_txt = safe_txt.replace("{", "(").replace("}", ")")
            try:
                return sp.sympify(safe_txt)
            except Exception:
                return None

        expr_a = _to_expr(a)
        expr_b = _to_expr(b)
        if expr_a is None or expr_b is None:
            return False
        try:
            return sp.simplify(expr_a - expr_b) == 0
        except Exception:
            try:
                return sp.simplify(expr_a) == sp.simplify(expr_b)
            except Exception:
                return False

    def _to_sympy_expr_ext(self, txt: str):
        """使用 latex2sympy2_extended 將 latex 轉成 sympy expr。"""
        if not _HAS_LATEX2SYMPY_EXT:
            return None
        try:
            config = _NormConfig()
            pre_norm = self._math_normalize(txt)
            normalized = _normalize_latex(pre_norm, config)
            # 特處理 \pm -> 生成 FiniteSet 兩個分支
            if r"\pm" in normalized:
                # 直接讓 latex2sympy2_extended 解析，會回 FiniteSet
                return _latex2sympy(normalized, normalization_config=config)
            return _latex2sympy(normalized, normalization_config=config)
        except Exception:
            return None

    def _commutative_ast_equal(self, a: str, b: str) -> bool:
        """沒有 sympy 時的 fallback：對加法/乘法做排序後比較 AST。"""
        import ast
        from fractions import Fraction

        def canonical(node):
            if isinstance(node, ast.Expression):
                return canonical(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return ("num", Fraction(node.value).limit_denominator())
            if isinstance(node, ast.Name):
                return ("sym", node.id)
            if isinstance(node, ast.UnaryOp):
                operand = canonical(node.operand)
                if isinstance(node.op, ast.USub):
                    return ("mul", ("num", Fraction(-1)), operand)
                if isinstance(node.op, ast.UAdd):
                    return operand
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    left = canonical(node.left)
                    right = canonical(node.right)
                    if isinstance(node.op, ast.Sub):
                        right = ("mul", ("num", Fraction(-1)), right)
                    return ("add",) + tuple(sorted(flatten_add([left, right]), key=str))
                if isinstance(node.op, ast.Mult):
                    left = canonical(node.left)
                    right = canonical(node.right)
                    return ("mul",) + tuple(sorted(flatten_mul([left, right]), key=str))
                if isinstance(node.op, ast.Div):
                    left = canonical(node.left)
                    right = canonical(node.right)
                    inv = ("pow", right, ("num", Fraction(-1)))
                    return ("mul",) + tuple(sorted(flatten_mul([left, inv]), key=str))
                if isinstance(node.op, ast.Pow):
                    left = canonical(node.left)
                    right = canonical(node.right)
                    return ("pow", left, right)
            return ("raw", ast.dump(node))

        def flatten_add(items):
            out = []
            for i in items:
                if isinstance(i, tuple) and i and i[0] == "add":
                    out.extend(i[1:])
                else:
                    out.append(i)
            return out

        def flatten_mul(items):
            out = []
            for i in items:
                if isinstance(i, tuple) and i and i[0] == "mul":
                    out.extend(i[1:])
                else:
                    out.append(i)
            return out

        def to_canonical(expr: str):
            expr = self._latex_frac_to_slash(expr).replace("^", "**")
            # 插入隱含乘號：3x -> 3*x, (a)b -> (a)*b
            expr = re.sub(r"(?<=\d)(?=[A-Za-z\(])", "*", expr)
            expr = re.sub(r"(?<=[A-Za-z\)])(?=\d)", "*", expr)
            expr = expr.replace("\\pi", "pi")
            expr = re.sub(r"\\sqrt\\s*{([^{}]+)}", r"sqrt(\1)", expr)
            expr = re.sub(r"\\sqrt\\s*([A-Za-z0-9]+)", r"sqrt(\1)", expr)
            expr = expr.replace("\\sqrt", "sqrt")
            expr = re.sub(r"sqrt\\s*([A-Za-z0-9]+)", r"sqrt(\\1)", expr)
            expr = expr.replace("{", "(").replace("}", ")")
            try:
                tree = ast.parse(expr, mode="eval")
            except Exception:
                return None
            return canonical(tree)

        ca = to_canonical(a)
        cb = to_canonical(b)
        return ca is not None and cb is not None and ca == cb
    
    
class EvaluationStrategyFactory:
    """Factory class for creating evaluation strategy instances."""

    _registry: Dict[str, Type[EvaluationStrategy]] = {
        "pattern": PatternMatchingStrategy,
        "box": BoxExtractionStrategy,
        "custom_regex": CustomRegexStrategy,
        "math": MathEvaluationStrategy,
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
