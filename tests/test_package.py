"""
tests/test_package.py

套件層級的基本功能測試，涵蓋：
- 版本資訊的可存取性
- get_info() 不拋出例外且回傳結構正確
- CLI --version 指令正常退出
- pyproject.toml 與 __init__.py 版本號一致性
"""

import importlib.metadata
import subprocess
import sys

import pytest


class TestVersionConsistency:
    """版本號一致性測試 — pyproject.toml 與 __init__.py 必須同步"""

    def test_module_version_matches_package_metadata(self):
        """__init__.__version__ 必須與 pyproject.toml 的 version 欄位一致"""
        import twinkle_eval

        installed_version = importlib.metadata.version("twinkle-eval")
        assert twinkle_eval.__version__ == installed_version, (
            f"__init__.py __version__='{twinkle_eval.__version__}' "
            f"與 pyproject.toml version='{installed_version}' 不一致"
        )


class TestGetVersion:
    """測試 get_version()"""

    def test_returns_string(self):
        from twinkle_eval import get_version

        assert isinstance(get_version(), str)

    def test_not_empty(self):
        from twinkle_eval import get_version

        assert get_version().strip() != ""

    def test_matches_module_version(self):
        import twinkle_eval

        from twinkle_eval import get_version

        assert get_version() == twinkle_eval.__version__


class TestGetInfo:
    """測試 get_info() — 對應 Issue #10"""

    REQUIRED_KEYS = {"name", "version", "author", "license", "description", "url"}

    def test_does_not_raise(self):
        """get_info() 不得拋出任何例外（修正前會 NameError: __email__ is not defined）"""
        from twinkle_eval import get_info

        # 若這行爆炸，表示 __email__ 尚未定義
        info = get_info()
        assert info is not None

    def test_returns_dict(self):
        from twinkle_eval import get_info

        assert isinstance(get_info(), dict)

    def test_required_keys_present(self):
        from twinkle_eval import get_info

        info = get_info()
        missing = self.REQUIRED_KEYS - info.keys()
        assert not missing, f"get_info() 缺少必要欄位: {missing}"

    def test_values_are_non_empty_strings(self):
        from twinkle_eval import get_info

        info = get_info()
        for key in self.REQUIRED_KEYS:
            assert isinstance(info[key], str) and info[key].strip(), (
                f"get_info()['{key}'] 不應為空字串"
            )

    def test_version_matches_module(self):
        import twinkle_eval

        from twinkle_eval import get_info

        assert get_info()["version"] == twinkle_eval.__version__


class TestCLIVersion:
    """測試 twinkle-eval --version 指令正常退出"""

    def test_version_flag_exit_zero(self):
        """--version 必須以 exit code 0 退出，不得拋出任何例外"""
        result = subprocess.run(
            [sys.executable, "-m", "twinkle_eval.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"--version 回傳非零 exit code {result.returncode}\n"
            f"stderr: {result.stderr}"
        )

    def test_version_output_contains_version_string(self):
        import twinkle_eval

        result = subprocess.run(
            [sys.executable, "-m", "twinkle_eval.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert twinkle_eval.__version__ in result.stdout, (
            f"--version 輸出中找不到版本號 {twinkle_eval.__version__}\n"
            f"stdout: {result.stdout}"
        )
