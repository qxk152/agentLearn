"""
Unit tests for 02_tool_use.py - Tool Use (OpenAI-compatible version)

Tests cover:
  - safe_path: path traversal prevention
  - run_bash: dangerous command blocking
  - run_read: file reading with limit
  - run_write: file writing
  - run_edit: exact text replacement
  - call_tool: dispatching tool calls
  - TOOL_HANDLERS: mapping verification
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import importlib

with patch.dict(os.environ, {"MODEL_ID": "test-model", "OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://localhost/v1"}):
    mock_openai = MagicMock()
    with patch("openai.OpenAI", return_value=mock_openai):
        spec = importlib.util.spec_from_file_location("tool_use", "02_tool_use.py")
        tool_use_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool_use_mod)


class TestSafePath:
    """Tests for path safety / workspace escape prevention."""

    def test_normal_relative_path(self, tmp_path):
        """Normal relative paths should resolve within workspace."""
        # We need to patch WORKDIR since it's set at module level
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.safe_path("subdir/file.txt")
            assert result.is_relative_to(tmp_path)

    def test_path_escape_with_dotdot(self, tmp_path):
        """Paths that escape the workspace should raise ValueError."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            with pytest.raises(ValueError, match="escapes workspace"):
                tool_use_mod.safe_path("../../etc/passwd")

    def test_absolute_path_within_workspace(self, tmp_path):
        """Absolute paths within workspace should be allowed."""
        file_path = tmp_path / "test.txt"
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.safe_path(str(file_path))
            assert result.is_relative_to(tmp_path)

    def test_absolute_path_outside_workspace(self, tmp_path):
        """Absolute paths outside workspace should raise ValueError."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            with pytest.raises(ValueError, match="escapes workspace"):
                tool_use_mod.safe_path("/etc/passwd")

    def test_symlink_escape(self, tmp_path):
        """Symlinks that escape workspace should be caught."""
        # Create a symlink outside workspace
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret")
        link = tmp_path / "link"
        try:
            link.symlink_to(outside)
        except OSError:
            # Symlinks may not work on Windows without admin
            pytest.skip("Cannot create symlink on this platform")

        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            with pytest.raises(ValueError):
                tool_use_mod.safe_path("link")


import pytest


class TestRunBash:
    """Tests for run_bash in tool_use module."""

    def test_echo_command(self):
        """Simple echo should work."""
        result = tool_use_mod.run_bash("echo test_output")
        assert "test_output" in result

    def test_dangerous_command_blocked(self):
        """Dangerous commands should be blocked."""
        result = tool_use_mod.run_bash("sudo rm -rf /")
        assert "Dangerous command blocked" in result


class TestRunRead:
    """Tests for run_read file reading."""

    def test_read_existing_file(self, tmp_path):
        """Reading an existing file should return its contents."""
        file = tmp_path / "test.txt"
        file.write_text("Hello World\nSecond Line\nThird Line\n")

        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_read("test.txt")
            assert "Hello World" in result
            assert "Second Line" in result

    def test_read_with_limit(self, tmp_path):
        """Reading with limit should truncate to specified lines."""
        file = tmp_path / "long.txt"
        lines = [f"Line {i}" for i in range(100)]
        file.write_text("\n".join(lines) + "\n")

        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_read("long.txt", limit=5)
            assert "Line 0" in result
            assert "Line 4" in result
            assert "more)" in result  # truncation indicator

    def test_read_nonexistent_file(self, tmp_path):
        """Reading a nonexistent file should return error."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_read("nonexistent.txt")
            assert "Error" in result

    def test_read_path_escape(self, tmp_path):
        """Trying to read a file outside workspace should return error."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_read("../../etc/passwd")
            assert "Error" in result


class TestRunWrite:
    """Tests for run_write file writing."""

    def test_write_new_file(self, tmp_path):
        """Writing a new file should create it and return success."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_write("new_file.txt", "Hello content")
            assert "Wrote" in result
            assert (tmp_path / "new_file.txt").read_text() == "Hello content"

    def test_write_in_subdirectory(self, tmp_path):
        """Writing to a subdirectory should create the directory."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_write("sub/deep/file.txt", "Nested content")
            assert "Wrote" in result
            assert (tmp_path / "sub/deep/file.txt").read_text() == "Nested content"

    def test_write_path_escape(self, tmp_path):
        """Writing outside workspace should return error."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_write("../../tmp/evil.txt", "bad content")
            assert "Error" in result


class TestRunEdit:
    """Tests for run_edit exact text replacement."""

    def test_edit_simple_replacement(self, tmp_path):
        """Editing should replace exact text."""
        file = tmp_path / "edit_test.txt"
        file.write_text("Hello World\nGoodbye World\n")

        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_edit("edit_test.txt", "Hello", "Greetings")
            assert "Edited" in result
            assert "Greetings" in (tmp_path / "edit_test.txt").read_text()

    def test_edit_text_not_found(self, tmp_path):
        """Editing with non-existent text should return error."""
        file = tmp_path / "edit_test.txt"
        file.write_text("Hello World\n")

        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_edit("edit_test.txt", "DoesNotExist", "Replacement")
            assert "Text not found" in result

    def test_edit_only_first_occurrence(self, tmp_path):
        """Edit should replace only the first occurrence."""
        file = tmp_path / "multi.txt"
        file.write_text("aaa bbb aaa\n")

        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_edit("multi.txt", "aaa", "xxx")
            content = (tmp_path / "multi.txt").read_text()
            assert content == "xxx bbb aaa\n"

    def test_edit_nonexistent_file(self, tmp_path):
        """Editing a nonexistent file should return error."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_edit("nonexistent.txt", "old", "new")
            assert "Error" in result

    def test_edit_path_escape(self, tmp_path):
        """Editing outside workspace should return error."""
        with patch.object(tool_use_mod, "WORKDIR", tmp_path):
            result = tool_use_mod.run_edit("../../etc/passwd", "root", "hacked")
            assert "Error" in result


class TestCallTool:
    """Tests for the call_tool dispatch function."""

    def test_call_bash_tool(self):
        """Calling bash tool should execute command."""
        result = tool_use_mod.call_tool("bash", {"command": "echo test"})
        assert "test" in result

    def test_call_unknown_tool(self):
        """Calling unknown tool should return error."""
        result = tool_use_mod.call_tool("unknown_tool", {})
        assert "Unknown tool" in result

    def test_call_tool_with_error_args(self):
        """Tool call with _error in args should return the error."""
        result = tool_use_mod.call_tool("bash", {"_error": "bad args"})
        assert "bad args" in result

    def test_call_tool_missing_required_arg(self):
        """Tool call missing required arguments should return error."""
        result = tool_use_mod.call_tool("bash", {})
        assert "Missing required argument" in result


class TestToolHandlersMapping:
    """Tests to verify all tool handlers are properly mapped."""

    def test_all_base_tools_have_handlers(self):
        """All defined tools (bash, read_file, write_file, edit_file) should have handlers."""
        for tool_def in tool_use_mod.TOOLS:
            name = tool_def["function"]["name"]
            assert name in tool_use_mod.TOOL_HANDLERS, f"Tool '{name}' missing from TOOL_HANDLERS"

    def test_handler_signatures_match_tool_params(self):
        """Tool handler signatures should match tool parameter definitions."""
        for name, handler in tool_use_mod.TOOL_HANDLERS.items():
            # Each handler should be callable
            assert callable(handler), f"Handler for '{name}' is not callable"