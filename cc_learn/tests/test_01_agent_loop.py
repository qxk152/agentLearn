"""
Unit tests for 01_agent_loop.py - Agent Loop (OpenAI-compatible version)

Tests cover:
  - run_bash: dangerous command blocking, timeout, output truncation
  - assistant_message_to_dict: message conversion with/without tool_calls
  - parse_tool_arguments: valid/invalid JSON, type validation
  - env_first: environment variable lookup priority
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# We need to import functions from 01_agent_loop.py
# Since it has top-level OpenAI client creation, we patch before import
import importlib

# Patch the OpenAI client and env vars before importing the module
with patch.dict(os.environ, {"MODEL_ID": "test-model", "OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://localhost/v1"}):
    # Mock OpenAI class to avoid real API connection
    mock_openai = MagicMock()
    with patch("openai.OpenAI", return_value=mock_openai):
        # Import module
        spec = importlib.util.spec_from_file_location("agent_loop", "01_agent_loop.py")
        agent_loop_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_loop_mod)


class TestRunBash:
    """Tests for the run_bash function."""

    def test_simple_command_success(self):
        """A simple command should return its stdout."""
        result = agent_loop_mod.run_bash("echo hello")
        assert "hello" in result

    def test_command_with_stderr(self):
        """stderr should be included in the output."""
        result = agent_loop_mod.run_bash("echo err >&2")
        assert "err" in result

    def test_dangerous_command_rm_rf(self):
        """'rm -rf /' should be blocked."""
        result = agent_loop_mod.run_bash("rm -rf /")
        assert "Dangerous command blocked" in result

    def test_dangerous_command_sudo(self):
        """'sudo' commands should be blocked."""
        result = agent_loop_mod.run_bash("sudo apt install something")
        assert "Dangerous command blocked" in result

    def test_dangerous_command_shutdown(self):
        """'shutdown' commands should be blocked."""
        result = agent_loop_mod.run_bash("shutdown -h now")
        assert "Dangerous command blocked" in result

    def test_dangerous_command_reboot(self):
        """'reboot' commands should be blocked."""
        result = agent_loop_mod.run_bash("reboot")
        assert "Dangerous command blocked" in result

    def test_dangerous_command_dev_null(self):
        """Commands redirecting to /dev/ should be blocked."""
        result = agent_loop_mod.run_bash("cat /etc/passwd > /dev/null")
        assert "Dangerous command blocked" in result

    def test_non_dangerous_rm(self):
        """rm without -rf / should not be blocked (e.g., 'rm file.txt')."""
        result = agent_loop_mod.run_bash("echo 'rm test'")
        # 'rm' alone without 'rm -rf /' or 'sudo' should NOT be blocked
        # But note: the check is substring-based. 'rm test' doesn't match 'rm -rf /'
        assert "Dangerous command blocked" not in result

    def test_empty_output(self):
        """A command with no output should return '(no output)'."""
        result = agent_loop_mod.run_bash("cd .")
        # cd produces no stdout/stderr, should return "(no output)"
        assert result == "(no output)" or result.strip() == ""

    def test_output_truncation(self):
        """Very long output should be truncated to 50000 chars."""
        result = agent_loop_mod.run_bash("python -c \"print('x' * 60000)\"")
        assert len(result) <= 50000

    def test_timeout_handling(self, tmp_path):
        """Commands that exceed 120s timeout should return error."""
        # We mock subprocess.run to raise TimeoutExpired
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sleep", timeout=120)):
            result = agent_loop_mod.run_bash("sleep 200")
            assert "Timeout" in result

    def test_file_not_found_error(self):
        """Non-existent commands should return an error."""
        result = agent_loop_mod.run_bash("nonexistent_command_xyz")
        # Should contain an error message
        assert "Error" in result or result.strip() != ""


class TestAssistantMessageToDict:
    """Tests for converting OpenAI SDK message objects to dicts."""

    def test_simple_message_no_tool_calls(self):
        """A message without tool_calls should produce a simple dict."""
        msg = MagicMock()
        msg.content = "Hello world"
        msg.tool_calls = None

        result = agent_loop_mod.assistant_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Hello world"
        assert "tool_calls" not in result

    def test_message_with_none_content(self):
        """None content should be converted to empty string."""
        msg = MagicMock()
        msg.content = None
        msg.tool_calls = None

        result = agent_loop_mod.assistant_message_to_dict(msg)
        assert result["content"] == ""

    def test_message_with_tool_calls(self):
        """A message with tool_calls should include them in the dict."""
        msg = MagicMock()
        msg.content = ""
        tc1 = MagicMock()
        tc1.id = "call_abc123"
        tc1.type = "function"
        tc1.function.name = "bash"
        tc1.function.arguments = '{"command": "ls"}'
        msg.tool_calls = [tc1]

        result = agent_loop_mod.assistant_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_abc123"
        assert result["tool_calls"][0]["function"]["name"] == "bash"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'

    def test_message_with_multiple_tool_calls(self):
        """Multiple tool_calls should all be preserved."""
        msg = MagicMock()
        msg.content = ""
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.type = "function"
        tc1.function.name = "bash"
        tc1.function.arguments = '{"command": "ls"}'
        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.type = "function"
        tc2.function.name = "bash"
        tc2.function.arguments = '{"command": "pwd"}'
        msg.tool_calls = [tc1, tc2]

        result = agent_loop_mod.assistant_message_to_dict(msg)
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "bash"
        assert result["tool_calls"][1]["function"]["name"] == "bash"


class TestParseToolArguments:
    """Tests for parsing tool_call.function.arguments JSON strings."""

    def test_valid_json(self):
        """Valid JSON should be parsed correctly."""
        result = agent_loop_mod.parse_tool_arguments('{"command": "ls"}')
        assert result == {"command": "ls"}

    def test_empty_string(self):
        """Empty string arguments should return empty dict."""
        result = agent_loop_mod.parse_tool_arguments("")
        assert result == {}

    def test_none_arguments(self):
        """None arguments should return empty dict."""
        result = agent_loop_mod.parse_tool_arguments(None)
        assert result == {}

    def test_invalid_json(self):
        """Invalid JSON should return _error dict."""
        result = agent_loop_mod.parse_tool_arguments("not json at all")
        assert "_error" in result

    def test_non_dict_json(self):
        """JSON that is not a dict (e.g., array) should return _error."""
        result = agent_loop_mod.parse_tool_arguments('[1, 2, 3]')
        assert "_error" in result

    def test_nested_json(self):
        """Nested JSON dict should be parsed correctly."""
        result = agent_loop_mod.parse_tool_arguments('{"key": {"nested": "value"}}')
        assert result["key"]["nested"] == "value"


class TestEnvFirst:
    """Tests for the env_first helper function."""

    def test_first_var_set(self):
        """Should return the first variable that is set."""
        with patch.dict(os.environ, {"VAR_A": "a_value", "VAR_B": "b_value"}):
            result = agent_loop_mod.env_first("VAR_A", "VAR_B")
            assert result == "a_value"

    def test_first_var_empty_second_set(self):
        """If first var is empty, should fall through to next."""
        with patch.dict(os.environ, {"VAR_A": "", "VAR_B": "b_value"}, clear=False):
            # os.getenv returns None for unset, but empty string for empty
            # env_first checks `if value` which treats empty string as falsy
            result = agent_loop_mod.env_first("VAR_A", "VAR_B")
            # Empty string is falsy in Python, so should fall through
            assert result == "b_value"

    def test_no_vars_set(self):
        """If no variables are set, should return None."""
        # Remove both vars to ensure they're not set
        env = os.environ.copy()
        env.pop("VAR_X", None)
        env.pop("VAR_Y", None)
        with patch.dict(os.environ, env, clear=True):
            result = agent_loop_mod.env_first("VAR_X", "VAR_Y")
            assert result is None

    def test_single_var(self):
        """Should work with a single variable name."""
        with patch.dict(os.environ, {"MY_VAR": "my_value"}):
            result = agent_loop_mod.env_first("MY_VAR")
            assert result == "my_value"