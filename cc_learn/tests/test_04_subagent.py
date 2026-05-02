"""
Unit tests for 04_subagent.py - Subagents (OpenAI-compatible version)

Tests cover:
  - run_subagent: subagent execution (mocked LLM)
  - call_base_tool: tool dispatch excluding task
  - PARENT_TOOLS / CHILD_TOOLS: tool definitions
  - assistant_message_to_dict / parse_tool_arguments: shared utilities
"""

import json
import os
from unittest.mock import MagicMock, patch, PropertyMock

import importlib
import pytest

with patch.dict(os.environ, {"MODEL_ID": "test-model", "OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://localhost/v1"}):
    mock_openai = MagicMock()
    with patch("openai.OpenAI", return_value=mock_openai):
        spec = importlib.util.spec_from_file_location("subagent_mod", "04_subagent.py")
        subagent_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(subagent_mod)


class TestToolDefinitions:
    """Tests for PARENT_TOOLS and CHILD_TOOLS definitions."""

    def test_child_tools_no_task(self):
        """Child tools should NOT include the 'task' tool."""
        child_names = [t["function"]["name"] for t in subagent_mod.CHILD_TOOLS]
        assert "task" not in child_names
        assert "bash" in child_names
        assert "read_file" in child_names

    def test_parent_tools_include_task(self):
        """Parent tools should include the 'task' tool."""
        parent_names = [t["function"]["name"] for t in subagent_mod.PARENT_TOOLS]
        assert "task" in parent_names
        assert "bash" in parent_names

    def test_parent_tools_extend_child_tools(self):
        """Parent tools should be child tools + task."""
        parent_names = [t["function"]["name"] for t in subagent_mod.PARENT_TOOLS]
        child_names = [t["function"]["name"] for t in subagent_mod.CHILD_TOOLS]
        # Parent should contain all child tools + task
        for name in child_names:
            assert name in parent_names

    def test_task_tool_has_prompt_param(self):
        """Task tool should have 'prompt' as required parameter."""
        task_def = None
        for t in subagent_mod.PARENT_TOOLS:
            if t["function"]["name"] == "task":
                task_def = t
                break
        assert task_def is not None
        assert "prompt" in task_def["function"]["parameters"]["required"]


class TestCallBaseTool:
    """Tests for call_base_tool (excludes task tool)."""

    def test_bash_via_base_tool(self):
        """bash should be callable via call_base_tool."""
        result = subagent_mod.call_base_tool("bash", {"command": "echo test"})
        assert "test" in result

    def test_unknown_tool_via_base_tool(self):
        """Unknown tool should return error."""
        result = subagent_mod.call_base_tool("unknown_tool", {})
        assert "Unknown tool" in result

    def test_error_args_via_base_tool(self):
        """Args with _error should return the error directly."""
        result = subagent_mod.call_base_tool("bash", {"_error": "bad"})
        assert "bad" in result

    def test_missing_required_arg_via_base_tool(self):
        """Missing required argument should return error."""
        result = subagent_mod.call_base_tool("bash", {})
        assert "Missing required argument" in result


class TestRunSubagent:
    """Tests for run_subagent with mocked LLM calls."""

    def test_subagent_returns_summary(self):
        """Subagent should return a text summary when LLM has no tool_calls."""
        mock_msg = MagicMock()
        mock_msg.content = "Summary of findings"
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        subagent_mod.client.chat.completions.create = MagicMock(return_value=mock_response)

        result = subagent_mod.run_subagent("Find all Python files")
        assert result == "Summary of findings"

    def test_subagent_with_tool_calls(self):
        """Subagent should execute tool calls then return final message."""
        # First call: LLM wants to use bash
        mock_msg1 = MagicMock()
        mock_msg1.content = ""
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function.name = "bash"
        tc1.function.arguments = '{"command": "echo hello"}'
        mock_msg1.tool_calls = [tc1]

        # Second call: LLM returns summary (no more tool calls)
        mock_msg2 = MagicMock()
        mock_msg2.content = "Found 3 Python files"
        mock_msg2.tool_calls = None

        mock_choice1 = MagicMock()
        mock_choice1.message = mock_msg1
        mock_choice2 = MagicMock()
        mock_choice2.message = mock_msg2

        mock_response1 = MagicMock()
        mock_response1.choices = [mock_choice1]
        mock_response2 = MagicMock()
        mock_response2.choices = [mock_choice2]

        subagent_mod.client.chat.completions.create = MagicMock(side_effect=[mock_response1, mock_response2])

        result = subagent_mod.run_subagent("Find Python files")
        assert result == "Found 3 Python files"

    def test_subagent_no_content(self):
        """Subagent with empty content should return '(no summary)'."""
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        subagent_mod.client.chat.completions.create = MagicMock(return_value=mock_response)

        result = subagent_mod.run_subagent("test prompt")
        assert "(no summary)" in result


class TestSafePathSubagent:
    """Tests for safe_path in subagent module."""

    def test_path_escape_rejected(self, tmp_path):
        """Path escaping workspace should be rejected."""
        with patch.object(subagent_mod, "WORKDIR", tmp_path):
            with pytest.raises(ValueError, match="escapes workspace"):
                subagent_mod.safe_path("../../outside")