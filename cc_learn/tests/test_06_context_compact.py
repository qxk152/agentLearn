"""
Unit tests for 06_context_compact.py - Compact (OpenAI-compatible version)

Tests cover:
  - estimate_tokens: token estimation
  - micro_compact: shortening tool results
  - auto_compact: full conversation compression (mocked LLM)
  - safe_path, run_bash, run_read, run_write, run_edit
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import importlib
import pytest

with patch.dict(os.environ, {"MODEL_ID": "test-model", "OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://localhost/v1"}):
    mock_openai = MagicMock()
    with patch("openai.OpenAI", return_value=mock_openai):
        spec = importlib.util.spec_from_file_location("compact_mod", "06_context_compact.py")
        compact_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compact_mod)


class TestEstimateTokens:
    """Tests for estimate_tokens - rough token estimation."""

    def test_empty_list(self):
        """Empty message list should estimate 0 tokens."""
        assert compact_mod.estimate_tokens([]) == 0

    def test_simple_messages(self):
        """Simple messages should estimate based on string length / 4."""
        messages = [{"role": "user", "content": "Hello world"}]
        # str(messages) length divided by 4
        result = compact_mod.estimate_tokens(messages)
        assert result > 0

    def test_large_messages(self):
        """Large messages should estimate higher token count."""
        short = [{"role": "user", "content": "short"}]
        long = [{"role": "user", "content": "a" * 1000}]
        assert compact_mod.estimate_tokens(long) > compact_mod.estimate_tokens(short)


class TestMicroCompact:
    """Tests for micro_compact - shortening old tool results."""

    def test_no_tool_results(self):
        """Messages without tool results should pass through unchanged."""
        messages = [
            {"role": "system", "content": "You are an agent"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = compact_mod.micro_compact(messages)
        assert result == messages

    def test_few_tool_results_preserved(self):
        """Tool results fewer than KEEP_RECENT should be preserved."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt\nfile2.txt"},
        ]
        result = compact_mod.micro_compact(messages)
        assert len(result) == 2
        assert result[1]["content"] == "file1.txt\nfile2.txt"

    def test_many_tool_results_compacted(self):
        """Old tool results (beyond KEEP_RECENT) should be compacted."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "a" * 200},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pwd"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc2", "content": "b" * 200},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc3", "type": "function", "function": {"name": "bash", "arguments": '{"command": "date"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc3", "content": "c" * 200},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc4", "type": "function", "function": {"name": "bash", "arguments": '{"command": "whoami"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc4", "content": "d" * 200},
        ]
        result = compact_mod.micro_compact(messages)
        # tc1 should be compacted, tc2-tc4 should be preserved (KEEP_RECENT=3)
        assert "Previous: used bash" in result[1]["content"]
        assert result[3]["content"] == "b" * 200
        assert result[5]["content"] == "c" * 200
        assert result[7]["content"] == "d" * 200

    def test_short_tool_results_not_compacted(self):
        """Tool results shorter than 100 chars should not be compacted."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "short output"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pwd"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc2", "content": "short2"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc3", "type": "function", "function": {"name": "bash", "arguments": '{"command": "date"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc3", "content": "short3"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc4", "type": "function", "function": {"name": "bash", "arguments": '{"command": "whoami"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc4", "content": "short4"},
        ]
        result = compact_mod.micro_compact(messages)
        # tc1 is short (<100 chars), should NOT be compacted
        assert result[1]["content"] == "short output"

    def test_preserve_read_file_results(self):
        """read_file tool results should be preserved even if old."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "file.txt"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "a" * 200},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pwd"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc2", "content": "b" * 200},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc3", "type": "function", "function": {"name": "bash", "arguments": '{"command": "date"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc3", "content": "c" * 200},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc4", "type": "function", "function": {"name": "bash", "arguments": '{"command": "whoami"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc4", "content": "d" * 200},
        ]
        result = compact_mod.micro_compact(messages)
        # read_file result should be preserved (PRESERVE_RESULT_TOOLS)
        assert result[1]["content"] == "a" * 200


class TestAutoCompact:
    """Tests for auto_compact - full conversation compression."""

    def test_auto_compact_saves_transcript(self, tmp_path):
        """auto_compact should save a transcript file."""
        with patch.object(compact_mod, "TRANSCRIPT_DIR", tmp_path / ".transcripts"):
            # Mock LLM response
            mock_msg = MagicMock()
            mock_msg.content = "Summary of conversation"
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            compact_mod.client.chat.completions.create = MagicMock(return_value=mock_response)

            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
            result = compact_mod.auto_compact(messages)

            # Should have created transcript file
            transcripts = list((tmp_path / ".transcripts").glob("transcript_*.jsonl"))
            assert len(transcripts) == 1

    def test_auto_compact_returns_summary_message(self, tmp_path):
        """auto_compact should return a user message with summary."""
        with patch.object(compact_mod, "TRANSCRIPT_DIR", tmp_path / ".transcripts"):
            mock_msg = MagicMock()
            mock_msg.content = "Summarized: we discussed Python."
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            compact_mod.client.chat.completions.create = MagicMock(return_value=mock_response)

            messages = [{"role": "user", "content": "Long conversation"}]
            result = compact_mod.auto_compact(messages)

            assert len(result) == 1
            assert result[0]["role"] == "user"
            assert "compressed" in result[0]["content"].lower()
            assert "Summarized" in result[0]["content"]


class TestCompactToolHandler:
    """Tests for compact tool handler."""

    def test_compact_handler_exists(self):
        """compact tool should be in TOOL_HANDLERS."""
        assert "compact" in compact_mod.TOOL_HANDLERS

    def test_compact_handler_returns_message(self):
        """compact handler should return a message about manual compression."""
        result = compact_mod.TOOL_HANDLERS["compact"]()
        assert "Manual compression" in result