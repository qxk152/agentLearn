"""
Unit tests for 03_todo.py - TodoWrite (OpenAI-compatible version)

Tests cover:
  - TodoManager: creation, update, validation, rendering
  - TodoManager: max items, in_progress limit, status validation
  - Integration with call_tool for todo dispatch
"""

import json
import os
from unittest.mock import MagicMock, patch

import importlib
import pytest

with patch.dict(os.environ, {"MODEL_ID": "test-model", "OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://localhost/v1"}):
    mock_openai = MagicMock()
    with patch("openai.OpenAI", return_value=mock_openai):
        spec = importlib.util.spec_from_file_location("todo_mod", "03_todo.py")
        todo_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(todo_mod)


class TestTodoManagerUpdate:
    """Tests for TodoManager.update() - the core CRUD operation."""

    def test_update_basic_items(self):
        """Basic todo items should be accepted and stored."""
        tm = todo_mod.TodoManager()
        items = [
            {"id": "1", "text": "Task one", "status": "pending"},
            {"id": "2", "text": "Task two", "status": "completed"},
        ]
        result = tm.update(items)
        assert "[ ]" in result  # pending marker
        assert "[x]" in result  # completed marker
        assert "Task one" in result
        assert "Task two" in result

    def test_update_in_progress_marker(self):
        """in_progress items should show [>] marker."""
        tm = todo_mod.TodoManager()
        items = [
            {"id": "1", "text": "Working on this", "status": "in_progress"},
        ]
        result = tm.update(items)
        assert "[>]" in result

    def test_update_max_20_items(self):
        """Should reject more than 20 items."""
        tm = todo_mod.TodoManager()
        items = [{"id": str(i), "text": f"Task {i}", "status": "pending"} for i in range(21)]
        with pytest.raises(ValueError, match="Max 20"):
            tm.update(items)

    def test_update_empty_text_rejected(self):
        """Items with empty text should be rejected."""
        tm = todo_mod.TodoManager()
        items = [{"id": "1", "text": "", "status": "pending"}]
        with pytest.raises(ValueError, match="text required"):
            tm.update(items)

    def test_update_whitespace_only_text_rejected(self):
        """Items with whitespace-only text should be rejected."""
        tm = todo_mod.TodoManager()
        items = [{"id": "1", "text": "   ", "status": "pending"}]
        with pytest.raises(ValueError, match="text required"):
            tm.update(items)

    def test_update_invalid_status_rejected(self):
        """Invalid status values should be rejected."""
        tm = todo_mod.TodoManager()
        items = [{"id": "1", "text": "Task", "status": "started"}]
        with pytest.raises(ValueError, match="invalid status"):
            tm.update(items)

    def test_update_multiple_in_progress_rejected(self):
        """Only one item can be in_progress at a time."""
        tm = todo_mod.TodoManager()
        items = [
            {"id": "1", "text": "Task A", "status": "in_progress"},
            {"id": "2", "text": "Task B", "status": "in_progress"},
        ]
        with pytest.raises(ValueError, match="Only one"):
            tm.update(items)

    def test_update_replaces_all_items(self):
        """Calling update replaces all existing items."""
        tm = todo_mod.TodoManager()
        tm.update([{"id": "1", "text": "Old task", "status": "completed"}])
        tm.update([{"id": "2", "text": "New task", "status": "pending"}])
        assert len(tm.items) == 1
        assert tm.items[0]["text"] == "New task"

    def test_update_auto_generates_id(self):
        """Items without id should get auto-generated id."""
        tm = todo_mod.TodoManager()
        items = [{"text": "No ID task", "status": "pending"}]
        result = tm.update(items)
        assert "#1" in result  # auto-generated id = "1"

    def test_update_default_status_pending(self):
        """Items without status should default to pending."""
        tm = todo_mod.TodoManager()
        items = [{"id": "1", "text": "Task"}]
        result = tm.update(items)
        assert "[ ]" in result

    def test_update_completed_count(self):
        """Rendered output should show completion count."""
        tm = todo_mod.TodoManager()
        items = [
            {"id": "1", "text": "Done", "status": "completed"},
            {"id": "2", "text": "Pending", "status": "pending"},
        ]
        result = tm.update(items)
        assert "1/2 completed" in result


class TestTodoManagerRender:
    """Tests for TodoManager.render() output format."""

    def test_render_empty(self):
        """Empty todo list should return 'No todos.'"""
        tm = todo_mod.TodoManager()
        assert tm.render() == "No todos."

    def test_render_format(self):
        """Each item should be formatted as: marker #id: text"""
        tm = todo_mod.TodoManager()
        items = [
            {"id": "1", "text": "Setup project", "status": "completed"},
            {"id": "2", "text": "Write tests", "status": "in_progress"},
            {"id": "3", "text": "Deploy", "status": "pending"},
        ]
        tm.update(items)
        rendered = tm.render()
        assert "[x] #1: Setup project" in rendered
        assert "[>] #2: Write tests" in rendered
        assert "[ ] #3: Deploy" in rendered


class TestTodoToolIntegration:
    """Tests for todo tool integration via call_tool."""

    def test_todo_via_call_tool(self):
        """todo tool should work through call_tool dispatch."""
        # Reset TODO state
        todo_mod.TODO = todo_mod.TodoManager()
        items = [
            {"id": "1", "text": "Test task", "status": "pending"},
        ]
        result = todo_mod.call_tool("todo", {"items": items})
        assert "Test task" in result

    def test_todo_error_via_call_tool(self):
        """todo validation errors should be caught by call_tool."""
        todo_mod.TODO = todo_mod.TodoManager()
        items = [
            {"id": "1", "text": "", "status": "pending"},  # empty text
        ]
        result = todo_mod.call_tool("todo", {"items": items})
        assert "Error" in result