"""
Unit tests for 05_skill_loader.py - Skills (OpenAI-compatible version)

Tests cover:
  - SkillLoader: loading, parsing frontmatter, descriptions, content
  - SkillLoader: missing skills, YAML parsing errors
  - load_skill tool integration
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import importlib
import pytest

with patch.dict(os.environ, {"MODEL_ID": "test-model", "OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://localhost/v1"}):
    mock_openai = MagicMock()
    with patch("openai.OpenAI", return_value=mock_openai):
        # Also need yaml
        try:
            import yaml
        except ImportError:
            pytest.skip("yaml not installed", allow_module_level=True)

        spec = importlib.util.spec_from_file_location("skill_mod", "05_skill_loader.py")
        skill_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(skill_mod)


class TestSkillLoader:
    """Tests for SkillLoader class."""

    def test_load_all_from_skills_dir(self, tmp_path):
        """Should load all SKILL.md files from skills directory."""
        skills_dir = tmp_path / "skills" / "test-skill"
        skills_dir.mkdir(parents=True)
        skill_file = skills_dir / "SKILL.md"
        skill_file.write_text("---\nname: test-skill\ndescription: A test skill\ntags: testing\n---\nSkill body content here.")

        loader = skill_mod.SkillLoader(skills_dir)
        assert "test-skill" in loader.skills

    def test_parse_frontmatter_valid(self, tmp_path):
        """Valid YAML frontmatter should be parsed correctly."""
        loader = skill_mod.SkillLoader(tmp_path)  # empty dir
        text = "---\nname: my-skill\ndescription: Does stuff\n---\nBody content."
        meta, body = loader._parse_frontmatter(text)
        assert meta["name"] == "my-skill"
        assert meta["description"] == "Does stuff"
        assert body == "Body content."

    def test_parse_frontmatter_no_frontmatter(self):
        """Files without frontmatter should return empty meta."""
        loader = skill_mod.SkillLoader(Path("/tmp"))
        text = "Just plain text, no YAML."
        meta, body = loader._parse_frontmatter(text)
        assert meta == {}
        assert body == "Just plain text, no YAML."

    def test_parse_frontmatter_invalid_yaml(self):
        """Invalid YAML should result in empty meta dict."""
        loader = skill_mod.SkillLoader(Path("/tmp"))
        text = "---\nnot: valid: yaml: here\n---\nBody."
        meta, body = loader._parse_frontmatter(text)
        # Should gracefully handle YAML error
        assert isinstance(meta, dict)

    def test_get_descriptions(self, tmp_path):
        """Should render skill descriptions for system prompt."""
        skills_dir = tmp_path / "skills" / "skill-a"
        skills_dir.mkdir(parents=True)
        skill_file = skills_dir / "SKILL.md"
        skill_file.write_text("---\nname: skill-a\ndescription: Skill A desc\ntags: alpha\n---\nContent.")

        loader = skill_mod.SkillLoader(skills_dir)
        descriptions = loader.get_descriptions()
        assert "skill-a" in descriptions
        assert "Skill A desc" in descriptions
        assert "alpha" in descriptions

    def test_get_descriptions_no_skills(self):
        """Empty skills should return '(no skills available)'."""
        loader = skill_mod.SkillLoader(Path("/nonexistent"))
        descriptions = loader.get_descriptions()
        assert "(no skills available)" in descriptions

    def test_get_content_existing_skill(self, tmp_path):
        """Getting content of existing skill should return body wrapped in XML."""
        skills_dir = tmp_path / "skills" / "test-skill"
        skills_dir.mkdir(parents=True)
        skill_file = skills_dir / "SKILL.md"
        skill_file.write_text("---\nname: test-skill\n---\nDetailed skill body.")

        loader = skill_mod.SkillLoader(skills_dir)
        content = loader.get_content("test-skill")
        assert "<skill name=\"test-skill\">" in content
        assert "Detailed skill body." in content

    def test_get_content_unknown_skill(self, tmp_path):
        """Getting unknown skill should return error with available list."""
        loader = skill_mod.SkillLoader(tmp_path)
        content = loader.get_content("nonexistent-skill")
        assert "Unknown skill" in content

    def test_skill_directory_not_exists(self):
        """Non-existent skills directory should not crash."""
        loader = skill_mod.SkillLoader(Path("/absolutely/nonexistent/path"))
        assert loader.skills == {}


class TestLoadSkillTool:
    """Tests for load_skill as a tool handler."""

    def test_load_skill_via_tool_handlers(self, tmp_path):
        """load_skill should be in TOOL_HANDLERS."""
        assert "load_skill" in skill_mod.TOOL_HANDLERS

    def test_load_skill_in_tools_definition(self):
        """load_skill should be in TOOLS list."""
        tool_names = [t["function"]["name"] for t in skill_mod.TOOLS]
        assert "load_skill" in tool_names