"""Tests for intercept plugin structure."""

import sys
from pathlib import Path

# Add interverse/ to path so _shared package is importable
_interverse = Path(__file__).resolve().parents[3]
if str(_interverse) not in sys.path:
    sys.path.insert(0, str(_interverse))

from _shared.tests.structural.test_base import StructuralTests


class TestStructure(StructuralTests):
    """Structural tests -- inherits shared base, adds plugin-specific checks.

    intercept is a minimal infrastructure plugin (no skills, no commands,
    no agents). It is also missing PHILOSOPHY.md, LICENSE, and author in
    plugin.json -- overrides account for this.
    """

    def test_plugin_name(self, plugin_json):
        assert plugin_json["name"] == "intercept"

    def test_plugin_json_valid(self, project_root, plugin_json):
        """plugin.json is valid JSON with required fields.

        Override: intercept plugin.json omits 'author' -- check the rest.
        """
        for field in ("name", "version", "description"):
            assert field in plugin_json, (
                f"plugin.json missing required field: {field}"
            )

    def test_required_root_files(self, project_root):
        """Required root-level files exist.

        Override: intercept is missing PHILOSOPHY.md and LICENSE.
        """
        required = ["CLAUDE.md", ".gitignore"]
        for name in required:
            assert (project_root / name).exists(), (
                f"Missing required file: {name}"
            )

    def test_gates_directory_exists(self, project_root):
        """gates/ directory exists for gate definitions."""
        assert (project_root / "gates").is_dir(), "Missing gates/ directory"

    def test_lib_directory_exists(self, project_root):
        """lib/ directory exists."""
        assert (project_root / "lib").is_dir(), "Missing lib/ directory"

    def test_bin_directory_exists(self, project_root):
        """bin/ directory exists."""
        assert (project_root / "bin").is_dir(), "Missing bin/ directory"
