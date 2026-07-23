"""Tests for the data loader module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from retrieval.data_loader import load_dataset, load_multiple_datasets
from retrieval.models import AgentRecord, AgentSpec


class TestAgentSpec:
    """Tests for AgentSpec class."""

    def test_creation(self):
        """Test creating AgentSpec."""
        agent = AgentSpec(
            agent_id="math_helper",
            display_name="Math Helper",
            persona="A friendly math tutor",
            description="Helps with mathematical problems",
            tools=["calculator", "plotter"],
        )

        assert agent.agent_id == "math_helper"
        assert agent.display_name == "Math Helper"
        assert agent.persona == "A friendly math tutor"
        assert agent.description == "Helps with mathematical problems"
        assert agent.tools == ["calculator", "plotter"]

    def test_default_values(self):
        """Test AgentSpec with default values."""
        agent = AgentSpec(
            agent_id="test",
            display_name="Test",
        )

        assert agent.persona == ""
        assert agent.description == ""
        assert agent.tools == []

    def test_get_indexable_text(self):
        """Test getting indexable text."""
        agent = AgentSpec(
            agent_id="test",
            display_name="Test Agent",
            persona="A helpful assistant",
            description="Does testing things",
        )

        text = agent.get_indexable_text()

        assert "Test Agent" in text
        assert "Does testing things" in text
        assert "A helpful assistant" in text


class TestAgentRecord:
    """Tests for AgentRecord class."""

    def test_from_json(self):
        """Test creating AgentRecord from JSON."""
        data = {
            "input": "Help me with math",
            "output": {
                "agent_id": "math_helper",
                "display_name": "Math Helper",
                "persona": "A friendly math tutor",
                "description": "Helps with mathematical problems",
                "tools": ["calculator", "plotter"],
            },
        }

        record = AgentRecord.from_json(data, "test.jsonl")

        assert record.input_text == "Help me with math"
        assert record.agent.agent_id == "math_helper"
        assert record.agent.display_name == "Math Helper"
        assert record.agent.persona == "A friendly math tutor"
        assert record.agent.description == "Helps with mathematical problems"
        assert record.agent.tools == ["calculator", "plotter"]
        assert record.source_file == "test.jsonl"

    def test_from_json_missing_fields(self):
        """Test creating AgentRecord with missing optional fields."""
        data = {
            "input": "Test query",
            "output": {
                "agent_id": "test_agent",
                "display_name": "Test Agent",
            },
        }

        record = AgentRecord.from_json(data, "test.jsonl")

        assert record.agent.agent_id == "test_agent"
        assert record.agent.persona == ""
        assert record.agent.description == ""
        assert record.agent.tools == []


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_valid_jsonl(self):
        """Test loading a valid JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "input": "Query 1",
                        "output": {
                            "agent_id": "agent1",
                            "display_name": "Agent 1",
                            "persona": "",
                            "description": "",
                            "tools": [],
                        },
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "input": "Query 2",
                        "output": {
                            "agent_id": "agent2",
                            "display_name": "Agent 2",
                            "persona": "",
                            "description": "",
                            "tools": [],
                        },
                    }
                )
                + "\n"
            )
            temp_path = Path(f.name)

        try:
            records = list(load_dataset(temp_path))
            assert len(records) == 2
            assert records[0].agent.agent_id == "agent1"
            assert records[1].agent.agent_id == "agent2"
        finally:
            temp_path.unlink()

    def test_load_with_empty_lines(self):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "input": "Query",
                        "output": {
                            "agent_id": "agent",
                            "display_name": "Agent",
                            "persona": "",
                            "description": "",
                            "tools": [],
                        },
                    }
                )
                + "\n"
            )
            f.write("\n")
            f.write("  \n")
            temp_path = Path(f.name)

        try:
            records = list(load_dataset(temp_path))
            assert len(records) == 1
        finally:
            temp_path.unlink()

    def test_jsonl_is_not_reopened_as_a_json_document(self):
        """Test that a recognized JSONL file is read only once."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "input": "Query",
                        "output": {
                            "agent_id": "agent",
                            "display_name": "Agent",
                        },
                    }
                )
                + "\n"
            )
            temp_path = Path(f.name)

        try:
            real_open = open
            with patch("builtins.open", wraps=real_open) as mocked_open:
                records = list(load_dataset(temp_path))

            assert len(records) == 1
            assert mocked_open.call_count == 1
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            list(load_dataset(Path("nonexistent.jsonl")))


class TestLoadMultipleDatasets:
    """Tests for load_multiple_datasets function."""

    def test_load_multiple_files(self):
        """Test loading from multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create first file
            file1 = tmpdir / "file1.jsonl"
            with open(file1, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "input": "Q1",
                            "output": {
                                "agent_id": "a1",
                                "display_name": "A1",
                                "persona": "",
                                "description": "",
                                "tools": [],
                            },
                        }
                    )
                    + "\n"
                )

            # Create second file
            file2 = tmpdir / "file2.jsonl"
            with open(file2, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "input": "Q2",
                            "output": {
                                "agent_id": "a2",
                                "display_name": "A2",
                                "persona": "",
                                "description": "",
                                "tools": [],
                            },
                        }
                    )
                    + "\n"
                )

            records = load_multiple_datasets([file1, file2])

            assert len(records) == 2
            agent_ids = {r.agent.agent_id for r in records}
            assert agent_ids == {"a1", "a2"}
