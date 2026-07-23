"""Data loader for agent datasets."""

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

from retrieval.models import AgentRecord, AgentSpec


def load_dataset(file_path: Path) -> Iterator[AgentRecord]:
    """
    Load agent records from a JSONL file or JSON file.

    Supports two formats:
    1. JSONL: one JSON object per line (legacy format)
    2. JSON: single JSON file with {"agents": [...]} structure (new format)

    Args:
        file_path: Path to the JSONL or JSON file

    Yields:
        AgentRecord objects

    """
    if not file_path.exists():
        msg = f"Dataset not found: {file_path}"
        raise FileNotFoundError(msg)

    # Try JSONL format first (one JSON per line)
    parsed_jsonl = False
    try:
        with open(file_path, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                # Try to parse as JSONL
                f.seek(0)
                for _line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Check if it's a JSONL record (has "input" and "output" keys)
                        if "input" in data and "output" in data:
                            parsed_jsonl = True
                            yield AgentRecord.from_json(data, str(file_path))
                            continue
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass

    if parsed_jsonl:
        return

    # If JSONL didn't work, try JSON format (single file with agents array)
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

            # Handle new format: {"domain": "...", "agents": [...]}
            if isinstance(data, dict) and "agents" in data:
                agents = data["agents"]
                domain = data.get("domain", "")
                for agent_data in agents:
                    # Convert agent dict to AgentSpec
                    agent = AgentSpec(
                        agent_id=agent_data.get("agent_id", ""),
                        display_name=agent_data.get("display_name", ""),
                        persona=agent_data.get("persona", ""),
                        description=agent_data.get("description", ""),
                        tools=agent_data.get("tools", []),
                    )
                    # Create AgentRecord with empty input_text (no query for domain-based agents)
                    yield AgentRecord(
                        input_text=f"Generate agent for {domain} domain",
                        agent=agent,
                        source_file=str(file_path),
                    )
                return

            # Handle array format: [{"agent_id": ..., ...}, ...]
            if isinstance(data, list):
                for agent_data in data:
                    if isinstance(agent_data, dict):
                        agent = AgentSpec(
                            agent_id=agent_data.get("agent_id", ""),
                            display_name=agent_data.get("display_name", ""),
                            persona=agent_data.get("persona", ""),
                            description=agent_data.get("description", ""),
                            tools=agent_data.get("tools", []),
                        )
                        yield AgentRecord(
                            input_text=agent_data.get("input", ""),
                            agent=agent,
                            source_file=str(file_path),
                        )
    except (json.JSONDecodeError, KeyError, TypeError):
        pass


def load_from_json_directory(directory: Path) -> Iterator[AgentRecord]:
    """
    Load agent records from all JSON files in a directory.

    Args:
        directory: Directory containing JSON files

    Yields:
        AgentRecord objects from all JSON files

    """
    if not directory.exists():
        return

    for json_file in directory.glob("*.json"):
        try:
            yield from load_dataset(json_file)
        except Exception:
            continue


def load_multiple_datasets(file_paths: list[Path]) -> list[AgentRecord]:
    """
    Load agent records from multiple JSONL/JSON files or directories.

    Args:
        file_paths: List of paths to JSONL files, JSON files, or directories

    Returns:
        List of all AgentRecord objects

    """
    records = []
    for path in file_paths:
        with contextlib.suppress(FileNotFoundError):
            if path.is_dir():
                # If it's a directory, load all JSON files in it
                records.extend(load_from_json_directory(path))
            else:
                # If it's a file, load it
                records.extend(load_dataset(path))
    return records
