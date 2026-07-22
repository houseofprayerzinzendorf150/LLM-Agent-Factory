"""Tests for the retriever module."""

import json
import tempfile
from pathlib import Path

import pytest

from retrieval.config import DatasetType, RetrievalConfig
from retrieval.models import AgentSpec, RetrievalResult
from retrieval.retriever import AgentRetriever


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test dataset
        dataset_file = tmpdir / "agents_eng.jsonl"

        agents = [
            {
                "input": "I need help with calculus and integration",
                "output": {
                    "agent_id": "math_tutor",
                    "display_name": "Math Tutor",
                    "persona": "An expert mathematician",
                    "description": "Helps students with mathematical problems including calculus, algebra, and geometry",
                    "tools": ["calculator"],
                },
            },
            {
                "input": "Can you help me write Python code?",
                "output": {
                    "agent_id": "python_coder",
                    "display_name": "Python Coder",
                    "persona": "A skilled Python developer",
                    "description": "Assists with Python programming, debugging, and code optimization",
                    "tools": ["code_executor", "linter"],
                },
            },
            {
                "input": "I want to learn about history",
                "output": {
                    "agent_id": "history_teacher",
                    "display_name": "History Teacher",
                    "persona": "A passionate historian",
                    "description": "Teaches world history, from ancient civilizations to modern times",
                    "tools": [],
                },
            },
            {
                "input": "Help me with algebra homework",
                "output": {
                    "agent_id": "math_tutor",
                    "display_name": "Math Tutor",
                    "persona": "An expert mathematician",
                    "description": "Helps students with mathematical problems including calculus, algebra, and geometry",
                    "tools": ["calculator"],
                },
            },
        ]

        with open(dataset_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(agent, ensure_ascii=False) + "\n" for agent in agents)

        yield tmpdir


class TestAgentRetriever:
    """Tests for AgentRetriever class."""

    def test_initialization(self, temp_dataset):
        """Test retriever initialization."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            agents_db_dir=temp_dataset,
            use_cached_index=False,
        )

        retriever = AgentRetriever(config)
        retriever.initialize()

        assert retriever._initialized
        assert len(retriever._records) == 4  # Including duplicate

    def test_search_basic(self, temp_dataset):
        """Test basic search functionality."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            agents_db_dir=temp_dataset,
            use_cached_index=False,
            top_k=3,
        )

        retriever = AgentRetriever(config)
        retriever.initialize()

        results = retriever.search("I need math help")

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Math tutor should be in results for math query
        agent_ids = [r.agent.agent_id for r in results]
        assert "math_tutor" in agent_ids

    def test_search_deduplication(self, temp_dataset):
        """Test that results are deduplicated by agent_id."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            agents_db_dir=temp_dataset,
            use_cached_index=False,
            deduplicate_results=True,
        )

        retriever = AgentRetriever(config)
        retriever.initialize()

        results = retriever.search("math algebra calculus", top_k=10)

        agent_ids = [r.agent.agent_id for r in results]
        # No duplicates
        assert len(agent_ids) == len(set(agent_ids))

    def test_search_without_deduplication(self, temp_dataset):
        """Test search without deduplication."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            agents_db_dir=temp_dataset,
            use_cached_index=False,
            deduplicate_results=False,
        )

        retriever = AgentRetriever(config)
        retriever.initialize()

        results = retriever.search("math help", top_k=10)

        # At least we get results (may contain duplicates)
        assert len(results) > 0

    def test_search_top_k(self, temp_dataset):
        """Test that top_k limits results."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            agents_db_dir=temp_dataset,
            use_cached_index=False,
        )

        retriever = AgentRetriever(config)
        retriever.initialize()

        results = retriever.search("help me", top_k=2)

        assert len(results) <= 2

    def test_search_threshold(self, temp_dataset):
        """Test similarity threshold filtering."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            agents_db_dir=temp_dataset,
            use_cached_index=False,
        )

        retriever = AgentRetriever(config)
        retriever.initialize()

        # Very high threshold should filter out most results
        results = retriever.search("random unrelated query", threshold=0.99)

        # Should have few or no results
        for r in results:
            assert r.score >= 0.99

    def test_dataset_stats(self, temp_dataset):
        """Test dataset statistics."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            agents_db_dir=temp_dataset,
            use_cached_index=False,
        )

        retriever = AgentRetriever(config)
        retriever.initialize()

        stats = retriever.dataset_stats

        assert stats["total_records"] == 4
        assert stats["unique_agents"] == 3  # math_tutor appears twice
        assert "agents_eng" in stats["dataset_type"]

    def test_get_unique_agents(self, temp_dataset):
        """Test getting unique agents."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            agents_db_dir=temp_dataset,
            use_cached_index=False,
        )

        retriever = AgentRetriever(config)
        retriever.initialize()

        unique = retriever.get_unique_agents()

        assert len(unique) == 3
        unique_ids = {a.agent_id for a in unique}
        assert unique_ids == {"math_tutor", "python_coder", "history_teacher"}


class TestRetrievalResult:
    """Tests for RetrievalResult class."""

    def test_format_output_full(self):
        """Test full output formatting (default)."""
        agent = AgentSpec(
            agent_id="test_agent",
            display_name="Test Agent",
            persona="A test persona",
            description="A test description",
            tools=["tool1"],
        )
        result = RetrievalResult(agent=agent, score=0.85, rank=1)

        output = result.format_output(full=True)

        assert "[1]" in output
        assert "0.85" in output
        # Full JSON format
        assert '"agent_id": "test_agent"' in output
        assert '"display_name": "Test Agent"' in output
        assert '"persona": "A test persona"' in output
        assert '"description": "A test description"' in output
        assert '"tools": [\n    "tool1"\n  ]' in output

    def test_format_output_compact(self):
        """Test compact output formatting."""
        agent = AgentSpec(
            agent_id="test_agent",
            display_name="Test Agent",
            persona="A test persona",
            description="A test description",
            tools=["tool1", "tool2"],
        )
        result = RetrievalResult(agent=agent, score=0.85, rank=1)

        output = result.format_output(full=False, verbose=True)

        assert "Test Agent" in output
        assert "ID: test_agent" in output
        assert "A test persona" in output
        assert "A test description" in output
        assert "tool1" in output

    def test_format_output_with_rerank_score(self):
        """Test output formatting with rerank score."""
        agent = AgentSpec(
            agent_id="test_agent",
            display_name="Test Agent",
        )
        result = RetrievalResult(agent=agent, score=0.85, rank=1, rerank_score=0.92)

        output = result.format_output(full=False)

        assert "rerank: 0.92" in output
        assert "retrieval: 0.85" in output
