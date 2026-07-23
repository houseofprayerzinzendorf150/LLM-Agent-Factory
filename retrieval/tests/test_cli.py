"""Regression tests for command-line output."""

from unittest.mock import Mock, patch

from retrieval import cli, rag_cli
from retrieval.models import AgentSpec, RetrievalResult
from retrieval.rag_config import RAGConfig


def test_agent_search_single_query_prints_json(capsys) -> None:
    """Search results should be emitted as one JSON object per line."""
    retriever = Mock()
    retriever.search.return_value = [
        RetrievalResult(
            agent=AgentSpec(agent_id="python_reviewer", display_name="Python Reviewer"),
            score=0.9,
            rank=1,
        )
    ]

    cli.single_query_mode(retriever, "review Python", top_k=3, use_reranker=False)

    captured = capsys.readouterr()
    assert '"agent_id": "python_reviewer"' in captured.out
    assert captured.err == ""


def test_agent_generate_single_query_prints_generated_agent(capsys) -> None:
    """Non-interactive generation should print the formatted result."""
    rag = Mock()
    rag.generate.return_value = [
        {
            "agent_id": "generated_agent",
            "display_name": "Generated Agent",
            "persona": "A generated persona",
            "description": "A generated description",
            "tools": [],
        }
    ]

    with patch.object(rag_cli, "AgentRAG", return_value=rag):
        exit_code = rag_cli.single_query_mode(RAGConfig(), "generate an agent")

    captured = capsys.readouterr()
    assert exit_code == 0
    rag.initialize.assert_called_once_with()
    rag.generate.assert_called_once_with("generate an agent")
    assert '"agent_id": "generated_agent"' in captured.out
    assert captured.err == ""
