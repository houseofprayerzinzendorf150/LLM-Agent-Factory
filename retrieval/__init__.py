"""
Agent Retrieval and RAG System for LLM Agent Factory.

This module provides:
1. Semantic search capabilities to find the most relevant agents
2. RAG (Retrieval-Augmented Generation) for generating new agents

Features:
- Multiple embedding models (BGE, MiniLM, E5, multilingual)
- Optional cross-encoder reranking for improved accuracy
- Support for 1 dataset (eng) or all combined
- PyTorch-based for GPU acceleration
- Caching for fast repeated queries
- RAG-based agent generation using LLM
"""

from retrieval.config import DatasetType, EmbeddingModel, RerankerModel, RetrievalConfig
from retrieval.models import AgentRecord, AgentSpec, RetrievalResult, SearchQuery
from retrieval.rag import AgentRAG, format_agent_output
from retrieval.rag_config import LLMConfig, RAGConfig
from retrieval.retriever import AgentRetriever

__all__ = [
    # RAG
    "AgentRAG",
    "AgentRecord",
    # Retrieval
    "AgentRetriever",
    "AgentSpec",
    "DatasetType",
    "EmbeddingModel",
    "LLMConfig",
    "RAGConfig",
    "RerankerModel",
    "RetrievalConfig",
    "RetrievalResult",
    "SearchQuery",
    "format_agent_output",
    "quick_generate",
    # Quick start functions
    "quick_search",
]


def quick_search(query: str, dataset: str = "eng", top_k: int = 5, use_reranker: bool = False):
    """
    Quick search for agents without manual configuration.

    Args:
        query: Search query string
        dataset: Dataset to use ("eng", "all")
        top_k: Number of results to return
        use_reranker: Enable two-stage retrieval for better accuracy

    Returns:
        List of RetrievalResult objects

    Example:
        >>> from retrieval import quick_search
        >>> results = quick_search("Python programming expert")
        >>> for r in results:
        ...     print(f"{r.agent.display_name}: {r.agent.description}")

    """
    dataset_map = {
        "eng": DatasetType.ENG,
        "all": DatasetType.ALL,
    }

    config = RetrievalConfig(
        dataset_type=dataset_map.get(dataset, DatasetType.ENG),
        top_k=top_k,
        use_reranker=use_reranker,
    )

    retriever = AgentRetriever(config)
    retriever.initialize()

    return retriever.search(query)


def quick_generate(
    query: str,
    dataset: str = "eng",
    num_agents: int = 1,
    model: str = "gpt-4",
    api_key: str | None = None,
    base_url: str | None = None,
):
    """
    Quick agent generation without manual configuration.

    Args:
        query: Description of the agent you want to generate
        dataset: Dataset to use for examples ("eng", "all")
        num_agents: Number of agent variants to generate
        model: LLM model name
        api_key: API key for LLM (optional, uses env var if not provided)
        base_url: Base URL for LLM API (optional)

    Returns:
        List of generated agent specifications (dicts)

    Example:
        >>> from retrieval import quick_generate
        >>> agents = quick_generate(
        ...     "code review assistant for Python",
        ...     api_key="your-key"
        ... )
        >>> print(agents[0]["display_name"])

    """
    import os

    dataset_map = {
        "eng": DatasetType.ENG,
        "all": DatasetType.ALL,
    }

    llm_config = LLMConfig(
        model=model,
        api_key=api_key or os.getenv("LLM_API_KEY", ""),
        base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )

    config = RAGConfig.with_dataset(
        dataset_type=dataset_map.get(dataset, DatasetType.ENG),
        llm=llm_config,
        num_agents_to_return=num_agents,
    )

    rag = AgentRAG(config)
    rag.initialize()

    return rag.generate(query)
