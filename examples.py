"""
Examples of using LLM Agent Factory programmatically.

This file demonstrates how to use the retrieval and RAG systems
in your own Python code.

Key concepts:
- Retrieval = Search for existing agents in the database
- RAG = Generate new agents using LLM based on similar agents
- Quick API = Simple functions for quick start (quick_search, quick_generate)
- Advanced API = Full control over configuration
"""

from retrieval import (
    AgentRAG,
    AgentRetriever,
    DatasetType,
    EmbeddingModel,
    LLMConfig,
    RAGConfig,
    RetrievalConfig,
    quick_search,
)


def example_0_quick_api():
    """Example 0: Quick API - simplest way to use the library."""
    # Quick search - no configuration needed
    results = quick_search("Python programming expert", top_k=3)

    for _result in results:
        pass

    # Quick generate - minimal configuration
    # agents = quick_generate(
    #     "code review assistant for Python",
    #     api_key="your-api-key"
    # )
    # print(f"Generated: {agents[0]['display_name']}")


def example_1_basic_search():
    """Example 1: Basic agent search (retrieval only)."""
    # Create retrieval configuration
    config = RetrievalConfig(
        dataset_type=DatasetType.ENG,
        embedding_model=EmbeddingModel.BGE_SMALL.value,
        top_k=3,
    )

    # Create and initialize retriever
    retriever = AgentRetriever(config)
    retriever.initialize()

    # Search for agents
    results = retriever.search("I need help with Python programming")

    # Print results
    for _result in results:
        pass


def example_2_search_with_reranking():
    """Example 2: Search with two-stage retrieval (reranking)."""
    config = RetrievalConfig(
        dataset_type=DatasetType.ENG,
        embedding_model=EmbeddingModel.BGE_BASE.value,
        use_reranker=True,  # Enable two-stage retrieval
        rerank_top_k=20,  # First retrieve 20 candidates
        top_k=5,  # Then rerank and return top 5
    )

    retriever = AgentRetriever(config)
    retriever.initialize()

    results = retriever.search("data analysis and visualization expert")

    for result in results:
        if result.rerank_score is not None:
            pass


def example_3_multilingual_search():
    """Example 3: Multilingual search across all datasets."""
    config = RetrievalConfig(
        dataset_type=DatasetType.ALL,  # Use all datasets
        embedding_model=EmbeddingModel.BGE_M3.value,  # Multilingual model
        top_k=5,
    )

    retriever = AgentRetriever(config)
    retriever.initialize()

    # Search in English
    results = retriever.search("Python programming assistant")

    for _result in results:
        pass


def example_4_basic_rag_generation():
    """Example 4: Generate a new agent using RAG."""
    # Configure LLM (replace with your actual settings)
    llm_config = LLMConfig(
        model="gpt-oss",
        base_url="https://your-llm-api.com/v1",
        api_key="your-api-key",
        temperature=0.7,
    )

    # Create RAG configuration
    config = RAGConfig.with_dataset(
        dataset_type=DatasetType.ENG,
        llm=llm_config,
        num_agents_to_return=1,
        num_retrieved_for_context=5,
    )

    # Create and initialize RAG
    rag = AgentRAG(config)
    rag.initialize()

    # Generate agent
    agents = rag.generate("I need an agent for automated code review in TypeScript")

    # Print generated agent
    for _agent in agents:
        pass


def example_5_generate_multiple_agents():
    """Example 5: Generate multiple agent variants."""
    llm_config = LLMConfig(
        model="gpt-oss",
        base_url="https://your-llm-api.com/v1",
        api_key="your-api-key",
        temperature=0.9,  # Higher temperature for more variety
    )

    config = RAGConfig.with_dataset(
        dataset_type=DatasetType.ENG,
        llm=llm_config,
        num_agents_to_return=3,  # Generate 3 variants
        num_retrieved_for_context=10,
    )

    rag = AgentRAG(config)
    rag.initialize()

    agents = rag.generate("customer support specialist for SaaS product")

    for _i, _agent in enumerate(agents, 1):
        pass


def example_6_rag_with_retrieval():
    """Example 6: Use RAG for both search and generation."""
    llm_config = LLMConfig(
        model="gpt-oss",
        base_url="https://your-llm-api.com/v1",
        api_key="your-api-key",
        temperature=0.7,
    )

    config = RAGConfig.with_dataset(
        dataset_type=DatasetType.ENG,
        llm=llm_config,
    )

    rag = AgentRAG(config)
    rag.initialize()

    query = "machine learning model deployment expert"

    # First, search for similar agents
    similar = rag.search_only(query, top_k=3)

    for _result in similar:
        pass

    # Then, generate a new customized agent
    agents = rag.generate(query)

    agents[0]


def example_7_switch_datasets():
    """Example 7: Switch between datasets dynamically."""
    config = RetrievalConfig(
        dataset_type=DatasetType.ENG,
        embedding_model=EmbeddingModel.BGE_M3.value,
        top_k=3,
    )

    retriever = AgentRetriever(config)
    retriever.initialize()

    # Search in English dataset
    retriever.search("coding assistant")

    # Switch to all datasets
    retriever.switch_dataset(DatasetType.ALL)
    retriever.initialize()

    retriever.search("programming assistant")

    # Switch to all datasets
    retriever.switch_dataset(DatasetType.ALL)
    retriever.initialize()

    retriever.search("coding assistant")


def example_8_get_statistics():
    """Example 8: Get dataset statistics."""
    config = RetrievalConfig(
        dataset_type=DatasetType.ALL,
    )

    retriever = AgentRetriever(config)
    retriever.initialize()


if __name__ == "__main__":
    # Run examples
    # Note: Some examples require LLM configuration and may fail without proper setup

    try:
        # Quick API - recommended for beginners
        example_0_quick_api()

        # These examples work without LLM
        example_1_basic_search()
        example_2_search_with_reranking()
        example_3_multilingual_search()
        example_7_switch_datasets()
        example_8_get_statistics()

        # These examples require LLM configuration
        # Uncomment if you have configured LLM
        # example_4_basic_rag_generation()
        # example_5_generate_multiple_agents()
        # example_6_rag_with_retrieval()

    except Exception:
        pass
