"""Command-line interface for the agent retrieval system."""

import argparse
import json
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import contextlib

from retrieval.config import DatasetType, EmbeddingModel, RerankerModel, RetrievalConfig
from retrieval.retriever import AgentRetriever


def get_dataset_type(name: str) -> DatasetType:
    """Convert string to DatasetType enum."""
    name = name.lower().strip()
    mapping = {
        "eng": DatasetType.ENG,
        "english": DatasetType.ENG,
        "all": DatasetType.ALL,
    }
    if name in mapping:
        return mapping[name]
    msg = f"Unknown dataset type: {name}. Options: eng, all"
    raise ValueError(msg)


def format_agent_raw(agent) -> str:
    """Format agent as raw JSON matching the dataset format."""
    agent_dict = {
        "agent_id": agent.agent_id,
        "display_name": agent.display_name,
        "persona": agent.persona,
        "description": agent.description,
        "tools": agent.tools,
    }
    return json.dumps(agent_dict, ensure_ascii=False)


def interactive_mode(
    retriever: AgentRetriever,
    top_k: int,
    use_reranker: bool,
) -> None:
    """Run the interactive query mode."""
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        # Handle commands
        if query.startswith("/"):
            parts = query.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                break
            if cmd == "/switch":
                if not arg:
                    continue
                try:
                    new_type = get_dataset_type(arg)
                    retriever.switch_dataset(new_type)
                    retriever.initialize()
                except ValueError:
                    pass
            elif cmd == "/topk":
                if not arg:
                    pass
                else:
                    with contextlib.suppress(ValueError):
                        top_k = int(arg)
            elif cmd == "/rerank":
                use_reranker = not use_reranker
            elif cmd in {"/stats", "/help"}:
                pass
            else:
                pass
            continue

        # Perform search
        results = retriever.search(query, top_k=top_k, use_reranker=use_reranker)

        if not results:
            continue

        for _result in results:
            pass


def single_query_mode(
    retriever: AgentRetriever,
    query: str,
    top_k: int,
    use_reranker: bool,
) -> None:
    """Run a single query and output results as raw JSONL (one agent per line)."""
    results = retriever.search(query, top_k=top_k, use_reranker=use_reranker)

    for _result in results:
        pass


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Agent Retrieval System - Find the best agents for your task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with English dataset
  python -m retrieval.cli

  # Single query
  python -m retrieval.cli -q "I need help with math"

  # Use English dataset with more results
  python -m retrieval.cli -d eng -k 10

  # Use all datasets combined
  python -m retrieval.cli -d all -q "coding assistant"

  # Enable reranking for better accuracy
  python -m retrieval.cli -q "python coding" --rerank

  # Use a specific embedding model
  python -m retrieval.cli --model bge-base -q "math tutor"

Available embedding models:
  minilm     - MiniLM-L6 (fast, 384 dim)
  mpnet      - MPNet base (768 dim)
  bge-small  - BGE small (recommended, 384 dim)
  bge-base   - BGE base (768 dim)
  bge-large  - BGE large (1024 dim)
  bge-m3     - BGE M3 multilingual (1024 dim)
  e5-multi   - E5 multilingual (1024 dim)
""",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="eng",
        choices=["eng", "all"],
        help="Dataset to use (default: eng)",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Query to search for (if not provided, enters interactive mode)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum similarity threshold (default: 0.0)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached embeddings index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge-small",
        choices=[
            "minilm",
            "mpnet",
            "bge-small",
            "bge-base",
            "bge-large",
            "bge-m3",
            "e5-multi",
        ],
        help="Embedding model to use (default: bge-small)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking for better accuracy",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="bge-reranker-base",
        choices=[
            "bge-reranker-base",
            "bge-reranker-large",
            "bge-reranker-v2-m3",
            "msmarco-minilm",
        ],
        help="Reranker model (default: bge-reranker-base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device for computation (default: auto)",
    )

    args = parser.parse_args()

    # Map model shortcuts to full names
    model_mapping = {
        "minilm": EmbeddingModel.MINILM.value,
        "mpnet": EmbeddingModel.MPNET.value,
        "bge-small": EmbeddingModel.BGE_SMALL.value,
        "bge-base": EmbeddingModel.BGE_BASE.value,
        "bge-large": EmbeddingModel.BGE_LARGE.value,
        "bge-m3": EmbeddingModel.BGE_M3.value,
        "e5-multi": EmbeddingModel.MULTILINGUAL_E5.value,
    }

    reranker_mapping = {
        "bge-reranker-base": RerankerModel.BGE_RERANKER_BASE.value,
        "bge-reranker-large": RerankerModel.BGE_RERANKER_LARGE.value,
        "bge-reranker-v2-m3": RerankerModel.BGE_RERANKER_V2_M3.value,
        "msmarco-minilm": RerankerModel.MSMARCO_MINILM.value,
    }

    # Create config
    config = RetrievalConfig(
        dataset_type=get_dataset_type(args.dataset),
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        use_cached_index=not args.no_cache,
        embedding_model=model_mapping.get(args.model, args.model),
        use_reranker=args.rerank,
        reranker_model=reranker_mapping.get(args.reranker_model, args.reranker_model),
        device=args.device,
    )

    # Create retriever
    retriever = AgentRetriever(config)

    # Initialize (load data and build index)
    retriever.initialize()

    if args.query:
        # Single query mode
        single_query_mode(
            retriever,
            args.query,
            args.top_k,
            args.rerank,
        )
    else:
        # Interactive mode
        interactive_mode(retriever, args.top_k, args.rerank)


if __name__ == "__main__":
    main()
