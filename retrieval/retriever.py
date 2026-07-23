"""Main retriever module for agent search with optional reranking."""

import sys
import time

from retrieval.config import DatasetType, RetrievalConfig
from retrieval.data_loader import load_multiple_datasets
from retrieval.embedder import EmbeddingIndex, Reranker, get_embedder
from retrieval.models import AgentRecord, AgentSpec, RetrievalResult


def _log(msg: str) -> None:
    """Print a status message to stderr so it doesn't interfere with JSON output."""
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


def _get_index_records(records: list[AgentRecord], *, deduplicate: bool) -> list[AgentRecord]:
    """
    Select the records represented in the embedding index.

    Task-agent datasets legitimately contain many questions for the same agent.
    The current embedding text contains agent fields only, so those rows would
    otherwise produce identical vectors. All source records remain available
    for statistics; only the vector index is collapsed by ``agent_id``.
    """
    if not deduplicate:
        return records

    unique_records: list[AgentRecord] = []
    seen_agent_ids: set[str] = set()
    for record in records:
        agent_id = record.agent.agent_id
        if agent_id and agent_id in seen_agent_ids:
            continue
        if agent_id:
            seen_agent_ids.add(agent_id)
        unique_records.append(record)
    return unique_records


class AgentRetriever:
    """
    Agent retrieval system using semantic search with optional reranking.

    Two-stage retrieval:
    1. Dense retrieval using bi-encoder (fast, approximate)
    2. Optional reranking using cross-encoder (slower, more accurate)
    """

    def __init__(self, config: RetrievalConfig | None = None, verbose: bool = True):
        """
        Initialize the retriever.

        Args:
            config: Retrieval configuration. Uses defaults if not provided.
            verbose: Whether to print progress messages during initialization.

        """
        self.config = config or RetrievalConfig()
        self.verbose = verbose
        self._records: list[AgentRecord] = []
        self._index_records: list[AgentRecord] = []
        self._index: EmbeddingIndex | None = None
        self._reranker: Reranker | None = None
        self._initialized = False

    def _status(self, msg: str) -> None:
        """Print a status message if verbose mode is on."""
        if self.verbose:
            _log(msg)

    def initialize(self) -> None:
        """
        Initialize the retriever by loading data and building the index.

        This is called automatically on first search if not called explicitly.
        """
        if self._initialized:
            return

        total_start = time.time()

        # Load data
        self._status(f"[1/3] Loading dataset ({self.config.dataset_type})...")
        t0 = time.time()
        dataset_paths = self.config.get_dataset_paths()
        self._records = load_multiple_datasets(dataset_paths)
        self._index_records = _get_index_records(self._records, deduplicate=self.config.deduplicate_results)
        unique_agents = len({record.agent.agent_id for record in self._records if record.agent.agent_id})
        self._status(
            f"      Loaded {len(self._records):,} records / {unique_agents:,} unique agents ({time.time() - t0:.1f}s)"
        )

        # Create embedder and index
        self._status(f"[2/3] Loading embedding model ({self.config.embedding_model.split('/')[-1]})...")
        t0 = time.time()
        embedder = get_embedder(self.config.embedding_model, self.config.device)
        self._index = EmbeddingIndex(embedder)
        self._status(f"      Model loaded ({time.time() - t0:.1f}s)")

        # Try to load cached index
        cache_path = self.config.get_cache_path()
        cache_loaded = self.config.use_cached_index and self._index.load(cache_path)
        cache_matches_records = cache_loaded and len(self._index.texts) == len(self._index_records)
        if cache_matches_records:
            self._status(f"[3/3] Loaded cached index from {cache_path.name}")
        else:
            # Build index
            self._status(f"[3/3] Building embedding index for {len(self._index_records):,} agents...")
            self._status("      Progress is shown below. The completed index will be cached for future use.")
            t0 = time.time()
            texts = [record.agent.get_indexable_text() for record in self._index_records]
            self._index.build(texts)
            self._status(f"      Index built ({time.time() - t0:.1f}s)")

            # Save cache
            if self.config.use_cached_index:
                self._index.save(cache_path)
                self._status(f"      Index cached to {cache_path.name}")

        # Initialize reranker if configured
        if self.config.use_reranker:
            self._status("Loading reranker model...")
            try:
                self._reranker = Reranker(
                    self.config.reranker_model,
                    self.config.device,
                )
                self._status("Reranker loaded!")
            except Exception:
                self._reranker = None
                self._status("Warning: Failed to load reranker, continuing without it.")

        self._initialized = True
        self._status(f"Ready! (total init: {time.time() - total_start:.1f}s)\n")

    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        use_reranker: bool | None = None,
    ) -> list[RetrievalResult]:
        """
        Search for agents matching the query.

        Args:
            query: User query to search for
            top_k: Number of results to return (overrides config)
            threshold: Minimum similarity threshold (overrides config)
            use_reranker: Whether to use reranking (overrides config)

        Returns:
            List of RetrievalResult objects sorted by relevance

        """
        if not self._initialized:
            self.initialize()

        k = top_k or self.config.top_k
        min_score = threshold if threshold is not None else self.config.similarity_threshold
        should_rerank = use_reranker if use_reranker is not None else self.config.use_reranker

        # For reranking, retrieve more candidates
        retrieve_k = max(self.config.rerank_top_k, k) if should_rerank and self._reranker else k

        # Stage 1: Dense retrieval
        assert self._index is not None, "Index not built. Call initialize() first."
        retrieval_results = self._index.search(query, top_k=retrieve_k)

        # Build initial results
        seen_agent_ids = set()
        candidates: list[tuple[AgentRecord, float]] = []

        for idx, score in retrieval_results:
            if score < min_score:
                continue

            record = self._index_records[idx]

            # Deduplicate by agent_id
            if self.config.deduplicate_results:
                if record.agent.agent_id and record.agent.agent_id in seen_agent_ids:
                    continue
                if record.agent.agent_id:
                    seen_agent_ids.add(record.agent.agent_id)

            candidates.append((record, score))

        # Stage 2: Reranking (optional)
        if should_rerank and self._reranker and candidates:
            # Prepare documents for reranking
            docs = [c[0].agent.get_indexable_text() for c in candidates]

            # Rerank
            reranked = self._reranker.rerank(query, docs, top_k=k)

            # Build final results with rerank scores
            results = []
            for rank, (orig_idx, rerank_score) in enumerate(reranked, 1):
                record, retrieval_score = candidates[orig_idx]
                results.append(
                    RetrievalResult(
                        agent=record.agent,
                        score=retrieval_score,
                        rank=rank,
                        rerank_score=rerank_score,
                    )
                )
            return results

        # No reranking - return top-k from retrieval
        results = []
        for rank, (record, score) in enumerate(candidates[:k], 1):
            results.append(
                RetrievalResult(
                    agent=record.agent,
                    score=score,
                    rank=rank,
                )
            )

        return results

    def switch_dataset(self, dataset_type: DatasetType) -> None:
        """
        Switch to a different dataset.

        Args:
            dataset_type: The dataset to switch to

        """
        new_type = dataset_type.value if isinstance(dataset_type, DatasetType) else dataset_type

        if new_type == self.config.dataset_type:
            return

        self.config.dataset_type = new_type
        self._initialized = False
        self._records = []
        self._index_records = []
        self._index = None

    def get_unique_agents(self) -> list[AgentSpec]:
        """Get a list of unique agents in the current dataset."""
        if not self._initialized:
            self.initialize()

        seen = set()
        unique = []
        for record in self._records:
            if record.agent.agent_id not in seen:
                seen.add(record.agent.agent_id)
                unique.append(record.agent)
        return unique

    @property
    def dataset_stats(self) -> dict:
        """Get statistics about the loaded dataset."""
        if not self._initialized:
            self.initialize()

        unique_agents = len({r.agent.agent_id for r in self._records})
        return {
            "total_records": len(self._records),
            "unique_agents": unique_agents,
            "indexed_records": len(self._index_records),
            "dataset_type": self.config.dataset_type,
            "embedding_model": self.config.embedding_model,
            "reranker_enabled": self._reranker is not None,
        }
