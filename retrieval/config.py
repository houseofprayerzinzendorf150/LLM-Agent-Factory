"""Configuration for the Agent Retrieval System."""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetType(str, Enum):
    """Available dataset types for retrieval."""

    ENG = "agents_eng"
    ALL = "all"  # Use all datasets combined


class EmbeddingModel(str, Enum):
    """Available embedding models."""

    # Small & fast
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, 22M params

    # Medium quality
    MPNET = "sentence-transformers/all-mpnet-base-v2"  # 768 dim, 110M params

    # High quality (recommended for production)
    BGE_SMALL = "BAAI/bge-small-en-v1.5"  # 384 dim, 33M params
    BGE_BASE = "BAAI/bge-base-en-v1.5"  # 768 dim, 110M params
    BGE_LARGE = "BAAI/bge-large-en-v1.5"  # 1024 dim, 335M params

    # Multilingual
    BGE_M3 = "BAAI/bge-m3"  # 1024 dim, supports 100+ languages
    MULTILINGUAL_E5 = "intfloat/multilingual-e5-large"  # 1024 dim


class RerankerModel(str, Enum):
    """Available reranker models."""

    # Cross-encoder rerankers
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    BGE_RERANKER_LARGE = "BAAI/bge-reranker-large"
    BGE_RERANKER_V2_M3 = "BAAI/bge-reranker-v2-m3"  # Multilingual

    # MS MARCO trained
    MSMARCO_MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval system."""

    # Dataset configuration
    dataset_type: DatasetType = Field(default=DatasetType.ENG, description="Which dataset to use")
    agents_db_dir: Path = Field(
        default=Path("task-agents_database"), description="Directory containing agents database"
    )

    # Embedding model configuration
    embedding_model: str = Field(
        default=EmbeddingModel.BGE_SMALL.value,
        description="Sentence transformer model for embeddings",
    )

    # Reranker configuration
    use_reranker: bool = Field(default=False, description="Whether to use reranking")
    reranker_model: str = Field(
        default=RerankerModel.BGE_RERANKER_BASE.value,
        description="Cross-encoder model for reranking",
    )
    rerank_top_k: int = Field(default=20, ge=1, description="Number of candidates to rerank")

    # Retrieval configuration
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")

    # Index configuration
    use_cached_index: bool = Field(default=True, description="Cache embeddings to disk")
    cache_dir: Path = Field(default=Path("retrieval/.cache"), description="Directory for cached indices")

    # Deduplication
    deduplicate_results: bool = Field(default=True, description="Remove duplicate agents from results")

    # Device configuration
    device: Literal["auto", "cuda", "cpu", "mps"] = Field(
        default="auto", description="Device for embeddings (auto, cuda, cpu, mps)"
    )

    model_config = ConfigDict(use_enum_values=True)

    def get_dataset_paths(self) -> list[Path]:
        """
        Get paths to the dataset files or directories based on configuration.

        Supports both old format (JSONL files) and new format (JSON files in directory).
        """
        # Get dataset_type as string (handles both enum and string)
        dataset_type_str = self.dataset_type.value if isinstance(self.dataset_type, DatasetType) else self.dataset_type

        # First, check for old format: JSONL files (priority for backward compatibility)
        if dataset_type_str == DatasetType.ALL.value:
            jsonl_paths = [self.agents_db_dir / f"{dt.value}.jsonl" for dt in DatasetType if dt != DatasetType.ALL]
            # If any JSONL file exists, return JSONL paths
            if any(p.exists() for p in jsonl_paths):
                return jsonl_paths
        else:
            jsonl_path = self.agents_db_dir / f"{dataset_type_str}.jsonl"
            # If JSONL file exists, return it
            if jsonl_path.exists():
                return [jsonl_path]

        # Fallback to new format: check if agents_db_dir contains JSON files directly
        if self.agents_db_dir.exists() and any(self.agents_db_dir.glob("*.json")):
            # New format: directory with JSON files
            if dataset_type_str == DatasetType.ALL.value:
                # Return the directory itself - data_loader will load all JSON files
                return [self.agents_db_dir]
            # For specific types, still return directory (no way to filter by type in new structure)
            return [self.agents_db_dir]

        # If neither format exists, return expected JSONL paths (for tests)
        if dataset_type_str == DatasetType.ALL.value:
            return [self.agents_db_dir / f"{dt.value}.jsonl" for dt in DatasetType if dt != DatasetType.ALL]
        return [self.agents_db_dir / f"{dataset_type_str}.jsonl"]

    def get_cache_path(self) -> Path:
        """Get the path for cached embeddings."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Include model name in cache path
        model_suffix = self.embedding_model.split("/")[-1]
        # Get dataset_type as string (handles both enum and string)
        dataset_type_str = self.dataset_type.value if isinstance(self.dataset_type, DatasetType) else self.dataset_type
        return self.cache_dir / f"index_{dataset_type_str}_{model_suffix}.pt"
