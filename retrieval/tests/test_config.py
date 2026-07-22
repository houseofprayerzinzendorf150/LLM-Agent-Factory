"""Tests for the configuration module."""

import tempfile
from pathlib import Path

import pytest

from retrieval.config import DatasetType, EmbeddingModel, RerankerModel, RetrievalConfig


class TestDatasetType:
    """Tests for DatasetType enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert DatasetType.ENG.value == "agents_eng"
        assert DatasetType.ALL.value == "all"


class TestEmbeddingModel:
    """Tests for EmbeddingModel enum."""

    def test_model_names(self):
        """Test that model names are valid."""
        assert "MiniLM" in EmbeddingModel.MINILM.value
        assert "bge" in EmbeddingModel.BGE_SMALL.value
        assert "bge" in EmbeddingModel.BGE_BASE.value
        assert "bge" in EmbeddingModel.BGE_LARGE.value


class TestRerankerModel:
    """Tests for RerankerModel enum."""

    def test_reranker_names(self):
        """Test that reranker names are valid."""
        assert "reranker" in RerankerModel.BGE_RERANKER_BASE.value
        assert "reranker" in RerankerModel.BGE_RERANKER_LARGE.value


class TestRetrievalConfig:
    """Tests for RetrievalConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetrievalConfig()

        assert config.dataset_type == DatasetType.ENG.value
        assert config.top_k == 5
        assert config.similarity_threshold == 0.0
        assert config.use_cached_index is True
        assert config.deduplicate_results is True
        assert config.use_reranker is False

    def test_get_dataset_paths_single(self):
        """Test getting path for a single dataset."""
        config = RetrievalConfig(dataset_type=DatasetType.ENG)
        paths = config.get_dataset_paths()

        assert len(paths) == 1
        assert paths[0].name == "agents_eng.jsonl"

    def test_get_dataset_paths_all(self):
        """Test getting paths for all datasets."""
        config = RetrievalConfig(dataset_type=DatasetType.ALL)
        paths = config.get_dataset_paths()

        assert len(paths) == 1
        names = {p.name for p in paths}
        assert names == {
            "agents_eng.jsonl",
        }

    def test_get_cache_path(self):
        """Test cache path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RetrievalConfig(
                dataset_type=DatasetType.ENG,
                cache_dir=Path(tmpdir) / "cache",
            )

            cache_path = config.get_cache_path()

            assert cache_path.parent.exists()
            assert "agents_eng" in cache_path.name
            assert cache_path.suffix == ".pt"  # PyTorch format

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = RetrievalConfig(
            dataset_type=DatasetType.ENG,
            top_k=10,
            similarity_threshold=0.5,
            use_cached_index=False,
            deduplicate_results=False,
            embedding_model=EmbeddingModel.BGE_LARGE.value,
            use_reranker=True,
            reranker_model=RerankerModel.BGE_RERANKER_LARGE.value,
            device="cpu",
        )

        assert config.dataset_type == DatasetType.ENG.value
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.use_cached_index is False
        assert config.deduplicate_results is False
        assert config.embedding_model == EmbeddingModel.BGE_LARGE.value
        assert config.use_reranker is True
        assert config.device == "cpu"

    def test_pydantic_validation(self):
        """Test that Pydantic validation works."""
        # top_k must be >= 1
        with pytest.raises(Exception):  # ValidationError
            RetrievalConfig(top_k=0)

        # similarity_threshold must be between 0 and 1
        with pytest.raises(Exception):
            RetrievalConfig(similarity_threshold=1.5)
