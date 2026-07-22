"""Embedding and reranking module using PyTorch."""

from pathlib import Path
from typing import Literal

import torch

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def get_device(device: Literal["auto", "cuda", "cpu", "mps"] = "auto") -> torch.device:
    """Get the appropriate torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class BaseEmbedder:
    """Base class for embedders."""

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to embeddings."""
        raise NotImplementedError

    def encode_query(self, text: str) -> torch.Tensor:
        """Encode a single query."""
        raise NotImplementedError

    def similarity(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and documents."""
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers models with PyTorch."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Literal["auto", "cuda", "cpu", "mps"] = "auto",
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            msg = "sentence-transformers is required. Install with: pip install sentence-transformers"
            raise ImportError(msg)
        self.device = get_device(device)
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model_name = model_name

    def encode(self, texts: list[str], show_progress: bool = True) -> torch.Tensor:
        """Encode texts to dense embeddings as PyTorch tensor."""
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=show_progress and len(texts) > 100,
            device=str(self.device),
        )

    def encode_query(self, text: str) -> torch.Tensor:
        """Encode a single query."""
        return self.model.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=str(self.device),
        )

    def similarity(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity using PyTorch."""
        # Normalize embeddings
        query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        doc_norms = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

        # Compute cosine similarity
        return torch.mm(query_norm, doc_norms.T).squeeze(0)


class TfidfEmbedder(BaseEmbedder):
    """Fallback embedder using TF-IDF (no neural models required)."""

    def __init__(self):
        if not HAS_SKLEARN:
            msg = "scikit-learn is required for TF-IDF. Install with: pip install scikit-learn"
            raise ImportError(msg)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._fitted = False
        self.device = torch.device("cpu")  # TF-IDF always on CPU

    def fit(self, texts: list[str]) -> None:
        """Fit the vectorizer on texts."""
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str], fit: bool = False) -> torch.Tensor:
        """Encode texts to TF-IDF vectors as PyTorch tensor."""
        if fit or not self._fitted:
            self.fit(texts)
        vectors = self.vectorizer.transform(texts).toarray()
        return torch.tensor(vectors, dtype=torch.float32)

    def encode_query(self, text: str) -> torch.Tensor:
        """Encode a single query using the fitted vectorizer."""
        if not self._fitted:
            msg = "Vectorizer must be fitted before encoding queries"
            raise ValueError(msg)
        vector = self.vectorizer.transform([text]).toarray()[0]
        return torch.tensor(vector, dtype=torch.float32)

    def similarity(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity."""
        # Normalize
        query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        doc_norms = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

        return torch.mm(query_norm, doc_norms.T).squeeze(0)


class Reranker:
    """Cross-encoder reranker for two-stage retrieval."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Literal["auto", "cuda", "cpu", "mps"] = "auto",
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            msg = "sentence-transformers is required for reranking. Install with: pip install sentence-transformers"
            raise ImportError(msg)
        self.device = get_device(device)
        self.model = CrossEncoder(model_name, device=str(self.device))
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The query text
            documents: List of document texts to rerank
            top_k: Return only top-k results (None = all)

        Returns:
            List of (original_index, score) tuples sorted by score descending

        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=len(pairs) > 50)

        # Create (index, score) pairs and sort by score descending
        indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores


def get_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto",
) -> BaseEmbedder:
    """
    Get the appropriate embedder based on available dependencies.

    Tries sentence-transformers first, falls back to TF-IDF.
    """
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            return SentenceTransformerEmbedder(model_name, device)
        except Exception:
            pass

    if HAS_SKLEARN:
        return TfidfEmbedder()

    msg = (
        "No embedding library available. Install either:\n"
        "  - sentence-transformers: pip install sentence-transformers\n"
        "  - scikit-learn: pip install scikit-learn"
    )
    raise ImportError(msg)


class EmbeddingIndex:
    """Index for fast retrieval using embeddings with PyTorch."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.embeddings: torch.Tensor | None = None
        self.texts: list[str] = []

    def build(self, texts: list[str]) -> None:
        """Build the index from texts."""
        self.texts = texts

        # For TF-IDF, we need to fit and transform
        if isinstance(self.embedder, TfidfEmbedder):
            self.embeddings = self.embedder.encode(texts, fit=True)
        else:
            self.embeddings = self.embedder.encode(texts)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Search the index for similar texts.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples

        """
        if self.embeddings is None:
            msg = "Index not built. Call build() first."
            raise ValueError(msg)

        # Encode query
        if isinstance(self.embedder, TfidfEmbedder):
            query_embedding = self.embedder.encode_query(query)
        else:
            query_embedding = self.embedder.encode_query(query)

        # Compute similarities
        similarities = self.embedder.similarity(query_embedding, self.embeddings)

        # Get top-k indices using torch
        k = min(top_k, len(self.texts))
        top_scores, top_indices = torch.topk(similarities, k)

        return [(int(idx), float(score)) for idx, score in zip(top_indices, top_scores, strict=False)]

    def save(self, path: Path) -> None:
        """Save the index to disk using PyTorch."""
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "embeddings": self.embeddings,
            "texts": self.texts,
            "embedder_type": type(self.embedder).__name__,
        }

        # For TF-IDF, also save the fitted vectorizer
        if isinstance(self.embedder, TfidfEmbedder):
            save_data["vectorizer"] = self.embedder.vectorizer

        torch.save(save_data, path)

    def load(self, path: Path) -> bool:
        """
        Load the index from disk.

        Returns:
            True if loaded successfully, False otherwise

        """
        if not path.exists():
            return False

        try:
            # Load with weights_only=False for TF-IDF vectorizer compatibility
            data = torch.load(path, map_location="cpu", weights_only=False)

            # Check if embedder type matches
            saved_type = data.get("embedder_type", "")
            current_type = type(self.embedder).__name__

            if saved_type != current_type:
                return False

            self.embeddings = data["embeddings"]
            self.texts = data["texts"]

            # Move embeddings to the right device
            if hasattr(self.embedder, "device"):
                self.embeddings = self.embeddings.to(self.embedder.device)

            # For TF-IDF, restore the fitted vectorizer
            if isinstance(self.embedder, TfidfEmbedder) and "vectorizer" in data:
                self.embedder.vectorizer = data["vectorizer"]
                self.embedder._fitted = True

            return True
        except Exception:
            return False
