"""Information retrieval and context scoring module for ReRAG.

Retrieves top-k relevant document chunks, scores their semantic relevance,
preserves source page citations, filters by similarity thresholds, and supports
rebuilding retrieval indices when chunking hyperparameters change.

### Retrieval Metric Documentation
- **Distance Metric**: The underlying ChromaDB collection is configured with
  `hnsw:space = 'cosine'`. ChromaDB returns Cosine Distance:
      d_cosine(u, v) = 1.0 - (u . v) / (||u||_2 * ||v||_2)
  Because the EmbeddingManager normalizes vectors to unit L2 norm (||u||_2 = 1),
  this simplifies to:
      d_cosine(u, v) = 1.0 - (u . v)  in range [0.0, 2.0]
  where d = 0.0 denotes identical direction and d = 2.0 denotes diametric opposition.

- **Similarity Score**: The returned `score` is exact Cosine Similarity:
      score = 1.0 - d_cosine = (u . v)  in range [-1.0, 1.0]
  A score of 1.0 represents maximal semantic relevance.
  Scores are computed directly from the vector distance without artificial scaling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.chunker import TextChunk, TextChunker
from src.document_loader import LoadedDocument
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedContext:
    """Represents a retrieved passage with cosine similarity score, distance, and citation metadata."""

    chunk_id: str
    text: str
    score: float
    distance: float
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    document_id: Optional[str] = None

    @property
    def citation(self) -> str:
        """Format a human-readable citation label for LLM attribution."""
        doc = self.document_id or "Doc"
        if self.page_number is not None:
            return f"[{doc}, p. {self.page_number}]"
        pages = self.metadata.get("page_numbers")
        if pages and isinstance(pages, list) and len(pages) > 0:
            if len(pages) == 1:
                return f"[{doc}, p. {pages[0]}]"
            return f"[{doc}, pp. {pages[0]}-{pages[-1]}]"
        return f"[{doc}]"


class Retriever:
    """Coordinates embedding generation and vector store search to fetch relevant context."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: Optional[EmbeddingManager] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        config: Optional[Any] = None,
    ) -> None:
        """Initialize retriever.

        Args:
            vector_store: Vector database instance.
            embedding_manager: Shared embedding generator. Defaults to vector_store's manager.
            top_k: Default number of chunks to retrieve. Defaults to 4.
            similarity_threshold: Minimum acceptable cosine similarity score. Defaults to 0.3.
            config: Optional RetrieverConfig dataclass instance to load parameters from.
        """
        cfg_top_k = getattr(config, "top_k", None) if config else None
        cfg_thresh = getattr(config, "similarity_threshold", None) if config else None

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager or vector_store.embedding_manager
        self.top_k = max(1, top_k or cfg_top_k or 4)
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else (cfg_thresh if cfg_thresh is not None else 0.3)
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """Retrieve the top-k most relevant chunks for a given query string.

        Safely handles empty queries, empty collections, and non-existent results.

        Args:
            query: User search query.
            top_k: Optional override for top-k.
            threshold: Optional override for similarity threshold.
            where_filter: Optional metadata filtering dictionary for ChromaDB.

        Returns:
            List[RetrievedContext]: Filtered and ranked context passages with citations.
        """
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            logger.debug("Empty query passed to Retriever. Returning empty results.")
            return []

        if self.vector_store.count() == 0:
            logger.debug("Retriever called against an empty vector collection.")
            return []

        k = top_k if top_k is not None and top_k > 0 else self.top_k
        min_threshold = threshold if threshold is not None else self.similarity_threshold

        # Query vector store using query text (uses shared embedding model)
        results = self.vector_store.query(
            query_text=cleaned_query,
            top_k=k,
            where_filter=where_filter,
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        similarities = results.get("similarities", [[]])[0]

        retrieved: List[RetrievedContext] = []

        for cid, doc_text, meta, dist, sim in zip(ids, docs, metadatas, distances, similarities):
            sim_score = float(sim)
            dist_val = float(dist)

            # Apply similarity threshold filter
            if sim_score < min_threshold:
                continue

            # Unpack metadata
            clean_meta = dict(meta or {})
            # Deserialise JSON strings for list metadata if needed
            if "page_numbers" in clean_meta and isinstance(clean_meta["page_numbers"], str):
                try:
                    clean_meta["page_numbers"] = json.loads(clean_meta["page_numbers"])
                except Exception:
                    pass

            page_num = clean_meta.get("page_number")
            if page_num is None and "page_numbers" in clean_meta and isinstance(clean_meta["page_numbers"], list) and clean_meta["page_numbers"]:
                page_num = clean_meta["page_numbers"][0]

            doc_id = clean_meta.get("document_id")

            retrieved.append(
                RetrievedContext(
                    chunk_id=cid,
                    text=doc_text,
                    score=sim_score,
                    distance=dist_val,
                    metadata=clean_meta,
                    page_number=int(page_num) if page_num is not None else None,
                    document_id=str(doc_id) if doc_id is not None else None,
                )
            )

        # Ensure sorted in descending order of similarity
        retrieved.sort(key=lambda x: x.score, reverse=True)
        return retrieved

    def rebuild_index_from_documents(
        self,
        documents: Sequence[LoadedDocument],
        chunker: TextChunker,
    ) -> int:
        """Rebuild the retrieval index from documents using a new chunker configuration.

        Args:
            documents: Sequence of LoadedDocument objects.
            chunker: Configured TextChunker with updated chunk_size or chunk_overlap.

        Returns:
            int: Total number of chunks indexed in the rebuilt collection.
        """
        logger.info(
            "Rebuilding retrieval index with chunk_size=%d, overlap=%d...",
            chunker.chunk_size,
            chunker.chunk_overlap,
        )
        all_chunks: List[TextChunk] = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        self.vector_store.rebuild_index(all_chunks)
        logger.info("Rebuilt index with %d chunks.", len(all_chunks))
        return len(all_chunks)

    def rebuild_index_from_chunks(
        self,
        chunks: Sequence[TextChunk],
    ) -> int:
        """Rebuild the retrieval index directly from precomputed TextChunk objects.

        Args:
            chunks: Sequence of TextChunk objects to index.

        Returns:
            int: Total number of chunks indexed in the rebuilt collection.
        """
        self.vector_store.rebuild_index(chunks)
        logger.info("Rebuilt index directly from %d chunks.", len(chunks))
        return len(chunks)

