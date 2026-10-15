"""Vector database storage and indexing module using ChromaDB for ReRAG.

Manages persistent and ephemeral collections, handles chunk and document indexing,
guarantees query and document embedding consistency via EmbeddingManager,
prevents stale chunks on document re-indexing, and provides index rebuilding.

### Retrieval Metric Documentation
- **Distance Metric**: The ChromaDB collection uses `hnsw:space = 'cosine'`. ChromaDB returns Cosine Distance:
      d_cosine(u, v) = 1.0 - (u . v) / (||u||_2 * ||v||_2)
  Because vectors are L2-normalized (||u||_2 = 1.0), this simplifies to:
      d_cosine(u, v) = 1.0 - (u . v)  in range [0.0, 2.0]
  where d = 0.0 indicates identical orientation, d = 1.0 indicates orthogonality,
  and d = 2.0 indicates opposite orientation.

- **Similarity Score**: Exact Cosine Similarity:
      similarity = 1.0 - d_cosine = (u . v)  in range [-1.0, 1.0]
  A score of 1.0 represents maximal semantic relevance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
import numpy as np

from src.chunker import TextChunk
from src.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """Sanitize metadata values to satisfy ChromaDB primitive type constraints."""
    clean: Dict[str, Union[str, int, float, bool]] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple, dict)):
            clean[k] = json.dumps(v)
        else:
            clean[k] = str(v)
    return clean


class VectorStore:
    """ChromaDB interface for vector indexing, retrieval, and collection lifecycle."""

    def __init__(
        self,
        persist_dir: Optional[Union[Path, str]] = None,
        collection_name: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        distance_metric: str = "cosine",
        config: Optional[Any] = None,
    ) -> None:
        """Initialize ChromaDB vector store.

        Args:
            persist_dir: Directory where ChromaDB indices are stored. If None or ':memory:',
                uses an in-memory EphemeralClient.
            collection_name: Target collection identifier. Defaults to 'rerag_collection'.
            embedding_manager: Shared EmbeddingManager instance for document and query embeddings.
            distance_metric: Distance metric space ('cosine', 'l2', or 'ip'). Default is 'cosine'.
            config: Optional VectorStoreConfig dataclass instance to load parameters from.
        """
        cfg_persist = getattr(config, "persist_dir", None) if config else None
        cfg_col = getattr(config, "collection_name", None) if config else None

        raw_persist = persist_dir if persist_dir is not None else cfg_persist
        self.persist_dir = Path(raw_persist) if raw_persist and str(raw_persist) != ":memory:" else None
        self.collection_name = collection_name or cfg_col or "rerag_collection"
        self.distance_metric = distance_metric.lower()
        self.embedding_manager = embedding_manager or EmbeddingManager()

        # Initialize Chroma Client
        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client: ClientAPI = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self.client = chromadb.EphemeralClient()

        self._collection: Optional[Collection] = None
        self._init_collection()

    def _init_collection(self) -> Collection:
        """Get or create the underlying ChromaDB collection with configured distance metric."""
        metadata = {"hnsw:space": self.distance_metric}
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata,
        )
        return self._collection

    @property
    def collection(self) -> Collection:
        """Return the active collection instance."""
        if self._collection is None:
            return self._init_collection()
        return self._collection

    def create_collection(
        self,
        name: str,
        distance_metric: Optional[str] = None,
    ) -> Collection:
        """Create a new collection for a specific document or document set.

        Args:
            name: New collection identifier.
            distance_metric: Optional metric ('cosine', 'l2', or 'ip').

        Returns:
            Collection: Created ChromaDB collection.
        """
        metric = (distance_metric or self.distance_metric).lower()
        return self.client.create_collection(name=name, metadata={"hnsw:space": metric})

    def get_or_create_collection(
        self,
        name: str,
        distance_metric: Optional[str] = None,
    ) -> Collection:
        """Get existing collection or create one if absent.

        Args:
            name: Target collection identifier.
            distance_metric: Optional metric override.

        Returns:
            Collection: ChromaDB collection.
        """
        metric = (distance_metric or self.distance_metric).lower()
        return self.client.get_or_create_collection(name=name, metadata={"hnsw:space": metric})

    def switch_collection(
        self,
        name: str,
        distance_metric: Optional[str] = None,
    ) -> Collection:
        """Switch active working collection to another collection name.

        Args:
            name: Collection name to switch to.
            distance_metric: Optional distance metric override.

        Returns:
            Collection: Newly activated ChromaDB collection.
        """
        self.collection_name = name
        if distance_metric:
            self.distance_metric = distance_metric.lower()
        self._collection = self._init_collection()
        return self._collection

    def list_collections(self) -> List[str]:
        """List all collection names in the ChromaDB database."""
        cols = self.client.list_collections()
        names: List[str] = []
        for col in cols:
            if hasattr(col, "name"):
                names.append(col.name)
            else:
                names.append(str(col))
        return names

    def delete_collection(self, name: Optional[str] = None) -> None:
        """Delete a collection by name or delete the active collection.

        Args:
            name: Optional collection name to delete. If None, deletes the active collection.
        """
        target_name = name or self.collection_name
        try:
            self.client.delete_collection(name=target_name)
            logger.info("Deleted collection '%s'.", target_name)
        except Exception as exc:
            logger.warning("Error deleting collection '%s': %s", target_name, exc)
        if target_name == self.collection_name:
            self._collection = None

    def add_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Optional[Union[List[List[float]], np.ndarray]] = None,
        prevent_stale: bool = True,
    ) -> None:
        """Add text chunks and embeddings to the collection.

        Prevents stale chunks from remaining if a document is re-indexed.

        Args:
            chunks: Sequence of TextChunk objects to index.
            embeddings: Optional precomputed embeddings. If None, generated using
                the configured embedding_manager.
            prevent_stale: If True, deletes existing chunks with matching document IDs
                before inserting new chunks.
        """
        if not chunks:
            return

        # Prevent stale chunks: purge old chunks belonging to these document IDs
        if prevent_stale and self.count() > 0:
            doc_ids = {c.document_id for c in chunks if c.document_id}
            for doc_id in doc_ids:
                try:
                    self.delete_document(doc_id)
                except Exception as exc:
                    logger.debug("Stale chunk check for doc '%s': %s", doc_id, exc)

        # Generate embeddings if not explicitly provided
        if embeddings is None:
            logger.info("Computing embeddings for %d chunks...", len(chunks))
            computed_embeddings = self.embedding_manager.embed_chunks(chunks)
            emb_list = computed_embeddings.tolist()
        else:
            if isinstance(embeddings, np.ndarray):
                emb_list = embeddings.tolist()
            else:
                emb_list = [list(e) for e in embeddings]

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]

        # Ensure provenance metadata fields are explicitly stored
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata)
            if chunk.document_id and "document_id" not in meta:
                meta["document_id"] = chunk.document_id
            if chunk.chunk_id and "chunk_id" not in meta:
                meta["chunk_id"] = chunk.chunk_id
            if chunk.page_number is not None and "page_number" not in meta:
                meta["page_number"] = chunk.page_number
            if chunk.chunk_index is not None and "chunk_index" not in meta:
                meta["chunk_index"] = chunk.chunk_index
            metadatas.append(_sanitize_metadata(meta))

        # Insert into ChromaDB in batches of 500 to prevent payload limits
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end_idx = i + batch_size
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                embeddings=emb_list[i:end_idx],
            )

        logger.info(
            "Successfully indexed %d chunks in collection '%s'. Total: %d",
            len(chunks),
            self.collection_name,
            self.count(),
        )


    def query(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[Union[List[float], np.ndarray]] = None,
        top_k: int = 4,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform similarity search on the vector store.

        Ensures query embeddings use the identical embedding model as document embeddings.

        Args:
            query_text: Query string.
            query_embedding: Optional precomputed query embedding vector.
            top_k: Number of nearest neighbors to retrieve.
            where_filter: Optional metadata filtering dictionary.

        Returns:
            Dict[str, Any]: Query results with 'ids', 'documents', 'metadatas',
                'distances', and computed 'similarities'.
        """
        if self.count() == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "similarities": [[]],
            }

        k = max(1, min(top_k, self.count()))

        if query_embedding is None:
            if not query_text or not query_text.strip():
                return {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                    "similarities": [[]],
                }
            q_emb = self.embedding_manager.embed_query(query_text).tolist()
        else:
            if isinstance(query_embedding, np.ndarray):
                q_emb = query_embedding.tolist()
            else:
                q_emb = list(query_embedding)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [q_emb],
            "n_results": k,
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        raw_results = self.collection.query(**query_kwargs)

        # Compute cosine similarity: s = 1.0 - distance
        similarities: List[List[float]] = []
        for dist_list in raw_results.get("distances", [[]]):
            sim_list = [float(1.0 - d) for d in dist_list]
            similarities.append(sim_list)

        results = dict(raw_results)
        results["similarities"] = similarities
        return results

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks originating from a specified document ID.

        Args:
            document_id: Identifier of the document to purge.
        """
        if self.count() == 0:
            return
        self.collection.delete(where={"document_id": document_id})
        logger.info("Deleted chunks for document '%s'. Remaining: %d", document_id, self.count())

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Delete specific chunks by their chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to remove.
        """
        if not chunk_ids or self.count() == 0:
            return
        self.collection.delete(ids=list(chunk_ids))

    def count(self) -> int:
        """Return total number of stored chunks in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all indexed chunks in the active collection."""
        try:
            if self.count() > 0:
                existing = self.collection.get()
                if existing and existing.get("ids"):
                    self.collection.delete(ids=existing["ids"])
                    logger.info("Cleared %d chunks from collection '%s'.", len(existing["ids"]), self.collection_name)
                    return
        except Exception as exc:
            logger.debug("Falling back to collection recreation: %s", exc)

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._init_collection()
        logger.info("Cleared collection '%s'.", self.collection_name)


    def rebuild_index(self, chunks: Optional[Sequence[TextChunk]] = None) -> None:
        """Reset the collection and re-index a new set of chunks.

        Args:
            chunks: Optional new set of chunks to index after reset.
        """
        self.clear()
        if chunks:
            self.add_chunks(chunks)
