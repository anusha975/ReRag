"""Core RAG and ReRAG pipeline coordination module.

Implements baseline naive RAG execution alongside the ReRAG iterative retrieval,
verification, and hallucination reduction workflow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from src.chunker import TextChunker
from src.document_loader import LoadedDocument
from src.llm import BaseLLMClient, OllamaClient
from src.prompts import (
    INSUFFICIENT_INFORMATION_RESPONSE,
    RAG_SYSTEM_PROMPT,
    build_rag_prompt,
    format_retrieved_context,
)
from src.retriever import RetrievedContext, Retriever

logger = logging.getLogger(__name__)


@dataclass
class PipelineStepTrace:
    """Detailed audit trace for an individual iteration/step within the ReRAG loop."""

    step_index: int
    query: str
    retrieved_contexts: List[RetrievedContext]
    candidate_answer: str
    verification_result: Optional[Dict[str, Any]] = None
    refinement_query: Optional[str] = None


@dataclass
class PipelineResult:
    """Encapsulates the final result, intermediate audit traces, and metrics."""

    question: str
    final_answer: str
    contexts: List[RetrievedContext] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    retrieval_distances: List[float] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    steps: List[PipelineStepTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: Optional[float] = None
    llm_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a structured dictionary consumable by dashboard and evaluators."""
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "contexts": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "score": c.score,
                    "distance": c.distance,
                    "citation": c.citation,
                    "page_number": c.page_number,
                    "document_id": c.document_id,
                    "metadata": c.metadata,
                }
                for c in self.contexts
            ],
            "source_documents": self.source_documents,
            "page_numbers": self.page_numbers,
            "retrieval_scores": self.retrieval_scores,
            "retrieval_distances": self.retrieval_distances,
            "citations": self.citations,
            "metadata": self.metadata,
            "execution_time_seconds": self.execution_time_seconds,
            "llm_model": self.llm_model,
        }


class RAGPipeline:
    """Coordinates retrieval, context assembly, and grounded LLM answer generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: Union[BaseLLMClient, OllamaClient],
        max_refinement_steps: int = 3,
        verification_threshold: float = 0.7,
        config: Optional[Any] = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            retriever: Configured passage retriever.
            llm_client: LLM client interface (OllamaClient or any BaseLLMClient).
            max_refinement_steps: Maximum iterative refinement cycles.
            verification_threshold: Threshold above which an answer is deemed grounded.
            config: Optional ReRAGConfig dataclass instance.
        """
        cfg_max = getattr(config, "max_refinement_steps", None) if config else None
        cfg_ver = getattr(config, "verification_threshold", None) if config else None

        self.retriever = retriever
        self.llm_client = llm_client
        self.max_refinement_steps = cfg_max if cfg_max is not None else max_refinement_steps
        self.verification_threshold = cfg_ver if cfg_ver is not None else verification_threshold

    def index_documents(
        self,
        documents: Sequence[LoadedDocument],
        chunker: Optional[TextChunker] = None,
        collection_name: Optional[str] = None,
    ) -> int:
        """Index a set of loaded documents into the pipeline's vector store.

        Args:
            documents: Sequence of LoadedDocument objects.
            chunker: Optional TextChunker. If None, default TextChunker is used.
            collection_name: Optional collection identifier to index into.

        Returns:
            int: Number of chunks indexed.
        """
        if collection_name:
            self.retriever.vector_store.switch_collection(collection_name)
        active_chunker = chunker or TextChunker()
        return self.retriever.rebuild_index_from_documents(documents, active_chunker)

    def run_baseline(
        self,
        question: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Execute single-hop grounded baseline RAG.

        1. Accept user question and target document collection.
        2. Retrieve relevant chunks.
        3. Construct context from those chunks.
        4. Generate answer using configured LLM.
        5. Return structured result with citations, page numbers, and retrieval scores.

        Args:
            question: Target query.
            collection_name: Optional collection to query against.
            top_k: Optional override for number of chunks to retrieve.
            threshold: Optional override for similarity threshold.
            where_filter: Optional metadata filtering dictionary for ChromaDB.
            similarity_threshold: Optional alias for threshold parameter.
            **kwargs: Additional parameters safely absorbed.

        Returns:
            PipelineResult: Traceable answer, retrieved contexts, and citation metadata.
        """
        start_time = time.perf_counter()
        cleaned_question = question.strip() if question else ""

        # 1. Handle empty question safely
        if not cleaned_question:
            return PipelineResult(
                question=question,
                final_answer=INSUFFICIENT_INFORMATION_RESPONSE,
                contexts=[],
                source_documents=[],
                page_numbers=[],
                retrieval_scores=[],
                retrieval_distances=[],
                citations=[],
                metadata={"status": "empty_question", "retrieval_count": 0},
                execution_time_seconds=time.perf_counter() - start_time,
                llm_model=getattr(self.llm_client, "model", "unknown"),
            )

        # 2. Switch collection if requested
        if collection_name:
            self.retriever.vector_store.switch_collection(collection_name)

        # 3. Retrieve relevant chunks
        effective_threshold = threshold if threshold is not None else similarity_threshold
        contexts = self.retriever.retrieve(
            query=cleaned_question,
            top_k=top_k,
            threshold=effective_threshold,
            where_filter=where_filter,
        )

        # 4. Handle empty retrieval results safely without hallucinating
        if not contexts:
            logger.info("No matching passages found for query '%s'.", cleaned_question)
            return PipelineResult(
                question=cleaned_question,
                final_answer=INSUFFICIENT_INFORMATION_RESPONSE,
                contexts=[],
                source_documents=[],
                page_numbers=[],
                retrieval_scores=[],
                retrieval_distances=[],
                citations=[],
                metadata={"status": "empty_retrieval", "retrieval_count": 0},
                execution_time_seconds=time.perf_counter() - start_time,
                llm_model=getattr(self.llm_client, "model", "unknown"),
            )

        # 5. Construct context and format prompt
        user_prompt = build_rag_prompt(question=cleaned_question, contexts=contexts)

        # 6. Generate answer using configured LLM
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
        )
        final_answer = response.content.strip()

        # 7. Extract traceable provenance without fabricating citations
        source_docs = sorted(list({c.document_id for c in contexts if c.document_id}))
        page_nums = sorted(list({c.page_number for c in contexts if c.page_number is not None}))
        retrieval_scores = [c.score for c in contexts]
        retrieval_distances = [c.distance for c in contexts]
        citations = [c.citation for c in contexts]

        elapsed = time.perf_counter() - start_time

        return PipelineResult(
            question=cleaned_question,
            final_answer=final_answer,
            contexts=list(contexts),
            source_documents=source_docs,
            page_numbers=page_nums,
            retrieval_scores=retrieval_scores,
            retrieval_distances=retrieval_distances,
            citations=citations,
            metadata={
                "status": "success",
                "retrieval_count": len(contexts),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            },
            execution_time_seconds=elapsed,
            llm_model=getattr(self.llm_client, "model", "unknown"),
        )

    def run(self, question: str, **kwargs: Any) -> PipelineResult:
        """Convenience alias for run_baseline."""
        return self.run_baseline(question, **kwargs)

    def run_rerag(self, question: str) -> PipelineResult:
        """Execute iterative ReRAG pipeline with verification and refinement.

        Args:
            question: Target query.

        Returns:
            PipelineResult: Refined answer with full step traces.
        """
        raise NotImplementedError("RAGPipeline.run_rerag is scheduled for subsequent implementation phases.")

