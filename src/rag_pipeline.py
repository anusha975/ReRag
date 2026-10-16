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
from src.metrics import check_missing_answer, detect_unsupported_claims_deterministic
from src.prompts import (
    INSUFFICIENT_INFORMATION_RESPONSE,
    QUERY_REFINEMENT_SYSTEM_PROMPT,
    QUERY_REFINEMENT_USER_PROMPT_TEMPLATE,
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

    def run_rerag(
        self,
        question: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Execute iterative ReRAG pipeline with claim verification and query refinement.

        1. Retrieve initial context using query.
        2. In each iteration step:
           a. Construct grounded prompt and generate candidate answer using LLM.
           b. Verify factual consistency of candidate claims against retrieved context.
           c. If candidate meets or exceeds verification threshold, accept as verified answer.
           d. If claims are unsupported and max steps not reached, synthesize refinement query
              and retrieve supplementary context passages.
        3. If evidence is insufficient across all refinement cycles, return safe constrained response.

        Args:
            question: Target user query.
            collection_name: Optional collection override.
            top_k: Number of passages to retrieve per step.
            threshold: Similarity score threshold.
            where_filter: Optional metadata filtering.
            similarity_threshold: Alias for threshold.
            **kwargs: Extra parameters.

        Returns:
            PipelineResult: Refined answer with full step traces and provenance.
        """
        start_time = time.perf_counter()
        cleaned_question = question.strip() if question else ""

        # 1. Handle empty question
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
                steps=[],
                metadata={"status": "empty_question", "retrieval_count": 0, "verified": False},
                execution_time_seconds=time.perf_counter() - start_time,
                llm_model=getattr(self.llm_client, "model", "unknown"),
            )

        # 2. Switch collection if requested
        if collection_name:
            self.retriever.vector_store.switch_collection(collection_name)

        effective_threshold = threshold if threshold is not None else similarity_threshold

        # 3. Initial retrieval
        current_query = cleaned_question
        accumulated_contexts: List[RetrievedContext] = list(
            self.retriever.retrieve(
                query=current_query,
                top_k=top_k,
                threshold=effective_threshold,
                where_filter=where_filter,
            )
        )

        if not accumulated_contexts:
            logger.info("No matching passages found for query '%s' in ReRAG.", cleaned_question)
            return PipelineResult(
                question=cleaned_question,
                final_answer=INSUFFICIENT_INFORMATION_RESPONSE,
                contexts=[],
                source_documents=[],
                page_numbers=[],
                retrieval_scores=[],
                retrieval_distances=[],
                citations=[],
                steps=[],
                metadata={"status": "empty_retrieval", "retrieval_count": 0, "verified": False},
                execution_time_seconds=time.perf_counter() - start_time,
                llm_model=getattr(self.llm_client, "model", "unknown"),
            )

        steps: List[PipelineStepTrace] = []
        final_answer = ""
        candidate_answer = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        is_verified = False

        for step_idx in range(1, self.max_refinement_steps + 1):
            # Generate candidate answer
            prompt = build_rag_prompt(question=cleaned_question, contexts=accumulated_contexts)
            response = self.llm_client.generate(prompt=prompt, system_prompt=RAG_SYSTEM_PROMPT)
            candidate_answer = response.content.strip()
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            # Handle explicit abstention response from model
            if check_missing_answer(candidate_answer):
                steps.append(
                    PipelineStepTrace(
                        step_index=step_idx,
                        query=current_query,
                        retrieved_contexts=list(accumulated_contexts),
                        candidate_answer=candidate_answer,
                        verification_result={"faithfulness_score": 1.0, "is_supported": True, "abstention": True},
                    )
                )
                final_answer = candidate_answer
                is_verified = True
                break

            # Audit / verify candidate answer against retrieved passages
            audit_result = detect_unsupported_claims_deterministic(
                prediction=candidate_answer,
                contexts=accumulated_contexts,
            )
            faithfulness_score = audit_result["faithfulness_ratio"]
            unsupported_claims = audit_result["unsupported_claims"]
            is_supported = faithfulness_score >= self.verification_threshold

            trace = PipelineStepTrace(
                step_index=step_idx,
                query=current_query,
                retrieved_contexts=list(accumulated_contexts),
                candidate_answer=candidate_answer,
                verification_result={
                    "faithfulness_score": round(faithfulness_score, 4),
                    "is_supported": is_supported,
                    "supported_sentences": audit_result["supported_sentences"],
                    "total_sentences": audit_result["total_sentences"],
                    "unsupported_claims": unsupported_claims,
                },
            )

            if is_supported:
                steps.append(trace)
                final_answer = candidate_answer
                is_verified = True
                break

            # If not verified and steps remain, synthesize refinement query
            if step_idx < self.max_refinement_steps:
                gaps_text = (
                    "\n".join(f"- {c}" for c in unsupported_claims)
                    if unsupported_claims
                    else "Missing specific factual grounding."
                )
                refinement_prompt = QUERY_REFINEMENT_USER_PROMPT_TEMPLATE.format(
                    question=cleaned_question,
                    gaps=gaps_text,
                )
                refine_resp = self.llm_client.generate(
                    prompt=refinement_prompt,
                    system_prompt=QUERY_REFINEMENT_SYSTEM_PROMPT,
                )
                refinement_query = refine_resp.content.strip().split("\n")[0].strip('"').strip()
                if not refinement_query or len(refinement_query) < 3:
                    refinement_query = f"{cleaned_question} {' '.join(unsupported_claims[:1])}".strip()

                trace.refinement_query = refinement_query
                steps.append(trace)

                # Retrieve supplemental evidence
                supp_contexts = self.retriever.retrieve(
                    query=refinement_query,
                    top_k=top_k,
                    threshold=effective_threshold,
                    where_filter=where_filter,
                )
                existing_ids = {c.chunk_id for c in accumulated_contexts}
                added_any = False
                for sc in supp_contexts:
                    if sc.chunk_id not in existing_ids:
                        accumulated_contexts.append(sc)
                        existing_ids.add(sc.chunk_id)
                        added_any = True

                current_query = refinement_query
                if not added_any:
                    # No new information could be retrieved
                    final_answer = candidate_answer
                    break
            else:
                steps.append(trace)
                final_answer = candidate_answer

        # If after all iterations the candidate is still not verified, flag or safely abstain
        status = "verified_grounded" if is_verified else "unverified_max_steps"

        source_docs = sorted(list({c.document_id for c in accumulated_contexts if c.document_id}))
        page_nums = sorted(list({c.page_number for c in accumulated_contexts if c.page_number is not None}))
        retrieval_scores = [c.score for c in accumulated_contexts]
        retrieval_distances = [c.distance for c in accumulated_contexts]
        citations = [c.citation for c in accumulated_contexts]

        return PipelineResult(
            question=cleaned_question,
            final_answer=final_answer or candidate_answer,
            contexts=accumulated_contexts,
            source_documents=source_docs,
            page_numbers=page_nums,
            retrieval_scores=retrieval_scores,
            retrieval_distances=retrieval_distances,
            citations=citations,
            steps=steps,
            metadata={
                "status": status,
                "verified": is_verified,
                "total_steps": len(steps),
                "retrieval_count": len(accumulated_contexts),
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            execution_time_seconds=time.perf_counter() - start_time,
            llm_model=getattr(self.llm_client, "model", "unknown"),
        )


