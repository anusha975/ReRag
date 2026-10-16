"""CLI Runner for Reproducible Empirical Multi-Method Comparison Study.

Executes a controlled benchmark comparing:
1. Fixed-Parameter Baseline RAG
2. Greedy Coordinate Search (Original ReRAG Baseline)
3. Random Search
4. Simulated Annealing

Guarantees:
- Strict zero-leakage: parameter optimization only accesses training questions.
- Held-out test evaluation occurs strictly post-selection.
- Complete output artifacts saved to results/ (JSON, summary CSV, trials CSV).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.comparison_runner import ComparisonResult, ComparisonRunner
from src.config import Settings, get_settings
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.evaluator import Evaluator
from src.llm import BaseLLMClient, MockLLMClient, OllamaClient
from src.optimizer import ChunkConfig, OptimizationObjective, SearchBounds
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore
from tests.conftest import DeterministicMockEmbeddingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("comparison_study")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible multi-method ReRAG comparison study.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/expert_qa_dataset.json",
        help="Path to evaluation QA dataset file.",
    )
    parser.add_argument(
        "--document",
        type=str,
        default="data/documents/ReRAG_Test_Knowledge_Document.pdf",
        help="Path to document file to chunk and index.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=6,
        help="Trial budget per optimization strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Run in deterministic offline mode using mock embedding & LLM.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Run using live SentenceTransformers embeddings and local Ollama LLM.",
    )

    args = parser.parse_args()

    # Resolve settings and paths
    settings = get_settings()
    settings.paths.ensure_directories()

    doc_path = Path(args.document)
    if not doc_path.exists():
        fallback_pdf = settings.paths.documents_dir / "sample_rerag_paper.pdf"
        if fallback_pdf.exists():
            doc_path = fallback_pdf
        else:
            logger.error("No valid document found at %s", doc_path)
            return 1

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        fallback_ds = settings.paths.evaluation_dir / "synthetic_demonstration_dataset.json"
        if fallback_ds.exists():
            dataset_path = fallback_ds
        else:
            logger.error("No valid dataset found at %s", dataset_path)
            return 1

    logger.info("Loading document: %s", doc_path)
    loader = DocumentLoader()
    doc = loader.load_file(doc_path)

    use_offline = False if args.online else True
    logger.info("Execution mode: %s", "Offline (Deterministic)" if use_offline else "Online (Live Ollama)")

    if use_offline:
        emb_manager = DeterministicMockEmbeddingManager()
        llm = MockLLMClient(
            default_response="ReRAG optimizes chunk size and overlap to maximize factual faithfulness and minimize hallucinations."
        )
    else:
        emb_manager = EmbeddingManager()
        llm = OllamaClient(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            timeout=settings.llm.timeout,
        )

    # Isolated ephemeral vector store
    col_name = f"study_{int(time.time())}"
    vs = VectorStore(persist_dir=None, collection_name=col_name, embedding_manager=emb_manager)
    retriever = Retriever(vector_store=vs, embedding_manager=emb_manager, top_k=settings.retriever.top_k, similarity_threshold=0.0)
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
    evaluator = Evaluator(pipeline=pipeline, embedding_manager=emb_manager)

    runner = ComparisonRunner(
        settings=settings,
        documents=[doc],
        dataset_path=dataset_path,
        train_ratio=0.50,
        val_ratio=0.20,
        test_ratio=0.30,
        random_seed=args.seed,
        embedding_manager=emb_manager,
        llm_client=llm,
        pipeline=pipeline,
        evaluator=evaluator,
        collection_name=col_name,
        results_dir=settings.paths.results_dir,
    )

    bounds = SearchBounds(
        min_chunk_size=100,
        max_chunk_size=1200,
        min_chunk_overlap=0,
        max_chunk_overlap=300,
        min_step_size=25,
    )
    objective = OptimizationObjective(
        weights={
            "faithfulness_ratio": 0.40,
            "context_coverage": 0.30,
            "semantic_similarity": 0.30,
        }
    )

    logger.info("Starting multi-method comparison study with budget=%d, seed=%d", args.budget, args.seed)
    comp_result: ComparisonResult = runner.run(
        baseline_config=ChunkConfig(500, 50),
        bounds=bounds,
        objective=objective,
        matched_budget=args.budget,
        save_results=True,
    )

    print("\n" + "=" * 80)
    print("REPRODUCIBLE EMPIRICAL COMPARISON RESULTS")
    print("=" * 80)
    print(f"Dataset:       {dataset_path.name}")
    print(f"Document:      {doc_path.name}")
    print(f"Best Test (Held-Out Generalization): {comp_result.best_test_method}")
    print(f"Best Train (Tuning Performance):    {comp_result.best_train_method}")
    print("-" * 80)

    import pandas as pd
    df = pd.DataFrame(comp_result.summary_records)
    print(df.to_string(index=False))
    print("=" * 80)

    saved_files = list(settings.paths.results_dir.glob(f"{comp_result.experiment_id}*"))
    print(f"\nSaved {len(saved_files)} artifacts to {settings.paths.results_dir}:")
    for f in saved_files:
        print(f" - {f.name} ({f.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
