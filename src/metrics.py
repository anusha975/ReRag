"""Evaluation metrics computation module for ReRAG.

Calculates deterministic quantitative metrics including semantic similarity,
lexical token F1, retrieval recall/precision, context coverage, unsupported claim checks,
and dataset-wide aggregate metrics.

### Metric Documentation & Scientific Limitations:
1. **Semantic Similarity (`compute_semantic_similarity`)**:
   - *Method*: Cosine similarity between L2-normalized SentenceTransformer embeddings of
     the predicted answer and reference answer: cos(u, v) = u . v in [-1.0, 1.0].
   - *Limitation*: Measures topical, conceptual, and stylistic alignment.
     **It is NOT proof of factual correctness**. A fluent, convincing hallucination can
     score high semantic similarity while stating completely false facts.
2. **Token F1 (`compute_token_f1`)**:
   - *Method*: Harmonic mean of unigram precision and recall against reference answer.
   - *Limitation*: Sensitive to phrasing differences and synonyms. High F1 does not guarantee
     logical consistency or absence of negation errors.
3. **Retrieval Recall & Precision (`compute_retrieval_recall_at_k`, `compute_retrieval_precision_at_k`)**:
   - *Method*: Proportion of known ground-truth supporting document or chunk IDs retrieved
     within top-k cutoff rank.
   - *Limitation*: Only computable when ground-truth relevant document IDs are annotated.
     Reported as `None` (unavailable) otherwise.
4. **Context Coverage (`compute_context_coverage`)**:
   - *Method*: Lexical overlap ratio between non-stopword tokens in the predicted answer
     and the retrieved context passages.
   - *Limitation*: Heuristic lexical proxy. Does not confirm syntactic or semantic entailment.
5. **Unsupported Claim Detection (`detect_unsupported_claims_deterministic`)**:
   - *Method*: Segment-based sentence grounding against context tokens.
   - *Limitation*: Heuristic; cannot identify complex deductions or nuanced factual shifts.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.embeddings import EmbeddingManager
from src.llm import BaseLLMClient
from src.prompts import INSUFFICIENT_INFORMATION_RESPONSE, VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Standard English stopwords for lexical overlap filtering
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and ground truth reference.

    Args:
        prediction: Generated text.
        reference: Reference ground-truth text.

    Returns:
        float: Token F1 score between 0.0 and 1.0.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(ref_tokens))

    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def compute_semantic_similarity(
    prediction: str,
    reference: str,
    embedding_manager: Optional[EmbeddingManager] = None,
) -> float:
    """Compute dense embedding cosine similarity between prediction and reference answer.

    Note: High semantic similarity indicates conceptual/topical alignment, NOT verified
    factual correctness.

    Args:
        prediction: Generated answer string.
        reference: Reference ground truth string.
        embedding_manager: Optional EmbeddingManager instance.

    Returns:
        float: Cosine similarity score between -1.0 and 1.0.
    """
    p_clean = prediction.strip() if prediction else ""
    r_clean = reference.strip() if reference else ""

    if not p_clean and not r_clean:
        return 1.0
    if not p_clean or not r_clean:
        return 0.0

    if p_clean.lower() == r_clean.lower():
        return 1.0

    manager = embedding_manager or EmbeddingManager()
    emb_pred = manager.embed_query(p_clean)
    emb_ref = manager.embed_query(r_clean)

    norm_p = np.linalg.norm(emb_pred)
    norm_r = np.linalg.norm(emb_ref)

    if norm_p == 0.0 or norm_r == 0.0:
        return 0.0

    similarity = float(np.dot(emb_pred, emb_ref) / (norm_p * norm_r))
    return float(np.clip(similarity, -1.0, 1.0))


def compute_retrieval_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Optional[Sequence[str]],
    k: int,
) -> Optional[float]:
    """Compute retrieval recall at cutoff rank k against known relevant IDs.

    Args:
        retrieved_ids: Ordered sequence of retrieved document or chunk IDs.
        relevant_ids: Set or sequence of known relevant IDs. If None or empty, returns None.
        k: Cutoff rank.

    Returns:
        Optional[float]: Recall@k between 0.0 and 1.0, or None if relevant_ids is unavailable.
    """
    if relevant_ids is None or len(relevant_ids) == 0:
        return None
    cutoff = max(1, k)
    top_k = retrieved_ids[:cutoff]
    matched = set(top_k) & set(relevant_ids)
    return float(len(matched) / len(set(relevant_ids)))


def compute_retrieval_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Optional[Sequence[str]],
    k: int,
) -> Optional[float]:
    """Compute retrieval precision at cutoff rank k against known relevant IDs.

    Args:
        retrieved_ids: Ordered sequence of retrieved document or chunk IDs.
        relevant_ids: Set or sequence of known relevant IDs. If None or empty, returns None.
        k: Cutoff rank.

    Returns:
        Optional[float]: Precision@k between 0.0 and 1.0, or None if relevant_ids is unavailable.
    """
    if relevant_ids is None or len(relevant_ids) == 0:
        return None
    cutoff = max(1, k)
    top_k = retrieved_ids[:cutoff]
    matched = set(top_k) & set(relevant_ids)
    return float(len(matched) / cutoff)


def compute_page_coverage(
    retrieved_pages: Sequence[int],
    expected_pages: Optional[Sequence[int]],
) -> Optional[float]:
    """Compute the fraction of expected reference pages retrieved.

    Args:
        retrieved_pages: Sequence of page numbers found in retrieved contexts.
        expected_pages: Expected ground-truth page numbers.

    Returns:
        Optional[float]: Coverage ratio [0.0, 1.0], or None if expected_pages is absent.
    """
    if expected_pages is None or len(expected_pages) == 0:
        return None
    exp_set = set(expected_pages)
    ret_set = set(retrieved_pages)
    matched = exp_set & ret_set
    return float(len(matched) / len(exp_set))


def compute_source_match(
    retrieved_sources: Sequence[str],
    expected_source: Optional[Union[str, Sequence[str]]],
) -> Optional[float]:
    """Compute whether the expected document source was retrieved.

    Args:
        retrieved_sources: Sequence of document IDs from retrieved contexts.
        expected_source: Expected document ID string or list of document IDs.

    Returns:
        Optional[float]: 1.0 if match, 0.0 if not, or None if expected_source is absent.
    """
    if expected_source is None:
        return None

    if isinstance(expected_source, str):
        expected_set = {expected_source} if expected_source.strip() else set()
    else:
        expected_set = {str(s).strip() for s in expected_source if str(s).strip()}

    if not expected_set:
        return None

    ret_set = {str(s).strip() for s in retrieved_sources if str(s).strip()}
    return 1.0 if (expected_set & ret_set) else 0.0


def compute_context_coverage(
    prediction: str,
    contexts: Sequence[Any],
) -> float:
    """Compute content-word coverage of the generated answer by retrieved context.

    Calculates the proportion of non-stopword tokens in the prediction that appear
    within the retrieved context text.

    Args:
        prediction: Generated response text.
        contexts: Sequence of RetrievedContext objects or context strings.

    Returns:
        float: Coverage score between 0.0 and 1.0.
    """
    pred_words = [w for w in _tokenize(prediction) if w not in _STOPWORDS]
    if not pred_words:
        return 1.0 if check_missing_answer(prediction) else 0.0

    context_texts: List[str] = []
    for c in contexts:
        if hasattr(c, "text"):
            context_texts.append(c.text)
        else:
            context_texts.append(str(c))
    all_context = " ".join(context_texts)
    context_words = set(_tokenize(all_context))

    matched_words = [w for w in pred_words if w in context_words]
    return float(len(matched_words) / len(pred_words))


def check_missing_answer(prediction: str) -> bool:
    """Determine whether an answer indicates missing information or safe abstention.

    Args:
        prediction: Generated answer string.

    Returns:
        bool: True if answer indicates insufficient evidence or is blank.
    """
    cleaned = prediction.strip().lower() if prediction else ""
    if not cleaned:
        return True

    indicators = [
        INSUFFICIENT_INFORMATION_RESPONSE.lower(),
        "insufficient information",
        "cannot answer",
        "not enough information",
        "context does not contain",
        "no relevant information",
        "i do not have enough information",
        "information is insufficient",
        "cannot be answered based on the provided",
        "does not provide sufficient",
    ]
    return any(ind in cleaned for ind in indicators)


def detect_unsupported_claims_deterministic(
    prediction: str,
    contexts: Sequence[Any],
    min_overlap_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Detect potentially unsupported claims in the prediction using sentence lexical overlap.

    Splits the prediction into candidate sentences and checks whether each sentence
    contains sufficient non-stopword overlap with the retrieved passages.

    Args:
        prediction: Generated text.
        contexts: Retrieved contexts or passages.
        min_overlap_ratio: Minimum ratio of sentence content words that must appear in context.

    Returns:
        Dict[str, Any]: Detailed breakdown including unsupported sentences and faithfulness ratio.
    """
    if check_missing_answer(prediction):
        return {
            "total_sentences": 0,
            "supported_sentences": 0,
            "unsupported_sentences": 0,
            "unsupported_claims": [],
            "faithfulness_ratio": 1.0,
            "abstention": True,
        }

    # Split into sentences (by period, exclamation, question mark, or newline)
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", prediction.strip())
    sentences = [s.strip() for s in raw_sentences if s and len(s.strip()) > 5]

    if not sentences:
        return {
            "total_sentences": 0,
            "supported_sentences": 0,
            "unsupported_sentences": 0,
            "unsupported_claims": [],
            "faithfulness_ratio": 1.0,
            "abstention": False,
        }

    context_texts = [getattr(c, "text", str(c)) for c in contexts]
    all_context_words = set(_tokenize(" ".join(context_texts)))

    supported_count = 0
    unsupported_claims: List[str] = []

    for s in sentences:
        s_words = [w for w in _tokenize(s) if w not in _STOPWORDS]
        if not s_words:
            supported_count += 1
            continue
        overlap = [w for w in s_words if w in all_context_words]
        ratio = len(overlap) / len(s_words)
        if ratio >= min_overlap_ratio:
            supported_count += 1
        else:
            unsupported_claims.append(s)

    faithfulness = supported_count / len(sentences)

    return {
        "total_sentences": len(sentences),
        "supported_sentences": supported_count,
        "unsupported_sentences": len(unsupported_claims),
        "unsupported_claims": unsupported_claims,
        "faithfulness_ratio": float(faithfulness),
        "abstention": False,
    }


def llm_judge_faithfulness(
    prediction: str,
    contexts: Sequence[Any],
    llm_client: Optional[BaseLLMClient] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate factual consistency using an optional separate LLM judge.

    Args:
        prediction: Generated answer string.
        contexts: Sequence of retrieved contexts.
        llm_client: Configured separate LLM judge client. If None, returns None (unavailable).

    Returns:
        Optional[Dict[str, Any]]: Judge response or None if LLM judge is not configured.
    """
    if llm_client is None:
        return None

    context_text = "\n\n".join([getattr(c, "text", str(c)) for c in contexts])
    user_prompt = VERIFICATION_USER_PROMPT_TEMPLATE.format(
        context=context_text,
        candidate_response=prediction,
    )

    try:
        response = llm_client.generate(prompt=user_prompt, system_prompt=VERIFICATION_SYSTEM_PROMPT)
        content = response.content.strip()
        # Parse JSON from model output
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        return {"raw_evaluation": content, "is_supported": None}
    except Exception as exc:
        logger.warning("LLM judge evaluation failed: %s", exc)
        return None


def compute_hallucination_rate(
    total_claims: int,
    unsupported_claims: int,
) -> float:
    """Compute empirical hallucination rate.

    Args:
        total_claims: Total extracted atomic claims.
        unsupported_claims: Number of claims lacking context grounding.

    Returns:
        float: Rate between 0.0 and 1.0.
    """
    if total_claims <= 0:
        return 0.0
    return float(np.clip(unsupported_claims / total_claims, 0.0, 1.0))


def compute_aggregate_metrics(per_question_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate summary metrics across all evaluated questions.

    Explicitly ignores unavailable metrics (`None`) without substituting zero or another metric.

    Args:
        per_question_results: Sequence of per-question metric evaluation dictionaries.

    Returns:
        Dict[str, Any]: Dataset-level aggregate statistics.
    """
    total = len(per_question_results)
    if total == 0:
        return {
            "total_samples": 0,
            "mean_semantic_similarity": None,
            "mean_token_f1": None,
            "mean_context_coverage": None,
            "mean_retrieval_recall": None,
            "mean_retrieval_precision": None,
            "mean_page_coverage": None,
            "mean_source_match": None,
            "mean_faithfulness_ratio": None,
            "abstention_rate": None,
        }

    def _mean_valid(key: str) -> Optional[float]:
        valid_vals = [r[key] for r in per_question_results if r.get(key) is not None]
        if not valid_vals:
            return None
        return float(np.mean(valid_vals))

    abstentions = sum(1 for r in per_question_results if r.get("is_missing_answer") is True)
    abstention_rate = float(abstentions / total)

    return {
        "total_samples": total,
        "mean_semantic_similarity": _mean_valid("semantic_similarity"),
        "mean_token_f1": _mean_valid("token_f1"),
        "mean_context_coverage": _mean_valid("context_coverage"),
        "mean_retrieval_recall": _mean_valid("retrieval_recall"),
        "mean_retrieval_precision": _mean_valid("retrieval_precision"),
        "mean_page_coverage": _mean_valid("page_coverage"),
        "mean_source_match": _mean_valid("source_match"),
        "mean_faithfulness_ratio": _mean_valid("faithfulness_ratio"),
        "abstention_rate": abstention_rate,
    }

