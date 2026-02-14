"""RAG Q&A evaluation for retrieval and answer quality.

Evaluates the RAG pipeline along two axes:
1. Retrieval quality: whether the correct chunks are retrieved (hit rate, MRR)
2. Answer quality: how well the generated answer matches the reference (ROUGE-L, exact match)

Usage:
    python -m tests.evaluation.rag_eval --eval-set eval_qa.json --data-dir ~/.notetaker
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Load a Q&A evaluation set.

    Expected JSON format:
    [
        {
            "video_id": "abc123",
            "question": "What is discussed at the beginning?",
            "reference_answer": "The speaker introduces the topic of...",
            "expected_timestamps": ["00:00", "00:30"],
            "expected_keywords": ["introduction", "topic"]
        },
        ...
    ]

    Args:
        path: Path to the evaluation JSON file.

    Returns:
        List of evaluation items.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Evaluation set must be a JSON array.")

    return data


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    sources: list[dict],
    expected_timestamps: list[str],
    tolerance_seconds: float = 30.0,
) -> dict[str, float]:
    """Evaluate retrieval quality based on timestamp overlap.

    Checks if retrieved chunks cover the expected time regions.

    Args:
        sources: Retrieved source chunks with start_time/end_time.
        expected_timestamps: Expected timestamps in "MM:SS" format.
        tolerance_seconds: How close a retrieved chunk must be (seconds).

    Returns:
        Retrieval metrics: hit_rate, mrr.
    """
    if not expected_timestamps:
        return {"hit_rate": 1.0, "mrr": 1.0, "note": "No expected timestamps"}

    expected_seconds = [_parse_timestamp(ts) for ts in expected_timestamps]
    hits = 0
    reciprocal_ranks: list[float] = []

    for exp_sec in expected_seconds:
        found = False
        for rank, src in enumerate(sources, 1):
            src_start = src.get("start_time", 0)
            src_end = src.get("end_time", 0)

            # Check if expected timestamp falls within chunk +/- tolerance
            if (src_start - tolerance_seconds) <= exp_sec <= (src_end + tolerance_seconds):
                if not found:
                    reciprocal_ranks.append(1.0 / rank)
                    found = True
                    hits += 1
                break

        if not found:
            reciprocal_ranks.append(0.0)

    hit_rate = hits / len(expected_seconds) if expected_seconds else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "hits": hits,
        "total_expected": len(expected_seconds),
    }


def _parse_timestamp(ts: str) -> float:
    """Parse MM:SS timestamp to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


# ---------------------------------------------------------------------------
# Answer quality evaluation
# ---------------------------------------------------------------------------


def evaluate_answer(
    generated_answer: str,
    reference_answer: str,
    expected_keywords: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate answer quality using ROUGE-L and keyword coverage.

    Args:
        generated_answer: The RAG-generated answer.
        reference_answer: Human-written reference answer.
        expected_keywords: Optional keywords expected in the answer.

    Returns:
        Answer quality metrics.
    """
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference_answer, generated_answer)
    rouge_l = scores["rougeL"]

    result = {
        "rougeL_precision": round(rouge_l.precision, 4),
        "rougeL_recall": round(rouge_l.recall, 4),
        "rougeL_f1": round(rouge_l.fmeasure, 4),
    }

    # Keyword coverage
    if expected_keywords:
        gen_lower = generated_answer.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in gen_lower)
        result["keyword_coverage"] = round(found / len(expected_keywords), 4)
        result["keywords_found"] = found
        result["keywords_total"] = len(expected_keywords)
    else:
        result["keyword_coverage"] = 1.0

    # Exact match (normalized)
    gen_norm = " ".join(generated_answer.lower().split())
    ref_norm = " ".join(reference_answer.lower().split())
    result["exact_match"] = 1.0 if gen_norm == ref_norm else 0.0

    # Non-empty check
    result["non_empty"] = 1.0 if generated_answer.strip() else 0.0

    # "Not found" detection
    not_found_phrases = [
        "could not find",
        "not in the video",
        "no information",
        "i don't have",
    ]
    result["is_refusal"] = 1.0 if any(p in gen_lower for p in not_found_phrases) else 0.0

    return result


# ---------------------------------------------------------------------------
# Full pipeline evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    eval_set: list[dict[str, Any]],
    data_dir: str,
    persist_directory: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    ollama_model: str = "llama3.1:8b",
    ollama_base_url: str = "http://localhost:11434",
) -> list[dict[str, Any]]:
    """Run full RAG evaluation against an eval set.

    This requires Ollama and ChromaDB to be running with data loaded.

    Args:
        eval_set: List of evaluation items.
        data_dir: Base data directory.
        persist_directory: ChromaDB directory.
        embedding_model: Embedding model name.
        ollama_model: LLM model name.
        ollama_base_url: Ollama URL.

    Returns:
        List of per-question evaluation results.
    """
    from notetaker.pipeline.qa import answer_question

    if persist_directory is None:
        persist_directory = str(Path(data_dir) / "chroma")

    results: list[dict[str, Any]] = []

    for i, item in enumerate(eval_set):
        video_id = item["video_id"]
        question = item["question"]
        ref_answer = item.get("reference_answer", "")
        expected_ts = item.get("expected_timestamps", [])
        expected_kw = item.get("expected_keywords", [])

        print(f"  [{i + 1}/{len(eval_set)}] Q: {question[:60]}...")

        try:
            response = answer_question(
                query=question,
                video_id=video_id,
                persist_directory=persist_directory,
                embedding_model=embedding_model,
                ollama_model=ollama_model,
                ollama_base_url=ollama_base_url,
            )

            # Evaluate retrieval
            retrieval_metrics = evaluate_retrieval(response.sources, expected_ts)

            # Evaluate answer
            answer_metrics = evaluate_answer(response.answer, ref_answer, expected_kw)

            results.append(
                {
                    "question": question,
                    "video_id": video_id,
                    "generated_answer": response.answer,
                    "reference_answer": ref_answer,
                    "retrieval": retrieval_metrics,
                    "answer": answer_metrics,
                }
            )

        except Exception as e:
            results.append(
                {
                    "question": question,
                    "video_id": video_id,
                    "error": str(e),
                }
            )

    return results


def aggregate_rag_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate RAG evaluation results.

    Args:
        results: Per-question evaluation results.

    Returns:
        Aggregated metrics.
    """
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "No valid results"}

    # Retrieval aggregation
    hit_rates = [r["retrieval"]["hit_rate"] for r in valid]
    mrrs = [r["retrieval"]["mrr"] for r in valid]

    # Answer aggregation
    rouge_f1s = [r["answer"]["rougeL_f1"] for r in valid]
    kw_coverages = [r["answer"]["keyword_coverage"] for r in valid]
    refusals = [r["answer"]["is_refusal"] for r in valid]

    n = len(valid)
    return {
        "num_questions": len(results),
        "num_valid": n,
        "num_errors": len(results) - n,
        # Retrieval
        "mean_hit_rate": round(sum(hit_rates) / n, 4),
        "mean_mrr": round(sum(mrrs) / n, 4),
        # Answer
        "mean_rougeL_f1": round(sum(rouge_f1s) / n, 4),
        "mean_keyword_coverage": round(sum(kw_coverages) / n, 4),
        "refusal_rate": round(sum(refusals) / n, 4),
    }


def print_report(results: list[dict], aggregate: dict) -> None:
    """Print formatted RAG evaluation report."""
    print("=" * 70)
    print("RAG Q&A QUALITY REPORT")
    print("=" * 70)

    for i, r in enumerate(results):
        if "error" in r:
            print(f"\n  Q{i + 1}: {r['question'][:50]}... ERROR: {r['error']}")
        else:
            ret = r["retrieval"]
            ans = r["answer"]
            print(f"\n  Q{i + 1}: {r['question'][:50]}...")
            print(f"    Retrieval: hit_rate={ret['hit_rate']:.2%}  MRR={ret['mrr']:.2f}")
            print(
                f"    Answer:    ROUGE-L={ans['rougeL_f1']:.2%}  "
                f"KW={ans['keyword_coverage']:.2%}  "
                f"Refusal={'Yes' if ans['is_refusal'] else 'No'}"
            )

    print("\n" + "-" * 70)
    print("AGGREGATE:")
    if "error" in aggregate:
        print(f"  {aggregate['error']}")
    else:
        print(f"  Questions:       {aggregate['num_questions']} ({aggregate['num_valid']} valid)")
        print("  Retrieval:")
        print(f"    Mean Hit Rate: {aggregate['mean_hit_rate']:.2%}")
        print(f"    Mean MRR:      {aggregate['mean_mrr']:.4f}")
        print("  Answer Quality:")
        print(f"    Mean ROUGE-L:  {aggregate['mean_rougeL_f1']:.2%}")
        print(f"    Mean KW Cov:   {aggregate['mean_keyword_coverage']:.2%}")
        print(f"    Refusal Rate:  {aggregate['refusal_rate']:.2%}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG Q&A quality.")
    parser.add_argument("--eval-set", type=Path, required=True, help="Evaluation Q&A set JSON.")
    parser.add_argument(
        "--data-dir", type=str, default=str(Path.home() / ".notetaker"), help="Base data directory."
    )
    parser.add_argument("--chroma-dir", type=str, help="ChromaDB directory.")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model.")
    parser.add_argument("--output", type=Path, help="Save results as JSON.")

    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    print(f"Loaded {len(eval_set)} evaluation questions.")

    results = run_evaluation(
        eval_set=eval_set,
        data_dir=args.data_dir,
        persist_directory=args.chroma_dir,
        ollama_model=args.model,
    )

    aggregate = aggregate_rag_results(results)
    print_report(results, aggregate)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "results": results,
            "aggregate": aggregate,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
