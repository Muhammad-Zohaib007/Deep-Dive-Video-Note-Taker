#!/usr/bin/env python3
"""Full evaluation suite runner for Deep-Dive Video Note Taker.

Runs all evaluation metrics:
1. WER (Word Error Rate) for transcription quality
2. ROUGE for note generation quality
3. BERTScore for semantic similarity
4. RAG Q&A evaluation for retrieval + answer quality

Usage:
    python scripts/evaluate.py --eval-dir tests/fixtures/eval_data/
    python scripts/evaluate.py --eval-dir tests/fixtures/eval_data/ --skip-bert
    python scripts/evaluate.py --eval-dir tests/fixtures/eval_data/ --rag-only --eval-set qa.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_wer_eval(
    reference_dir: Path,
    hypothesis_dir: Path,
) -> dict[str, Any]:
    """Run WER evaluation on transcripts.

    Args:
        reference_dir: Directory with reference transcripts.
        hypothesis_dir: Directory with generated transcripts.

    Returns:
        WER evaluation results.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: Word Error Rate (WER) Evaluation")
    print("=" * 70)

    try:
        from tests.evaluation.wer_eval import aggregate_results, evaluate_directory

        results = evaluate_directory(reference_dir, hypothesis_dir)
        aggregate = aggregate_results(results)

        # Print summary
        if "error" not in aggregate:
            print(f"  Mean WER: {aggregate['mean_wer']:.2%}")
            print(f"  Files:    {aggregate['num_files']}")
        else:
            print(f"  {aggregate['error']}")

        return {"per_file": results, "aggregate": aggregate}

    except ImportError as e:
        print(f"  Skipped: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def run_rouge_eval(
    reference_dir: Path,
    hypothesis_dir: Path,
) -> dict[str, Any]:
    """Run ROUGE evaluation on generated notes.

    Args:
        reference_dir: Directory with reference notes.
        hypothesis_dir: Directory with generated notes.

    Returns:
        ROUGE evaluation results.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: ROUGE Evaluation")
    print("=" * 70)

    try:
        from tests.evaluation.rouge_eval import aggregate_results, evaluate_directory

        results = evaluate_directory(reference_dir, hypothesis_dir)
        aggregate = aggregate_results(results)

        if "error" not in aggregate:
            for key in ["rouge1_mean_f", "rouge2_mean_f", "rougeL_mean_f"]:
                if key in aggregate:
                    print(f"  {key}: {aggregate[key]:.2%}")
        else:
            print(f"  {aggregate['error']}")

        return {"per_file": results, "aggregate": aggregate}

    except ImportError as e:
        print(f"  Skipped: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def run_bert_score_eval(
    reference_dir: Path,
    hypothesis_dir: Path,
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, Any]:
    """Run BERTScore evaluation on generated notes.

    Args:
        reference_dir: Directory with reference notes.
        hypothesis_dir: Directory with generated notes.
        model_type: BERTScore model.

    Returns:
        BERTScore evaluation results.
    """
    print("\n" + "=" * 70)
    print("STAGE 3: BERTScore Evaluation")
    print("=" * 70)

    try:
        from tests.evaluation.bert_score_eval import (
            aggregate_results,
            evaluate_directory,
        )

        results = evaluate_directory(reference_dir, hypothesis_dir, model_type=model_type)
        aggregate = aggregate_results(results)

        if "error" not in aggregate:
            print(f"  Mean F1:        {aggregate['mean_f1']:.4f}")
            print(f"  Mean Precision: {aggregate['mean_precision']:.4f}")
            print(f"  Mean Recall:    {aggregate['mean_recall']:.4f}")
        else:
            print(f"  {aggregate['error']}")

        return {"per_file": results, "aggregate": aggregate}

    except ImportError as e:
        print(f"  Skipped: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def run_rag_eval(
    eval_set_path: Path,
    data_dir: str,
    ollama_model: str = "llama3.1:8b",
) -> dict[str, Any]:
    """Run RAG Q&A evaluation.

    Args:
        eval_set_path: Path to evaluation Q&A set JSON.
        data_dir: Base data directory.
        ollama_model: Ollama model name.

    Returns:
        RAG evaluation results.
    """
    print("\n" + "=" * 70)
    print("STAGE 4: RAG Q&A Evaluation")
    print("=" * 70)

    try:
        from tests.evaluation.rag_eval import (
            aggregate_rag_results,
            load_eval_set,
            run_evaluation,
        )

        eval_set = load_eval_set(eval_set_path)
        print(f"  Loaded {len(eval_set)} questions.")

        results = run_evaluation(
            eval_set=eval_set,
            data_dir=data_dir,
            ollama_model=ollama_model,
        )
        aggregate = aggregate_rag_results(results)

        if "error" not in aggregate:
            print(f"  Mean Hit Rate:  {aggregate['mean_hit_rate']:.2%}")
            print(f"  Mean ROUGE-L:   {aggregate['mean_rougeL_f1']:.2%}")
            print(f"  Refusal Rate:   {aggregate['refusal_rate']:.2%}")
        else:
            print(f"  {aggregate['error']}")

        return {"results": results, "aggregate": aggregate}

    except FileNotFoundError:
        print(f"  Skipped: Evaluation set not found at {eval_set_path}")
        return {"error": f"File not found: {eval_set_path}"}
    except ImportError as e:
        print(f"  Skipped: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full evaluation suite for Deep-Dive Video Note Taker."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation data (transcripts/, notes/, qa.json).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / ".notetaker"),
        help="Base data directory for RAG evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save full results as JSON.",
    )
    parser.add_argument(
        "--skip-wer",
        action="store_true",
        help="Skip WER evaluation.",
    )
    parser.add_argument(
        "--skip-rouge",
        action="store_true",
        help="Skip ROUGE evaluation.",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip BERTScore evaluation (slow without GPU).",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG evaluation (requires Ollama).",
    )
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="Only run RAG evaluation.",
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        default="qa.json",
        help="Name of Q&A evaluation set file within eval-dir.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.1:8b",
        help="Ollama model for RAG evaluation.",
    )
    parser.add_argument(
        "--bert-model",
        default="microsoft/deberta-xlarge-mnli",
        help="BERTScore model type.",
    )

    args = parser.parse_args()

    eval_dir = args.eval_dir
    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        sys.exit(1)

    start_time = time.time()
    all_results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "eval_dir": str(eval_dir),
    }

    print("=" * 70)
    print("DEEP-DIVE VIDEO NOTE TAKER — EVALUATION SUITE")
    print(f"Evaluation data: {eval_dir}")
    print("=" * 70)

    # Expected directory structure:
    #   eval_dir/
    #     transcripts/
    #       reference/   — ground truth transcripts
    #       hypothesis/  — machine transcripts
    #     notes/
    #       reference/   — ground truth notes
    #       hypothesis/  — machine notes
    #     qa.json        — Q&A evaluation set

    # Stage 1: WER
    if not args.skip_wer and not args.rag_only:
        ref_transcripts = eval_dir / "transcripts" / "reference"
        hyp_transcripts = eval_dir / "transcripts" / "hypothesis"
        if ref_transcripts.exists() and hyp_transcripts.exists():
            all_results["wer"] = run_wer_eval(ref_transcripts, hyp_transcripts)
        else:
            print("\n  WER: Skipped (no transcripts/ directory)")

    # Stage 2: ROUGE
    if not args.skip_rouge and not args.rag_only:
        ref_notes = eval_dir / "notes" / "reference"
        hyp_notes = eval_dir / "notes" / "hypothesis"
        if ref_notes.exists() and hyp_notes.exists():
            all_results["rouge"] = run_rouge_eval(ref_notes, hyp_notes)
        else:
            print("\n  ROUGE: Skipped (no notes/ directory)")

    # Stage 3: BERTScore
    if not args.skip_bert and not args.rag_only:
        ref_notes = eval_dir / "notes" / "reference"
        hyp_notes = eval_dir / "notes" / "hypothesis"
        if ref_notes.exists() and hyp_notes.exists():
            all_results["bert_score"] = run_bert_score_eval(
                ref_notes, hyp_notes, model_type=args.bert_model
            )
        else:
            print("\n  BERTScore: Skipped (no notes/ directory)")

    # Stage 4: RAG
    if not args.skip_rag:
        qa_path = eval_dir / args.eval_set
        if qa_path.exists():
            all_results["rag"] = run_rag_eval(
                qa_path, args.data_dir, ollama_model=args.ollama_model
            )
        else:
            print(f"\n  RAG: Skipped (no {args.eval_set} found)")

    # Summary
    elapsed = time.time() - start_time
    all_results["elapsed_seconds"] = round(elapsed, 2)

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE ({elapsed:.1f}s)")
    print("=" * 70)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
