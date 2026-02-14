"""BERTScore evaluation for semantic similarity of generated notes.

Uses BERTScore to measure how semantically similar generated notes are
to reference notes, complementing lexical metrics like ROUGE.

Usage:
    python -m tests.evaluation.bert_score_eval --reference ref.txt --hypothesis hyp.txt
    python -m tests.evaluation.bert_score_eval --reference-dir refs/ --hypothesis-dir hyps/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compute_bert_score(
    references: list[str],
    hypotheses: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    batch_size: int = 8,
) -> dict[str, list[float]]:
    """Compute BERTScore for reference-hypothesis pairs.

    Args:
        references: List of reference texts.
        hypotheses: List of hypothesis texts.
        model_type: BERTScore model. Default is recommended for English.
        lang: Language code.
        batch_size: Batch size for computation.

    Returns:
        Dictionary with precision, recall, f1 lists.
    """
    from bert_score import score as bert_score_fn

    prec, rec, f1 = bert_score_fn(
        hypotheses,
        references,
        model_type=model_type,
        lang=lang,
        verbose=False,
        batch_size=batch_size,
    )

    return {
        "precision": [round(p.item(), 4) for p in prec],
        "recall": [round(r.item(), 4) for r in rec],
        "f1": [round(f.item(), 4) for f in f1],
    }


def evaluate_pair(
    reference_path: Path,
    hypothesis_path: Path,
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, float]:
    """Evaluate BERTScore for a single file pair.

    Args:
        reference_path: Reference text file.
        hypothesis_path: Hypothesis text file.
        model_type: BERTScore model.

    Returns:
        BERTScore metrics (precision, recall, f1).
    """
    ref_text = _load_text(reference_path)
    hyp_text = _load_text(hypothesis_path)

    if not ref_text.strip():
        raise ValueError(f"Empty reference: {reference_path}")
    if not hyp_text.strip():
        raise ValueError(f"Empty hypothesis: {hypothesis_path}")

    scores = compute_bert_score([ref_text], [hyp_text], model_type=model_type)

    return {
        "precision": scores["precision"][0],
        "recall": scores["recall"][0],
        "f1": scores["f1"][0],
    }


def evaluate_directory(
    reference_dir: Path,
    hypothesis_dir: Path,
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, Any]:
    """Evaluate BERTScore for all matching file pairs.

    Batches computation for efficiency.

    Args:
        reference_dir: Directory of reference texts.
        hypothesis_dir: Directory of hypothesis texts.
        model_type: BERTScore model.

    Returns:
        Per-file BERTScore results.
    """
    ref_files = {f.stem: f for f in reference_dir.glob("*") if f.suffix in (".txt", ".json")}
    hyp_files = {f.stem: f for f in hypothesis_dir.glob("*") if f.suffix in (".txt", ".json")}
    matched = sorted(set(ref_files.keys()) & set(hyp_files.keys()))

    if not matched:
        print("Warning: No matching files found.")
        return {}

    refs: list[str] = []
    hyps: list[str] = []
    names: list[str] = []
    errors: dict[str, str] = {}

    for name in matched:
        try:
            ref_text = _load_text(ref_files[name])
            hyp_text = _load_text(hyp_files[name])
            if not ref_text.strip() or not hyp_text.strip():
                errors[name] = "Empty file"
                continue
            refs.append(ref_text)
            hyps.append(hyp_text)
            names.append(name)
        except Exception as e:
            errors[name] = str(e)

    results: dict[str, Any] = {}

    if names:
        scores = compute_bert_score(refs, hyps, model_type=model_type)
        for i, name in enumerate(names):
            results[name] = {
                "precision": scores["precision"][i],
                "recall": scores["recall"][i],
                "f1": scores["f1"][i],
            }

    for name, error in errors.items():
        results[name] = {"error": error}

    return results


def _load_text(path: Path) -> str:
    """Load text, handling JSON notes format."""
    content = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        data = json.loads(content)
        return _extract_notes_text(data)

    return content.strip()


def _extract_notes_text(data: dict) -> str:
    """Extract all text from structured notes JSON."""
    parts: list[str] = []

    if "structured_notes" in data:
        notes = data["structured_notes"]
        for key in ("title", "summary"):
            if key in notes:
                parts.append(notes[key])
        for section in notes.get("sections", []):
            if "heading" in section:
                parts.append(section["heading"])
            parts.extend(section.get("key_points", []))

    for ts in data.get("timestamps", []):
        if "label" in ts:
            parts.append(ts["label"])

    for item in data.get("action_items", []):
        if "action" in item:
            parts.append(item["action"])

    return " ".join(parts) if parts else json.dumps(data)


def aggregate_results(results: dict[str, Any]) -> dict[str, float]:
    """Compute aggregate BERTScore statistics.

    Args:
        results: Per-file BERTScore results.

    Returns:
        Aggregated metrics.
    """
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return {"error": "No valid results"}

    f1_scores = [v["f1"] for v in valid.values()]
    precisions = [v["precision"] for v in valid.values()]
    recalls = [v["recall"] for v in valid.values()]

    return {
        "mean_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "mean_precision": round(sum(precisions) / len(precisions), 4),
        "mean_recall": round(sum(recalls) / len(recalls), 4),
        "min_f1": round(min(f1_scores), 4),
        "max_f1": round(max(f1_scores), 4),
        "num_files": len(valid),
    }


def print_report(results: dict, aggregate: dict) -> None:
    """Print formatted BERTScore evaluation report."""
    print("=" * 70)
    print("NOTE QUALITY REPORT (BERTScore)")
    print("=" * 70)

    for name, metrics in sorted(results.items()):
        if "error" in metrics:
            print(f"\n  {name}: ERROR - {metrics['error']}")
        else:
            print(f"\n  {name}:")
            print(
                f"    P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}"
            )

    print("\n" + "-" * 70)
    print("AGGREGATE:")
    if "error" in aggregate:
        print(f"  {aggregate['error']}")
    else:
        print(f"  Mean F1:        {aggregate['mean_f1']:.4f}")
        print(f"  Mean Precision: {aggregate['mean_precision']:.4f}")
        print(f"  Mean Recall:    {aggregate['mean_recall']:.4f}")
        print(f"  F1 Range:       [{aggregate['min_f1']:.4f}, {aggregate['max_f1']:.4f}]")
        print(f"  Files:          {aggregate['num_files']}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate notes with BERTScore.")
    parser.add_argument("--reference", type=Path, help="Reference notes file.")
    parser.add_argument("--hypothesis", type=Path, help="Generated notes file.")
    parser.add_argument("--reference-dir", type=Path, help="Directory of references.")
    parser.add_argument("--hypothesis-dir", type=Path, help="Directory of hypotheses.")
    parser.add_argument(
        "--model",
        default="microsoft/deberta-xlarge-mnli",
        help="BERTScore model type.",
    )
    parser.add_argument("--output", type=Path, help="Save results as JSON.")

    args = parser.parse_args()

    if args.reference and args.hypothesis:
        metrics = evaluate_pair(args.reference, args.hypothesis, model_type=args.model)
        print(json.dumps(metrics, indent=2))
        results = {"single": metrics}
    elif args.reference_dir and args.hypothesis_dir:
        results = evaluate_directory(args.reference_dir, args.hypothesis_dir, model_type=args.model)
        aggregate = aggregate_results(results)
        print_report(results, aggregate)
        results["_aggregate"] = aggregate
    else:
        parser.error("Provide --reference/--hypothesis or --reference-dir/--hypothesis-dir")
        return

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
