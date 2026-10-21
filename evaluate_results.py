
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from evaluate_factual import F1_CHECKLIST, checklist_score
from evaluate_structure import evaluate_json
from evaluate_variability import (
    baseline_length_change,
    concept_coverage,
    pairwise_tfidf_similarity,
    token_count,
    word_count,
)


DEFAULT_INPUT = Path(__file__).parent / "results" / "raw" / "revised_experiment_generations.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "evaluated" / "revised_evaluations.jsonl"
DEFAULT_COMPARISONS = Path(__file__).parent / "results" / "summaries" / "parameter_comparisons.csv"


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error
            if isinstance(record, dict):
                records.append(record)
    return records


def _factual_preliminary_labels(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Apply conservative lexical screening and preserve brief supporting evidence."""
    lowered = text.casefold()
    patterns = {
        "mae_definition": r"\bmae\b[\s\S]{0,500}(absolute|actual|prediction|error)",
        "rmse_definition": r"\brmse\b[\s\S]{0,500}(square root|squared)",
        "rmse_vs_mae": r"(rmse[\s\S]{0,500}(larger|greater|outlier)|larger[\s\S]{0,500}rmse)",
        "mape_definition": r"\bmape\b[\s\S]{0,500}(percentage|percent)",
        "mape_limitation": r"\bmape\b[\s\S]{0,500}(zero|near zero|small actual)",
    }
    labels = {}
    evidence = {}
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for item in F1_CHECKLIST:
        match = re.search(patterns[item.item_id], lowered)
        labels[item.item_id] = "correct" if match else "missing_or_unclear"
        if match:
            evidence[item.item_id] = next(
                (sentence.strip() for sentence in sentences if match.group(0)[:20].casefold() in sentence.casefold()),
                match.group(0)[:240],
            )
        else:
            evidence[item.item_id] = ""
    return labels, evidence


def evaluate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [
        record
        for record in records
        if record.get("status") == "success" and isinstance(record.get("raw_response"), str)
    ]
    output: list[dict[str, Any]] = []
    grouped_d: dict[tuple[str, str], dict[str, str]] = {}
    baselines: dict[tuple[str, str], str] = {}
    for record in successful:
        if record.get("prompt_id") == "D1":
            group = (record["model"], record["prompt_id"])
            grouped_d.setdefault(group, {})[record["configuration"]] = record["raw_response"]
            if record.get("configuration") == "baseline":
                baselines[group] = record["raw_response"]
    similarities = {
        group: pairwise_tfidf_similarity(values)
        for group, values in grouped_d.items()
    }
    for record in successful:
        prompt_id = record.get("prompt_id")
        response = record["raw_response"]
        evaluated: dict[str, Any] = {
            "generation_key": record.get("generation_key"),
            "prompt_id": prompt_id,
            "model": record.get("model"),
            "configuration": record.get("configuration"),
            "temperature": record.get("temperature"),
            "top_k": record.get("top_k"),
            "top_p": record.get("top_p"),
            "raw_response": response,
        }
        if prompt_id == "D1":
            group = (record["model"], prompt_id)
            evaluated["objective"] = "variability"
            evaluated["word_count"] = word_count(response)
            evaluated["token_count"] = token_count(response)
            evaluated["concept_coverage"] = concept_coverage(response)
            evaluated["baseline_length_change"] = baseline_length_change(
                baselines[group], response
            ) if group in baselines else None
            evaluated["pairwise_tfidf_similarity"] = similarities.get(group, {})
        elif prompt_id == "F1":
            labels, evidence = _factual_preliminary_labels(response)
            evaluated["objective"] = "factual"
            evaluated["preliminary_checklist_labels"] = labels
            evaluated["preliminary_score"] = checklist_score(labels)
            evaluated["evidence"] = evidence
            evaluated["final_reviewed_labels"] = {}
            evaluated["reviewer_notes"] = "Manual review required; preliminary lexical screening only."
        elif prompt_id == "S1":
            evaluated["objective"] = "structure"
            evaluated["structure_checks"] = evaluate_json(response)
        output.append(evaluated)
    return output


def parameter_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return baseline-relative rows without mixing prompts or models."""
    successful = [
        record
        for record in records
        if record.get("status") == "success" and isinstance(record.get("raw_response"), str)
    ]
    baselines = {
        (record["prompt_id"], record["model"]): record
        for record in successful
        if record.get("configuration") == "baseline"
    }
    rows: list[dict[str, Any]] = []
    for record in successful:
        if record.get("configuration") == "baseline":
            continue
        baseline = baselines.get((record["prompt_id"], record["model"]))
        if baseline is None:
            continue
        prompt_id = record["prompt_id"]
        if prompt_id == "D1":
            baseline_words = word_count(baseline["raw_response"])
            current_words = word_count(record["raw_response"])
            metric_pairs = [
                ("word_count", baseline_words, current_words),
                (
                    "concept_coverage_count",
                    sum(concept_coverage(baseline["raw_response"]).values()),
                    sum(concept_coverage(record["raw_response"]).values()),
                ),
                (
                    "baseline_similarity",
                    1.0,
                    pairwise_tfidf_similarity(
                        {
                            "baseline": baseline["raw_response"],
                            "current": record["raw_response"],
                        }
                    ).get("baseline|current"),
                ),
            ]
        elif prompt_id == "F1":
            baseline_score = checklist_score(
                _factual_preliminary_labels(baseline["raw_response"])[0]
            )["score"]
            current_score = checklist_score(
                _factual_preliminary_labels(record["raw_response"])[0]
            )["score"]
            metric_pairs = [("preliminary_checklist_score", baseline_score, current_score)]
        elif prompt_id == "S1":
            baseline_score = evaluate_json(baseline["raw_response"])["score"]
            current_score = evaluate_json(record["raw_response"])["score"]
            metric_pairs = [("strict_structure_score", baseline_score, current_score)]
        else:
            continue
        for metric, baseline_value, current_value in metric_pairs:
            if baseline_value is None or current_value is None:
                continue
            change = current_value - baseline_value
            rows.append(
                {
                    "objective": {"D1": "variability", "F1": "factual", "S1": "structure"}[
                        prompt_id
                    ],
                    "prompt_id": prompt_id,
                    "model": record["model"],
                    "configuration": record["configuration"],
                    "temperature": record["temperature"],
                    "top_k": record["top_k"],
                    "top_p": record["top_p"],
                    "metric": metric,
                    "baseline_metric_value": baseline_value,
                    "current_metric_value": current_value,
                    "absolute_change": change,
                    "percentage_change": change / baseline_value * 100
                    if baseline_value
                    else None,
                }
            )
    return rows


def write_evaluations(path: Path, evaluations: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for evaluation in evaluations:
            target.write(json.dumps(evaluation, ensure_ascii=False) + "\n")


def write_comparisons(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "objective", "prompt_id", "model", "configuration", "temperature",
        "top_k", "top_p", "metric", "baseline_metric_value",
        "current_metric_value", "absolute_change", "percentage_change",
    ]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--comparisons", type=Path, default=DEFAULT_COMPARISONS)
    args = parser.parse_args()
    records = read_records(args.input)
    evaluations = evaluate_records(records)
    write_evaluations(args.output, evaluations)
    comparisons = parameter_comparisons(records)
    write_comparisons(args.comparisons, comparisons)
    print(f"Wrote {len(evaluations)} evaluated records to {args.output}")
    print(f"Wrote {len(comparisons)} baseline comparisons to {args.comparisons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
