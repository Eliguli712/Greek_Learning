"""
Output formatting utilities for semantic compilation results.

Supports two output modes:
  - combined: Single JSON with all 6 pipeline stages
  - staged: Separate JSON files for each stage (phoneme, morpheme, lexeme, semantic, relations, DAG)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

from src.pipeline import CompilerResult


def save_result(
    result: CompilerResult,
    output_path: Path,
    format_mode: str = "combined",
    example_id: str | None = None,
) -> None:
    """
    Save a CompilerResult in the specified format.

    Parameters
    ----------
    result : CompilerResult
        The compiled semantic analysis result
    output_path : Path
        Base output path (for combined mode, used as-is; for staged mode, used as directory)
    format_mode : str
        Either 'combined' (single JSON) or 'staged' (6 separate JSONs)
    example_id : str, optional
        Used in filenames for staged mode (e.g., "{example_id}_phoneme.json")
    """
    if format_mode == "combined":
        _save_combined(result, output_path)
    elif format_mode == "staged":
        _save_staged(result, output_path, example_id or "result")
    else:
        raise ValueError(f"Unknown output format: {format_mode}")


def save_batch_results(
    results: list[tuple[str | None, CompilerResult]],
    output_base: Path,
    format_mode: str = "combined",
) -> None:
    """
    Save multiple CompilerResults.

    Parameters
    ----------
    results : list[tuple[str | None, CompilerResult]]
        List of (example_id, result) pairs
    output_base : Path
        Base output path or directory
    format_mode : str
        Either 'combined' or 'staged'
    """
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if format_mode == "combined":
        _save_batch_combined(results, output_base)
    elif format_mode == "staged":
        _save_batch_staged(results, output_base)
    else:
        raise ValueError(f"Unknown output format: {format_mode}")


def _save_combined(result: CompilerResult, output_path: Path) -> None:
    """Save all stages in a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def _save_staged(result: CompilerResult, output_dir: Path, example_id: str) -> None:
    """Save each stage in a separate JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()

    stages_map = {
        "phoneme": data.get("phoneme_layer", []),
        "morpheme": data.get("morpheme_layer", []),
        "lexeme": data.get("lexeme_layer", []),
        "semantic": data.get("semantic_tokens", []),
        "relations": data.get("relations", []),
        "dag": data.get("dag", {}),
    }

    metadata = {
        "text": data.get("text", ""),
        "language": data.get("language", ""),
        "sentences": data.get("sentences", []),
        "tokens": data.get("tokens", []),
        "validation": data.get("validation", {}),
        "notes": data.get("notes", []),
    }

    # Save metadata
    metadata_path = output_dir / f"{example_id}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save each stage
    for stage_name, stage_data in stages_map.items():
        stage_path = output_dir / f"{example_id}_{stage_name}.json"
        with stage_path.open("w", encoding="utf-8") as f:
            json.dump(stage_data, f, ensure_ascii=False, indent=2)


def _save_batch_combined(results: list[tuple[str | None, CompilerResult]], output_path: Path) -> None:
    """Save batch results in JSONL format (combined mode)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example_id, result in results:
            payload = result.to_dict()
            if example_id:
                payload["id"] = example_id
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _save_batch_staged(results: list[tuple[str | None, CompilerResult]], output_dir: Path) -> None:
    """Save batch results with each example in a subdirectory (staged mode)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for example_id, result in results:
        ex_id = example_id or "result"
        ex_dir = output_dir / ex_id
        _save_staged(result, ex_dir, "output")
        summary.append(
            {
                "id": ex_id,
                "text": result.text[:100],
                "directory": str(ex_dir.relative_to(output_dir.parent)),
            }
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


__all__ = ["save_result", "save_batch_results"]
