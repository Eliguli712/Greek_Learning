"""
Run the SemanticCompiler on a single example and on the dev gold set.

Usage:
    & python scripts/run_single.py
    & python scripts/run_single.py --format combined       # default: single JSON
    & python scripts/run_single.py --format staged         # 6 separate JSONs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import SemanticCompiler
from src.output_format import save_result, save_batch_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=["combined", "staged"],
        default="combined",
        help="Output format: 'combined' (single JSON) or 'staged' (6 separate JSONs per stage)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Output directory for saved results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compiler = SemanticCompiler(project_root=PROJECT_ROOT)

    print("=" * 70)
    print("Single-example run")
    print("=" * 70)
    text = "θεός γράφει λόγον"
    result = compiler.analyze(text)
    
    print(f"Output format: {args.format}")
    print()
    
    # Save single example
    if args.format == "combined":
        single_path = args.output_dir / "single_result.json"
        save_result(result, single_path, format_mode="combined")
        print(f"✓ Saved to: {single_path}")
    else:
        single_dir = args.output_dir / "single_result"
        save_result(result, single_dir, format_mode="staged", example_id="single")
        print(f"✓ Saved 7 files to: {single_dir}/")
        print("  - single_metadata.json")
        print("  - single_phoneme.json")
        print("  - single_morpheme.json")
        print("  - single_lexeme.json")
        print("  - single_semantic.json")
        print("  - single_relations.json")
        print("  - single_dag.json")

    dev_path = PROJECT_ROOT / "data/gold/dev_examples.jsonl"
    if not dev_path.exists() or dev_path.stat().st_size == 0:
        print("\nNo dev gold file found, skipping batch run.")
        return

    print("\n" + "=" * 70)
    print("Dev set batch run")
    print("=" * 70)
    items = []
    with dev_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    results = []
    for item in items:
        result = compiler.analyze(item["text"], item.get("language", "ancient_greek"))
        results.append((item.get("id"), result))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format == "combined":
        out_path = args.output_dir / "dev_results.jsonl"
        save_batch_results(results, out_path, format_mode="combined")
        print(f"✓ Wrote {len(results)} results to {out_path}")
    else:
        out_dir = args.output_dir / "dev_results_staged"
        save_batch_results(results, out_dir, format_mode="staged")
        print(f"✓ Wrote {len(results)} examples to {out_dir}/")
        print(f"  Each example has 7 files (metadata + 6 stages)")



if __name__ == "__main__":
    main()