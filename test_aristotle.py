#!/usr/bin/env python
"""
Process Aristotle's ancient Greek text through the full semantic compiler pipeline.

Usage:
    & python test_aristotle.py
    & python test_aristotle.py --sentences 10
    & python test_aristotle.py --format staged --output-dir outputs/aristotle_test
    & python test_aristotle.py --sentences 5 --format combined --verbose
    & python test_aristotle.py --single-output --format combined
"""
import sys
import argparse
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import SemanticCompiler
from src.normalize import sentence_split
from src.output_format import save_batch_results, save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentences",
        type=int,
        default=5,
        help="Number of sentences from Aristotle.txt to process (default: 5)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all sentences in Aristotle.txt (overrides --sentences)",
    )
    parser.add_argument(
        "--single-output",
        action="store_true",
        help="Analyze entire text as one sample and save as a single unified output",
    )
    parser.add_argument(
        "--format",
        choices=["combined", "staged"],
        default="combined",
        help="Output format: 'combined' (single JSON) or 'staged' (6 separate JSONs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/aristotle_test"),
        help="Output directory for results (default: outputs/aristotle_test)",
    )
    parser.add_argument(
        "--language",
        default="ancient_greek",
        help="Language tag (default: ancient_greek)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output with full token details",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    aristotle = Path('Text/Aristotle.txt').read_text(encoding='utf-8')

    # Handle analysis granularity
    if args.single_output:
        units = [aristotle]
        unit_label = "full text"
    else:
        if args.all:
            units = sentence_split(aristotle)
        else:
            units = sentence_split(aristotle)[:args.sentences]
        unit_label = "sentences"
    
    compiler = SemanticCompiler()
    print('=' * 70)
    if args.single_output:
        print('ARISTOTLE SEMANTIC COMPILATION (Single Unified Output)')
    else:
        print(f'ARISTOTLE SEMANTIC COMPILATION (First {len(units)} {unit_label})')
    print(f'Format: {args.format} | Language: {args.language}')
    print('=' * 70)
    print()
    
    results = []
    
    for i, sent in enumerate(units, 1):
        item_name = 'Text' if args.single_output else f'Sentence {i}'
        print(f'[{item_name}]')
        if args.verbose:
            print(f'  Text: {sent}')
        else:
            print(f'  Text: {sent[:100]}...')
        print()
        
        try:
            result = compiler.analyze(sent, language=args.language)
            result_id = "full_text" if args.single_output else f"s{i}"
            results.append((result_id, result))
            
            # Extract clean data
            tokens = result.tokens
            lemmas = [tok.get('lemma', '?') for tok in result.lexeme_layer if 'error' not in tok]
            sem_types = [tok.get('semantic_type', '?') for tok in result.semantic_tokens if 'error' not in tok]
            relations = result.relations
            dag_valid = result.validation.get('ok', False)
            
            print(f'  Tokens ({len(tokens)}):        {", ".join(tokens)}')
            print(f'  Lemmas ({len(lemmas)}):        {", ".join(lemmas[:5])}{"..." if len(lemmas) > 5 else ""}')
            print(f'  Sem Types ({len(sem_types)}):  {", ".join(set(sem_types))}')
            rel_labels = [r['label'] for r in relations]
            rel_str = ', '.join(set(rel_labels)) if rel_labels else 'none'
            print(f'  Relations ({len(relations)}):    {rel_str}')
            print(f'  DAG Valid: {dag_valid}')
            
        except Exception as exc:
            print(f'  ERROR: {exc}')
        
        print()
    
    # Save results
    print('=' * 70)
    print('Saving results...')
    print('=' * 70)

    if args.single_output:
        single_result = results[0][1]
        if args.format == "combined":
            output_path = args.output_dir / "aristotle_full_text.json"
            save_result(single_result, output_path, format_mode="combined")
            print(f'✓ Wrote single unified result to {output_path}')
        else:
            output_dir = args.output_dir / "aristotle_full_text_staged"
            save_result(single_result, output_dir, format_mode="staged", example_id="output")
            print(f'✓ Wrote single unified staged result to {output_dir}/')
            print('  Includes 7 files (metadata + 6 stages)')
    else:
        if args.format == "combined":
            output_path = args.output_dir / "aristotle_results.jsonl"
            save_batch_results(results, output_path, format_mode="combined")
            print(f'✓ Wrote {len(results)} results to {output_path}')
        else:
            output_dir = args.output_dir / "aristotle_results_staged"
            save_batch_results(results, output_dir, format_mode="staged")
            print(f'✓ Wrote {len(results)} examples to {output_dir}/')
            print(f'  Each example has 7 files (metadata + 6 stages)')
    
    print()


if __name__ == '__main__':
    main()
