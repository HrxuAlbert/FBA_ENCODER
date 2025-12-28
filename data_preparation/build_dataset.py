#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRSE Training Dataset Builder

Combines three types of data:
1. Anchor source texts (original paragraphs)
2. Faithful paraphrases (honest rewrites)
3. Byzantine attacks (B1-B4 attacks with mild/medium/strong budgets)

Output: Unified JSONL format where each line contains one anchor and its associated
positives (paraphrases) and negatives (byzantine attacks).

Usage:
  python3 build_dataset.py \\
    --anchors path/to/source.jsonl \\
    --paraphrases path/to/paraphrases.jsonl \\
    --attacks path/to/attacks.jsonl \\
    --output path/to/crse_dataset.jsonl \\
    --min-positives 1 \\
    --min-byzantine 1 \\
    --skip-trivial \\
    --verbose
"""

import json
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_anchors(anchors_path: Path, verbose: bool = False) -> Dict[str, Dict]:
    """
    Load anchor source texts
    
    Returns:
        {source_id: anchor_record}
    """
    anchors = {}
    
    with open(anchors_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                source_id = record.get('id')
                
                if not source_id:
                    if verbose:
                        print(f"Warning: Line {line_num} missing 'id' field, skipping")
                    continue
                
                # Standardize anchor structure
                anchor = {
                    'source_id': source_id,
                    'page_id': record.get('page_id'),
                    'title': record.get('title', 'Unknown'),
                    'topic': record.get('topic', 'science_tech'),
                    'tier': record.get('tier', 'unknown'),
                    'text': record.get('text', '')
                }
                
                if not anchor['text']:
                    if verbose:
                        print(f"Warning: Source {source_id} has no text, skipping")
                    continue
                
                anchors[source_id] = anchor
                
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Line {line_num} JSON parsing failed: {e}")
                continue
    
    return anchors


def load_paraphrases(paraphrases_path: Path, verbose: bool = False) -> Dict[str, List[Dict]]:
    """
    Load paraphrases
    
    Returns:
        {source_id: [paraphrase1, paraphrase2, ...]}
    """
    paraphrases_by_id = defaultdict(list)
    
    with open(paraphrases_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                source_id = record.get('source_id')
                
                if not source_id:
                    if verbose:
                        print(f"Warning: Paraphrase line {line_num} missing 'source_id', skipping")
                    continue
                
                # Extract paraphrases list
                paraphrases = record.get('paraphrases', [])
                
                for para in paraphrases:
                    para_obj = {
                        'id': para.get('id', f"{source_id}_P{len(paraphrases_by_id[source_id])+1}"),
                        'text': para.get('text', ''),
                        'source': 'paraphrase'
                    }
                    
                    if para_obj['text']:
                        paraphrases_by_id[source_id].append(para_obj)
                
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Paraphrase line {line_num} JSON parsing failed: {e}")
                continue
    
    return dict(paraphrases_by_id)


def load_attacks(attacks_path: Path, skip_trivial: bool = True, verbose: bool = False) -> Dict[str, List[Dict]]:
    """
    Load Byzantine attacks
    
    Returns:
        {source_id: [attack1, attack2, ...]}
    """
    attacks_by_id = defaultdict(list)
    
    skipped_trivial = 0
    
    with open(attacks_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                source_id = record.get('source_id')
                
                if not source_id:
                    if verbose:
                        print(f"Warning: Attack line {line_num} missing 'source_id', skipping")
                    continue
                
                # Check if trivial
                is_trivial = record.get('is_trivial', False)
                
                if skip_trivial and is_trivial:
                    skipped_trivial += 1
                    continue
                
                # Extract key fields
                attack_obj = {
                    'id': record.get('id', f"{source_id}_attack"),
                    'text': record.get('byzantine_text', ''),
                    'attack_type': record.get('attack_type', 'UNKNOWN'),
                    'budget': record.get('budget', 'unknown'),
                    'is_trivial': is_trivial
                }
                
                # Optional: add geometry information (if available)
                if 'geometry' in record:
                    attack_obj['cos_sim'] = record['geometry'].get('cos_sim')
                    attack_obj['angle_deg'] = record['geometry'].get('angle_deg')
                elif 'cos_sim' in record:  # Some may be at top level
                    attack_obj['cos_sim'] = record.get('cos_sim')
                    attack_obj['angle_deg'] = record.get('angle_deg')
                
                if attack_obj['text']:
                    attacks_by_id[source_id].append(attack_obj)
                
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Attack line {line_num} JSON parsing failed: {e}")
                continue
    
    if verbose and skip_trivial:
        print(f"Skipped {skipped_trivial} trivial attacks")
    
    return dict(attacks_by_id)


def build_dataset(
    anchors: Dict[str, Dict],
    paraphrases: Dict[str, List[Dict]],
    attacks: Dict[str, List[Dict]],
    min_positives: int = 1,
    min_byzantine: int = 1,
    verbose: bool = False
) -> tuple[List[Dict], Dict]:
    """
    Build CRSE dataset
    
    Returns:
        (dataset_records, statistics)
    """
    dataset = []
    
    stats = {
        'total_anchors': len(anchors),
        'anchors_with_paraphrases': 0,
        'anchors_with_byzantine': 0,
        'anchors_with_both': 0,
        'skipped_insufficient_positives': 0,
        'skipped_insufficient_byzantine': 0,
        'final_anchors': 0
    }
    
    for source_id, anchor in anchors.items():
        # Get paraphrases and attacks for this anchor
        positives = paraphrases.get(source_id, [])
        byzantine = attacks.get(source_id, [])
        
        n_pos = len(positives)
        n_byz = len(byzantine)
        
        # Statistics
        if n_pos > 0:
            stats['anchors_with_paraphrases'] += 1
        if n_byz > 0:
            stats['anchors_with_byzantine'] += 1
        if n_pos > 0 and n_byz > 0:
            stats['anchors_with_both'] += 1
        
        # Filtering logic
        if n_pos < min_positives:
            stats['skipped_insufficient_positives'] += 1
            if verbose:
                print(f"Skipping {source_id}: positives={n_pos} < {min_positives}")
            continue
        
        if n_byz < min_byzantine:
            stats['skipped_insufficient_byzantine'] += 1
            if verbose:
                print(f"Skipping {source_id}: byzantine={n_byz} < {min_byzantine}")
            continue
        
        # Byzantine distribution statistics
        byz_per_attack_type = defaultdict(int)
        byz_per_budget = defaultdict(int)
        
        for byz in byzantine:
            byz_per_attack_type[byz['attack_type']] += 1
            byz_per_budget[byz['budget']] += 1
        
        # Build record
        record = {
            'anchor': anchor,
            'positives': positives,
            'byzantine': byzantine,
            'stats': {
                'n_positives': n_pos,
                'n_byzantine': n_byz,
                'byz_per_attack_type': dict(byz_per_attack_type),
                'byz_per_budget': dict(byz_per_budget)
            }
        }
        
        dataset.append(record)
        stats['final_anchors'] += 1
    
    return dataset, stats


def print_statistics(dataset: List[Dict], stats: Dict):
    """Print statistical report"""
    
    print("\n" + "=" * 80)
    print("CRSE Dataset Builder")
    print("=" * 80)
    
    print(f"\nAnchors total                : {stats['total_anchors']}")
    print(f"Anchors with paraphrases     : {stats['anchors_with_paraphrases']}")
    print(f"Anchors with byzantine       : {stats['anchors_with_byzantine']}")
    print(f"Anchors with both (raw)      : {stats['anchors_with_both']}")
    
    print(f"\nFiltering:")
    print(f"  Skipped (insufficient positives) : {stats['skipped_insufficient_positives']}")
    print(f"  Skipped (insufficient byzantine) : {stats['skipped_insufficient_byzantine']}")
    
    print(f"\n→ Final anchors in dataset   : {stats['final_anchors']}")
    
    if stats['final_anchors'] > 0:
        # Calculate averages
        total_positives = sum(r['stats']['n_positives'] for r in dataset)
        total_byzantine = sum(r['stats']['n_byzantine'] for r in dataset)
        
        avg_pos = total_positives / stats['final_anchors']
        avg_byz = total_byzantine / stats['final_anchors']
        
        print(f"\nAverage positives per anchor : {avg_pos:.2f}")
        print(f"Average byzantine per anchor : {avg_byz:.2f}")
        
        # Attack type distribution
        attack_type_counts = defaultdict(int)
        budget_counts = defaultdict(int)
        
        for record in dataset:
            for at, count in record['stats']['byz_per_attack_type'].items():
                attack_type_counts[at] += count
            for budget, count in record['stats']['byz_per_budget'].items():
                budget_counts[budget] += count
        
        print(f"\nByzantine distribution (attack_type):")
        for at in sorted(attack_type_counts.keys()):
            print(f"  {at:<30}: {attack_type_counts[at]}")
        
        print(f"\nByzantine distribution (budget):")
        for budget in ['mild', 'medium', 'strong']:
            if budget in budget_counts:
                print(f"  {budget:<30}: {budget_counts[budget]}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Build CRSE training dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--anchors', type=str, required=True,
                        help='Anchor source file path (JSONL)')
    parser.add_argument('--paraphrases', type=str, required=True,
                        help='Paraphrase file path (JSONL)')
    parser.add_argument('--attacks', type=str, required=True,
                        help='Byzantine attacks file path (JSONL)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output dataset path (JSONL)')
    parser.add_argument('--min-positives', type=int, default=1,
                        help='Minimum paraphrases required per anchor (default: 1)')
    parser.add_argument('--min-byzantine', type=int, default=1,
                        help='Minimum byzantine samples required per anchor (default: 1)')
    parser.add_argument('--skip-trivial', action='store_true', default=True,
                        help='Skip is_trivial=True attacks (default: True)')
    parser.add_argument('--max-anchors', type=int, default=None,
                        help='Keep at most N anchor groups (for building mini datasets)')
    parser.add_argument('--subset-seed', type=int, default=42,
                        help='Random seed for subset sampling (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information')
    
    args = parser.parse_args()
    
    # Parse paths
    script_dir = Path(__file__).parent
    anchors_path = Path(args.anchors)
    paraphrases_path = Path(args.paraphrases)
    attacks_path = Path(args.attacks)
    output_path = Path(args.output)
    
    # Check input files
    for path, name in [
        (anchors_path, 'Anchors'),
        (paraphrases_path, 'Paraphrases'),
        (attacks_path, 'Attacks')
    ]:
        if not path.exists():
            print(f"Error: {name} file does not exist: {path}")
            sys.exit(1)
    
    print(f"Loading data...")
    
    # Load three types of data
    print(f"  Loading anchors: {anchors_path}")
    anchors = load_anchors(anchors_path, args.verbose)
    print(f"    → {len(anchors)} anchors")
    
    print(f"  Loading paraphrases: {paraphrases_path}")
    paraphrases = load_paraphrases(paraphrases_path, args.verbose)
    total_paraphrases = sum(len(v) for v in paraphrases.values())
    print(f"    → {len(paraphrases)} anchors have paraphrases, total {total_paraphrases}")
    
    print(f"  Loading attacks: {attacks_path}")
    attacks = load_attacks(attacks_path, args.skip_trivial, args.verbose)
    total_attacks = sum(len(v) for v in attacks.values())
    print(f"    → {len(attacks)} anchors have attacks, total {total_attacks}")
    
    # Build dataset
    print(f"\nBuilding dataset...")
    print(f"  min_positives: {args.min_positives}")
    print(f"  min_byzantine: {args.min_byzantine}")
    print(f"  skip_trivial: {args.skip_trivial}")
    
    dataset, stats = build_dataset(
        anchors, paraphrases, attacks,
        args.min_positives, args.min_byzantine,
        args.verbose
    )
    
    # Subset sampling (if max_anchors specified)
    if args.max_anchors is not None and args.max_anchors < len(dataset):
        print(f"\nSubset sampling...")
        print(f"  Original anchor groups: {len(dataset)}")
        print(f"  Target max_anchors: {args.max_anchors}")
        print(f"  Random seed: {args.subset_seed}")
        
        random.seed(args.subset_seed)
        dataset = random.sample(dataset, args.max_anchors)
        
        print(f"  → Retained after sampling: {len(dataset)} anchor groups")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    print(f"\nWriting output: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Print statistics
    print_statistics(dataset, stats)
    
    print(f"\n✅ Complete! Output file: {output_path}")


if __name__ == '__main__':
    main()

