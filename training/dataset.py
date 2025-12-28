#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRSE Training Dataset

Organizes training data by anchor groups, where each group contains:
- 1 anchor text
- N positive samples (honest paraphrases)
- M negative samples (Byzantine attacks)
"""

import json
import random
from typing import List, Dict
import torch
from torch.utils.data import Dataset


class CRSEDataset(Dataset):
    """CRSE training dataset - organized by anchor groups"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to JSONL file, where each line is an anchor group
        """
        self.data_path = data_path
        self.groups = []
        
        print(f"Loading data: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                group = json.loads(line)
                self.groups.append(group)
        
        print(f"  → Loaded {len(self.groups)} anchor groups")
        
        # Statistics
        total_positives = sum(g['stats']['n_positives'] for g in self.groups)
        total_byzantine = sum(g['stats']['n_byzantine'] for g in self.groups)
        
        print(f"  → Average per group: {total_positives/len(self.groups):.1f} positives, "
              f"{total_byzantine/len(self.groups):.1f} byzantine")
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        return self.groups[idx]


def collate_fn_groups(batch: List[Dict], negatives_per_group: int = 2):
    """
    Custom collate function: samples anchor + positives + byzantines from each group
    
    Args:
        batch: List of anchor groups
        negatives_per_group: Number of byzantine samples to draw per group
    
    Returns:
        dict with keys:
            texts: List[str] - all text samples
            labels: torch.Tensor - labels (0=honest, 1=byzantine)
            group_ids: torch.Tensor - group index for each text
    """
    texts = []
    labels = []
    group_ids = []
    
    for group_idx, group in enumerate(batch):
        # 1. Anchor text
        anchor_text = group['anchor']['text']
        texts.append(anchor_text)
        labels.append(0)  # honest
        group_ids.append(group_idx)
        
        # 2. Random 1 positive
        if group['positives']:
            positive = random.choice(group['positives'])
            texts.append(positive['text'])
            labels.append(0)  # honest
            group_ids.append(group_idx)
        
        # 3. Random K byzantines
        n_byz = min(negatives_per_group, len(group['byzantine']))
        if n_byz > 0:
            byzantines = random.sample(group['byzantine'], n_byz)
            for byz in byzantines:
                texts.append(byz['text'])
                labels.append(1)  # byzantine
                group_ids.append(group_idx)
    
    return {
        'texts': texts,
        'labels': torch.tensor(labels, dtype=torch.long),
        'group_ids': torch.tensor(group_ids, dtype=torch.long)
    }

