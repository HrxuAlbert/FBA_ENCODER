#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry Evaluation for CRSE Training

Evaluates the geometric distribution of embeddings:
- cos(anchor, paraphrase) - should be high (honest proximity)
- cos(anchor, byzantine) - should be low (attack separation)
"""

import random
from typing import Dict
import torch
import numpy as np


def evaluate_geometry(
    model,
    dataset,
    n_samples: int = 50,
    max_length: int = 256
) -> Dict[str, float]:
    """
    Evaluate embedding geometry: anchor vs paraphrase/byzantine cosine similarity
    
    Args:
        model: CRSE model
        dataset: CRSE dataset
        n_samples: Number of groups to sample
        max_length: Maximum sequence length
    
    Returns:
        dict with keys:
            cos_anchor_para_mean: Mean cosine similarity between anchor and paraphrase
            cos_anchor_para_std: Std of anchor-paraphrase similarity
            cos_anchor_byz_mean: Mean cosine similarity between anchor and byzantine
            cos_anchor_byz_std: Std of anchor-byzantine similarity
    """
    model.eval()
    
    # Randomly select n_samples groups
    n_samples = min(n_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), n_samples)
    
    cos_anchor_para = []
    cos_anchor_byz = []
    
    with torch.no_grad():
        for idx in sample_indices:
            group = dataset[idx]
            
            anchor_text = group['anchor']['text']
            
            # Anchor vs Paraphrase
            if group['positives']:
                para_text = random.choice(group['positives'])['text']
                
                anchor_emb = model.encode([anchor_text], max_length=max_length)[0]
                para_emb = model.encode([para_text], max_length=max_length)[0]
                
                cos_sim = torch.dot(anchor_emb, para_emb).item()
                cos_anchor_para.append(cos_sim)
            
            # Anchor vs Byzantine
            if group['byzantine']:
                byz_text = random.choice(group['byzantine'])['text']
                
                anchor_emb = model.encode([anchor_text], max_length=max_length)[0]
                byz_emb = model.encode([byz_text], max_length=max_length)[0]
                
                cos_sim = torch.dot(anchor_emb, byz_emb).item()
                cos_anchor_byz.append(cos_sim)
    
    results = {
        'cos_anchor_para_mean': float(np.mean(cos_anchor_para)) if cos_anchor_para else 0.0,
        'cos_anchor_para_std': float(np.std(cos_anchor_para)) if cos_anchor_para else 0.0,
        'cos_anchor_byz_mean': float(np.mean(cos_anchor_byz)) if cos_anchor_byz else 0.0,
        'cos_anchor_byz_std': float(np.std(cos_anchor_byz)) if cos_anchor_byz else 0.0,
    }
    
    model.train()
    return results

