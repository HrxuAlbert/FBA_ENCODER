#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrastive Loss for CRSE Training

Implements anchor-centered InfoNCE loss to:
- Pull anchor and honest paraphrases together
- Push anchor and Byzantine attacks apart
"""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Anchor-centered InfoNCE loss
    
    For each anchor:
    - positives: honest paraphrases from the same group
    - negatives: Byzantine attack samples from the same group
    
    Loss formula:
      -log(exp(sim(anchor, positive)/τ) / (exp(sim(anchor, positive)/τ) + Σ exp(sim(anchor, negative)/τ)))
    
    where sim is cosine similarity and τ is temperature.
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Temperature parameter for softmax (smaller = sharper)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        embeddings: torch.Tensor,  # (N, D)
        labels: torch.Tensor,      # (N,) - 0=honest, 1=byzantine
        group_ids: torch.Tensor    # (N,) - group index
    ):
        """
        Compute anchor-centered InfoNCE loss
        
        Assumes the first sample in each group is the anchor,
        followed by positive(s), then negatives.
        """
        device = embeddings.device
        
        # Compute similarity matrix (normalized dot product = cosine similarity)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # (N, N)
        
        total_loss = 0.0
        n_anchors = 0
        
        # Process by group
        unique_groups = torch.unique(group_ids)
        
        for group_id in unique_groups:
            group_mask = (group_ids == group_id)
            group_indices = torch.where(group_mask)[0]
            group_labels = labels[group_indices]
            
            # Find anchor and positives (both are honest, i.e., label=0)
            honest_in_group = group_indices[group_labels == 0]
            
            if len(honest_in_group) < 2:
                continue  # Need at least anchor + 1 positive
            
            # First honest sample is the anchor
            anchor_idx = honest_in_group[0]
            
            # Other honest samples are positives (usually just 1)
            positive_indices = honest_in_group[1:]
            
            # Byzantines are negatives
            negative_indices = group_indices[group_labels == 1]
            
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            
            # Compute loss for each positive (usually just 1)
            for pos_idx in positive_indices:
                # sim(anchor, positive)
                sim_pos = sim_matrix[anchor_idx, pos_idx]
                
                # sim(anchor, negatives)
                sim_negs = sim_matrix[anchor_idx, negative_indices]
                
                # InfoNCE: -log(exp(sim_pos) / (exp(sim_pos) + Σ exp(sim_negs)))
                numerator = torch.exp(sim_pos)
                denominator = numerator + torch.exp(sim_negs).sum()
                
                loss_i = -torch.log(numerator / (denominator + 1e-8))
                total_loss += loss_i
                n_anchors += 1
        
        if n_anchors > 0:
            return total_loss / n_anchors
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

