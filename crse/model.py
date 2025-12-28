#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRSE Model Definition

A semantic encoder combining:
- E5-base-v2 as backbone
- Mean pooling over token embeddings
- Projection head for contrastive learning
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over token embeddings (recommended by E5)
    
    Args:
        last_hidden_state: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len)
    
    Returns:
        pooled: (batch_size, hidden_dim)
    """
    # Expand attention_mask for broadcasting
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum over valid tokens
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    
    # Count valid tokens (avoid division by zero)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    # Average pooling
    pooled = sum_embeddings / sum_mask
    
    return pooled


class CRSEModel(nn.Module):
    """
    CRSE Model: E5-base + Projection Head
    
    Architecture:
    - Backbone: E5-base-v2 (transformer encoder)
    - Pooling: Mean pooling over token embeddings
    - Projection: 2-layer MLP with Tanh activation
    """
    
    def __init__(
        self, 
        model_name: str = "intfloat/e5-base-v2",
        projection_dim: int = 256,
        freeze_layers: int = 6
    ):
        """
        Args:
            model_name: HuggingFace model name
            projection_dim: Output dimension of projection head
            freeze_layers: Number of initial transformer layers to freeze
        """
        super().__init__()
        
        # Load backbone and tokenizer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        
        # Freeze initial N layers
        if freeze_layers > 0:
            for name, param in self.encoder.named_parameters():
                if 'encoder.layer.' in name:
                    layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                    if layer_num < freeze_layers:
                        param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, projection_dim)
        )
    
    def encode(self, texts: List[str], normalize: bool = True, max_length: int = 256):
        """
        Encode text list (for evaluation, no gradient computation)
        
        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize embeddings
            max_length: Maximum sequence length
        
        Returns:
            embeddings: (batch_size, projection_dim)
        """
        self.eval()
        with torch.no_grad():
            return self._forward_impl(texts, normalize, max_length)
    
    def _forward_impl(self, texts: List[str], normalize: bool = True, max_length: int = 256):
        """
        Forward pass implementation (shared by training and evaluation)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to model device
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward through encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        
        # Projection
        z = self.projection(pooled)
        
        # Normalization
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        
        return z
    
    def forward(self, texts: List[str], max_length: int = 256):
        """
        Forward pass for training (requires gradient computation)
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
        
        Returns:
            embeddings: (batch_size, projection_dim)
        """
        return self._forward_impl(texts, normalize=True, max_length=max_length)

