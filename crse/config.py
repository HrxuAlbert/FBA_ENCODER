#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRSE Training Configuration

Defines training hyperparameters and preset profiles for different training scenarios.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training configuration for CRSE"""
    # Data
    data_path: str
    output_dir: str = "checkpoints/crse"
    
    # Model
    model_name: str = "intfloat/e5-base-v2"
    max_length: int = 256
    freeze_layers: int = 6  # Number of initial layers to freeze
    projection_dim: int = 256  # Projection head output dimension
    
    # Training hyperparameters
    batch_groups: int = 8  # Number of anchor groups per batch
    negatives_per_group: int = 2  # Number of byzantine samples per group
    epochs: int = 5
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    temperature: float = 0.1  # Temperature for contrastive loss
    
    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    
    # Evaluation
    eval_every_steps: int = 50
    eval_n_samples: int = 50  # Number of samples for geometry evaluation
    
    # Random seed
    seed: int = 42


# Preset training profiles
PROFILES = {
    "local_mini": {
        "batch_groups": 8,
        "negatives_per_group": 2,
        "epochs": 5,
        "lr": 2e-5,
        "max_length": 256,
        "freeze_layers": 6,
        "projection_dim": 256,
        "device": "auto",  # Will automatically select MPS or CPU
        "eval_every_steps": 50,
        "eval_n_samples": 50,
    },
    "gpu_full": {
        "batch_groups": 32,
        "negatives_per_group": 3,
        "epochs": 10,
        "lr": 2e-5,
        "max_length": 256,
        "freeze_layers": 6,
        "projection_dim": 512,
        "device": "cuda",
        "eval_every_steps": 100,
        "eval_n_samples": 100,
    }
}

