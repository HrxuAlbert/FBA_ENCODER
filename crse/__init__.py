"""
CRSE (Certified Robust Semantic Encoder)

A contrastively trained text encoder designed to resist Byzantine semantic attacks
while preserving honest paraphrase proximity.
"""

from .model import CRSEModel, mean_pooling
from .config import TrainConfig, PROFILES

__version__ = "1.0.0"

__all__ = [
    "CRSEModel",
    "mean_pooling",
    "TrainConfig",
    "PROFILES",
]

