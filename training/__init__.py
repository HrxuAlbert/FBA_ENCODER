"""
CRSE Training Module

Contains training scripts, data loaders, loss functions, and evaluation utilities.
"""

from .dataset import CRSEDataset, collate_fn_groups
from .loss import ContrastiveLoss
from .evaluation import evaluate_geometry

__all__ = [
    "CRSEDataset",
    "collate_fn_groups",
    "ContrastiveLoss",
    "evaluate_geometry",
]

