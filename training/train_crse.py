#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRSE (Certified Robust Semantic Encoder) Training Script

Supports two training profiles:
1. local_mini: Local training on Mac/CPU (MPS, small batch, quick iteration)
2. gpu_full: Full training on GPU (large batch, full dataset)

Training strategy:
==================
Each training step:
1. Randomly sample G anchor groups (G = batch_groups)
2. For each group:
   - Take 1 anchor text
   - Randomly select 1 positive (paraphrase)
   - Randomly select K byzantines (K = negatives_per_group)
3. Compute contrastive loss:
   - HONEST pairs (anchor ↔ paraphrase): keep close
   - BYZANTINE pairs (anchor ↮ byzantine): push away
4. Update model parameters

Usage:
  # Local mini mode
  python3 train_crse.py --profile local_mini --data ../data/crse/scitech_crse_dataset_mini.jsonl
  
  # GPU full mode
  python3 train_crse.py --profile gpu_full --data ../data/crse/scitech_crse_dataset_full.jsonl
"""

import sys
import json
import random
import argparse
from pathlib import Path
from dataclasses import asdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Add parent directory to path to import crse and training modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from crse.model import CRSEModel
from crse.config import TrainConfig, PROFILES
from training.dataset import CRSEDataset, collate_fn_groups
from training.loss import ContrastiveLoss
from training.evaluation import evaluate_geometry


def train(config: TrainConfig):
    """Main training function"""
    
    # Set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Device selection
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    
    print(f"\n{'='*80}")
    print(f"CRSE Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Data: {config.data_path}")
    print(f"Configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Load dataset
    dataset = CRSEDataset(config.data_path)
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_groups,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_groups(batch, config.negatives_per_group)
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    model = CRSEModel(
        model_name=config.model_name,
        projection_dim=config.projection_dim,
        freeze_layers=config.freeze_layers
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = ContrastiveLoss(temperature=config.temperature)
    
    # Pre-training evaluation
    print(f"\nPre-training geometry evaluation...")
    geo_before = evaluate_geometry(model, dataset, config.eval_n_samples, config.max_length)
    print(f"  cos(anchor, paraphrase): mean={geo_before['cos_anchor_para_mean']:.4f}, "
          f"std={geo_before['cos_anchor_para_std']:.4f}")
    print(f"  cos(anchor, byzantine):  mean={geo_before['cos_anchor_byz_mean']:.4f}, "
          f"std={geo_before['cos_anchor_byz_std']:.4f}")
    
    # Training loop
    print(f"\nStarting training...")
    global_step = 0
    best_separation = 0.0  # cos_para - cos_byz
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            texts = batch['texts']
            labels = batch['labels'].to(device)
            group_ids = batch['group_ids'].to(device)
            
            # Forward
            embeddings = model(texts, max_length=config.max_length)
            
            # Compute loss
            loss = criterion(embeddings, labels, group_ids)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Periodic evaluation
            if global_step % config.eval_every_steps == 0:
                geo_curr = evaluate_geometry(model, dataset, config.eval_n_samples, config.max_length)
                separation = geo_curr['cos_anchor_para_mean'] - geo_curr['cos_anchor_byz_mean']
                
                print(f"\n  [Step {global_step}] Geometry evaluation:")
                print(f"    cos(anchor, para): {geo_curr['cos_anchor_para_mean']:.4f}")
                print(f"    cos(anchor, byz):  {geo_curr['cos_anchor_byz_mean']:.4f}")
                print(f"    separation: {separation:.4f}")
                
                # Save best model
                if separation > best_separation:
                    best_separation = separation
                    output_dir = Path(config.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), output_dir / "best_model.pt")
                    print(f"    → Saved best model (separation={separation:.4f})")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # Post-training evaluation
    print(f"\nPost-training geometry evaluation...")
    geo_after = evaluate_geometry(model, dataset, config.eval_n_samples, config.max_length)
    print(f"  cos(anchor, paraphrase): mean={geo_after['cos_anchor_para_mean']:.4f}, "
          f"std={geo_after['cos_anchor_para_std']:.4f}")
    print(f"  cos(anchor, byzantine):  mean={geo_after['cos_anchor_byz_mean']:.4f}, "
          f"std={geo_after['cos_anchor_byz_std']:.4f}")
    
    # Comparison
    print(f"\n{'='*80}")
    print(f"Before/After Comparison:")
    print(f"{'='*80}")
    print(f"cos(anchor, paraphrase):")
    print(f"  Before: {geo_before['cos_anchor_para_mean']:.4f} ± {geo_before['cos_anchor_para_std']:.4f}")
    print(f"  After:  {geo_after['cos_anchor_para_mean']:.4f} ± {geo_after['cos_anchor_para_std']:.4f}")
    print(f"  Change: {geo_after['cos_anchor_para_mean'] - geo_before['cos_anchor_para_mean']:+.4f}")
    
    print(f"\ncos(anchor, byzantine):")
    print(f"  Before: {geo_before['cos_anchor_byz_mean']:.4f} ± {geo_before['cos_anchor_byz_std']:.4f}")
    print(f"  After:  {geo_after['cos_anchor_byz_mean']:.4f} ± {geo_after['cos_anchor_byz_std']:.4f}")
    print(f"  Change: {geo_after['cos_anchor_byz_mean'] - geo_before['cos_anchor_byz_mean']:+.4f}")
    
    separation_before = geo_before['cos_anchor_para_mean'] - geo_before['cos_anchor_byz_mean']
    separation_after = geo_after['cos_anchor_para_mean'] - geo_after['cos_anchor_byz_mean']
    
    print(f"\nSeparation (cos_para - cos_byz):")
    print(f"  Before: {separation_before:.4f}")
    print(f"  After:  {separation_after:.4f}")
    print(f"  Improvement: {separation_after - separation_before:+.4f}")
    print(f"{'='*80}\n")
    
    # Save final model
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    
    print(f"✅ Training complete! Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='CRSE Training Script')
    
    parser.add_argument('--profile', type=str, default='local_mini',
                        choices=['local_mini', 'gpu_full'],
                        help='Training configuration profile')
    parser.add_argument('--data', type=str, required=True,
                        help='Training data path (JSONL)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Model output directory (default: checkpoints/crse_{profile})')
    
    # Allow overriding profile parameters
    parser.add_argument('--batch-groups', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load profile
    if args.profile not in PROFILES:
        print(f"Error: Unknown profile '{args.profile}'")
        print(f"Available profiles: {list(PROFILES.keys())}")
        return
    
    profile_config = PROFILES[args.profile].copy()
    
    # Override parameters
    if args.batch_groups is not None:
        profile_config['batch_groups'] = args.batch_groups
    if args.epochs is not None:
        profile_config['epochs'] = args.epochs
    if args.lr is not None:
        profile_config['lr'] = args.lr
    if args.device is not None:
        profile_config['device'] = args.device
    
    # Create configuration
    config = TrainConfig(
        data_path=args.data,
        output_dir=args.output_dir or f"checkpoints/crse_{args.profile}",
        **profile_config
    )
    
    # Start training
    train(config)


if __name__ == '__main__':
    main()

