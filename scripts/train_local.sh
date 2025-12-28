#!/bin/bash
# Local Training Script (Mac/CPU)

# Example: Train on mini dataset with local_mini profile
# Usage: bash scripts/train_local.sh path/to/data.jsonl

DATA_PATH=${1:-"data/crse/scitech_crse_dataset_mini.jsonl"}

python3 training/train_crse.py \
  --profile local_mini \
  --data "$DATA_PATH" \
  --output-dir checkpoints/crse_local \
  --epochs 5 \
  --batch-groups 8

echo "✅ Training complete! Check checkpoints/crse_local/"

