#!/usr/bin/bash
# GPU Training Script (CUDA)

# Example: Train on full dataset with gpu_full profile
# Usage: bash scripts/train_gpu.sh path/to/data.jsonl

DATA_PATH=${1:-"data/crse/scitech_crse_dataset_full.jsonl"}

python3 training/train_crse.py \
  --profile gpu_full \
  --data "$DATA_PATH" \
  --output-dir checkpoints/crse_gpu \
  --epochs 10 \
  --batch-groups 32

echo "✅ Training complete! Check checkpoints/crse_gpu/"

