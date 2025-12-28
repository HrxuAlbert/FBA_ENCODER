# CRSE Training Guide

Comprehensive guide to training CRSE models for Byzantine-robust semantic encoding.

---

## Table of Contents

1. [Training Overview](#training-overview)
2. [Dataset Requirements](#dataset-requirements)
3. [Training Profiles](#training-profiles)
4. [Hyperparameters](#hyperparameters)
5. [Monitoring Training](#monitoring-training)
6. [Advanced Tips](#advanced-tips)

---

## Training Overview

CRSE training uses **anchor-centered contrastive learning** with InfoNCE loss:

```
For each anchor group:
  anchor = original text
  positives = faithful paraphrases (honest)
  negatives = Byzantine attacks (adversarial)

Loss: Pull positives close, push negatives away
```

### Training Loop

```
For each epoch:
  For each batch (G anchor groups):
    1. Sample 1 anchor + 1 positive + K negatives per group
    2. Encode all texts → embeddings
    3. Compute InfoNCE loss
    4. Backprop + optimizer step
    5. Every N steps: evaluate geometry (cos similarities)
  
  Save best model (highest separation)
```

---

## Dataset Requirements

### Format

JSONL file, one anchor group per line:

```json
{
  "anchor": {
    "text": "Original paragraph...",
    "source_id": "id123",
    "title": "Article title",
    "topic": "science_tech"
  },
  "positives": [
    {"text": "Faithful paraphrase 1...", "id": "id123_P1"},
    {"text": "Faithful paraphrase 2...", "id": "id123_P2"}
  ],
  "byzantine": [
    {
      "text": "Attack text...",
      "id": "id123_B1_mild",
      "attack_type": "B1_POLARITY_FLIP",
      "budget": "mild"
    },
    ...
  ],
  "stats": {
    "n_positives": 2,
    "n_byzantine": 12
  }
}
```

### Size Recommendations

| Dataset Size | Anchor Groups | Use Case                        |
|--------------|--------------|---------------------------------|
| **Mini**     | 10-50        | Quick testing, debugging        |
| **Small**    | 100-300      | Local training, proof-of-concept|
| **Full**     | 500-1000+    | Production model, paper results |

### Quality Criteria

- **Positives**: Must preserve factual content and causal structure
- **Byzantines**: Should be subtle (not topic-divergent), cover multiple attack types
- **Diversity**: Include different topics, writing styles, complexity levels

---

## Training Profiles

### Profile: `local_mini`

**For:** Quick local testing on Mac/CPU

```bash
python3 training/train_crse.py \
  --profile local_mini \
  --data data/crse/mini_dataset.jsonl
```

**Config:**
- Device: MPS (Apple Silicon) or CPU
- Batch groups: 8
- Epochs: 5
- Projection dim: 256
- Training time: 10-30 minutes (depends on dataset size)

---

### Profile: `gpu_full`

**For:** Full training on GPU (local CUDA machine or cloud)

```bash
python3 training/train_crse.py \
  --profile gpu_full \
  --data data/crse/full_dataset.jsonl
```

**Config:**
- Device: CUDA
- Batch groups: 32
- Epochs: 10
- Projection dim: 512
- Training time: 1-3 hours (depends on dataset size and GPU)

---

## Hyperparameters

### Key Parameters

| Parameter           | Description                          | Default      | Recommended Range |
|---------------------|--------------------------------------|--------------|-------------------|
| `--batch-groups`    | Anchor groups per batch              | 8 (mini)     | 4-64              |
| `--negatives-per-group` | Byzantine samples per group      | 2            | 1-4               |
| `--epochs`          | Training epochs                      | 5 (mini)     | 3-15              |
| `--lr`              | Learning rate                        | 2e-5         | 1e-5 to 5e-5      |
| `--temperature`     | InfoNCE temperature                  | 0.1          | 0.05-0.2          |
| `--projection-dim`  | Projection head output dim           | 256 (mini)   | 128-512           |
| `--freeze-layers`   | Number of frozen encoder layers      | 6            | 0-12              |

### Override Examples

```bash
# Increase batch size for faster training (requires more VRAM)
python3 training/train_crse.py \
  --profile gpu_full \
  --data data/crse/full_dataset.jsonl \
  --batch-groups 64

# Lower learning rate for fine-tuning
python3 training/train_crse.py \
  --profile gpu_full \
  --data data/crse/full_dataset.jsonl \
  --lr 1e-5

# Larger projection dimension for higher capacity
python3 training/train_crse.py \
  --profile gpu_full \
  --data data/crse/full_dataset.jsonl \
  --projection-dim 1024
```

---

## Monitoring Training

### Metrics

CRSE tracks the following metrics:

1. **Loss**: InfoNCE contrastive loss (lower is better)
2. **cos(anchor, para)**: Mean cosine similarity between anchors and paraphrases (higher is better)
3. **cos(anchor, byz)**: Mean cosine similarity between anchors and Byzantine attacks (lower is better)
4. **Separation**: `cos(anchor, para) - cos(anchor, byz)` (higher is better)

### Expected Behavior

**Pre-training** (untrained E5-base):
- cos(anchor, para): ~0.75-0.85
- cos(anchor, byz): ~0.70-0.80
- Separation: ~0.05-0.10 (baseline encoder struggles to separate)

**Post-training** (trained CRSE):
- cos(anchor, para): ~0.85-0.95 (should increase or stay high)
- cos(anchor, byz): ~0.40-0.60 (should decrease significantly)
- Separation: ~0.30-0.50 (large increase)

### Training Output Example

```
================================================================================
CRSE Training
================================================================================
Device: cuda
Data: data/crse/full_dataset.jsonl
...

Pre-training geometry evaluation...
  cos(anchor, paraphrase): mean=0.7823, std=0.0412
  cos(anchor, byzantine):  mean=0.7645, std=0.0381

Starting training...
Epoch 1/10:   100%|██████████| 32/32 [01:23<00:00, loss: 0.5234]

  [Step 50] Geometry evaluation:
    cos(anchor, para): 0.8134
    cos(anchor, byz):  0.6789
    separation: 0.1345
    → Saved best model (separation=0.1345)

Epoch 2/10:   100%|██████████| 32/32 [01:21<00:00, loss: 0.3456]
...

Post-training geometry evaluation...
  cos(anchor, paraphrase): mean=0.8956, std=0.0234
  cos(anchor, byzantine):  mean=0.4523, std=0.0567

================================================================================
Before/After Comparison:
================================================================================
cos(anchor, paraphrase):
  Before: 0.7823 ± 0.0412
  After:  0.8956 ± 0.0234
  Change: +0.1133

cos(anchor, byzantine):
  Before: 0.7645 ± 0.0381
  After:  0.4523 ± 0.0567
  Change: -0.3122

Separation (cos_para - cos_byz):
  Before: 0.0178
  After:  0.4433
  Improvement: +0.4255
================================================================================

✅ Training complete! Model saved to: checkpoints/crse_full
```

---

## Advanced Tips

### 1. Hard Negative Mining

For better separation, sample byzantines with higher initial cosine similarity:

```python
# In collate_fn_groups, sort byzantines by cos_sim and take top K
byzantines_sorted = sorted(group['byzantine'], key=lambda x: x.get('cos_sim', 0), reverse=True)
byzantines = byzantines_sorted[:negatives_per_group]
```

### 2. Multi-Positive Contrastive

Use all paraphrases as positives (not just 1):

```python
# In collate_fn_groups
for positive in group['positives']:
    texts.append(positive['text'])
    labels.append(0)
    group_ids.append(group_idx)
```

### 3. Balanced Attack Sampling

Ensure each batch covers all attack types:

```python
# Sample 1 attack from each type (B1, B2, B3, B4)
attack_types = ['B1_POLARITY_FLIP', 'B2_EVIDENCE_OMISSION', 'B3_FAKE_CAUSALITY', 'B4_ON_TOPIC_HALLUCINATION']
for at in attack_types:
    candidates = [b for b in group['byzantine'] if b['attack_type'] == at]
    if candidates:
        texts.append(random.choice(candidates)['text'])
        labels.append(1)
        group_ids.append(group_idx)
```

### 4. Progressive Unfreezing

Start with more frozen layers, then unfreeze gradually:

```bash
# Epoch 0-3: Freeze first 6 layers
python3 training/train_crse.py ... --freeze-layers 6 --epochs 3

# Epoch 4-7: Freeze first 3 layers
python3 training/train_crse.py ... --freeze-layers 3 --epochs 4

# Epoch 8-10: Unfreeze all
python3 training/train_crse.py ... --freeze-layers 0 --epochs 3
```

### 5. Learning Rate Scheduling

Use warmup + linear decay (already built-in):

- Warmup: 10% of total steps
- Decay: Linear to 0

To adjust warmup:

```python
# In train_crse.py, modify:
warmup_steps = int(total_steps * 0.2)  # 20% warmup instead of 10%
```

---

## Troubleshooting

### Problem: Separation not increasing

**Possible causes:**
1. Attacks are too trivial → Filter with `--skip-trivial`
2. Learning rate too low → Try `--lr 5e-5`
3. Temperature too high → Try `--temperature 0.05`
4. Not enough negatives → Increase `--negatives-per-group`

### Problem: cos(anchor, para) decreasing

**Possible cause:** Too many negatives, model forgets honest proximity

**Solution:** Reduce `--negatives-per-group` to 1-2

### Problem: Training too slow

**Solutions:**
1. Increase `--batch-groups` (if VRAM allows)
2. Use smaller `--projection-dim` (256 instead of 512)
3. Reduce `--eval-every-steps` to 200 or 500

---

## Next Steps

- **Ablation studies**: Try different attack types, budgets, projection dims
- **Multi-topic training**: Combine datasets from different domains
- **Transfer learning**: Fine-tune on domain-specific data

---

## Questions?

Open an issue on [GitHub](https://github.com/HrxuAlbert/FBA_ENCODER/issues)

