# Quick Start Guide

This guide will walk you through training your first CRSE model in **under 30 minutes**.

---

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- 8GB+ RAM (local) or GPU with 16GB+ VRAM (for full training)

---

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/HrxuAlbert/FBA_ENCODER.git
cd FBA_ENCODER

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Prepare a Mini Dataset

For quick testing, create a minimal dataset with just a few anchor groups.

Create `data/crse/mini_dataset.jsonl`:

```json
{"anchor": {"text": "Machine learning is a subset of artificial intelligence.", "source_id": "ml1"}, "positives": [{"text": "ML is part of AI.", "id": "ml1_P1"}], "byzantine": [{"text": "Machine learning is completely unrelated to AI.", "id": "ml1_B1", "attack_type": "B1_POLARITY_FLIP", "budget": "strong"}], "stats": {"n_positives": 1, "n_byzantine": 1}}
{"anchor": {"text": "Neural networks consist of layers of interconnected nodes.", "source_id": "nn1"}, "positives": [{"text": "NNs have layers with connected nodes.", "id": "nn1_P1"}], "byzantine": [{"text": "Neural networks have no layers or connections.", "id": "nn1_B1", "attack_type": "B1_POLARITY_FLIP", "budget": "strong"}], "stats": {"n_positives": 1, "n_byzantine": 1}}
```

---

## Step 3: Train Locally

Run training with the `local_mini` profile:

```bash
python3 training/train_crse.py \
  --profile local_mini \
  --data data/crse/mini_dataset.jsonl \
  --output-dir checkpoints/crse_quickstart \
  --epochs 2 \
  --batch-groups 2
```

**Expected output:**
```
================================================================================
CRSE Training
================================================================================
Device: mps (or cpu)
Data: data/crse/mini_dataset.jsonl
...
Pre-training geometry evaluation...
  cos(anchor, paraphrase): mean=0.8500, std=0.0200
  cos(anchor, byzantine):  mean=0.7800, std=0.0300
...
Epoch 1/2:   100%|██████████| ... loss: 0.3245
...
✅ Training complete! Model saved to: checkpoints/crse_quickstart
```

---

## Step 4: Load and Use the Model

Create a test script `test_model.py`:

```python
import torch
from crse.model import CRSEModel

# Load model
model = CRSEModel(projection_dim=256)  # Match training config
checkpoint = torch.load("checkpoints/crse_quickstart/best_model.pt", map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint)
model.eval()

# Test texts
honest_pair = [
    "Machine learning is a subset of AI.",
    "ML is part of artificial intelligence."
]

attack_pair = [
    "Machine learning is a subset of AI.",
    "Machine learning is completely unrelated to AI."
]

# Encode
with torch.no_grad():
    honest_emb = model.encode(honest_pair, normalize=True)
    attack_emb = model.encode(attack_pair, normalize=True)

# Compute similarities
honest_sim = torch.dot(honest_emb[0], honest_emb[1]).item()
attack_sim = torch.dot(attack_emb[0], attack_emb[1]).item()

print(f"Honest paraphrase similarity: {honest_sim:.4f}")
print(f"Byzantine attack similarity:  {attack_sim:.4f}")
print(f"Separation: {honest_sim - attack_sim:.4f}")
```

Run it:

```bash
python3 test_model.py
```

**Expected output:**
```
Honest paraphrase similarity: 0.9234
Byzantine attack similarity:  0.4567
Separation: 0.4667
```

---

## Step 5: Full Training (Optional)

For production-quality models, train on a larger dataset with GPU:

```bash
# On Google Colab or a machine with CUDA
python3 training/train_crse.py \
  --profile gpu_full \
  --data data/crse/full_dataset.jsonl \
  --output-dir checkpoints/crse_full \
  --epochs 10 \
  --batch-groups 32
```

---

## Next Steps

- **Read** [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for advanced training options
- **Read** [DATA_PREPARATION.md](DATA_PREPARATION.md) to learn how to prepare larger datasets
- **Experiment** with different `projection_dim`, `freeze_layers`, and `temperature` values

---

## Troubleshooting

### Issue: "RuntimeError: MPS backend not available"

**Solution**: Your Mac doesn't support MPS. Use CPU instead:

```bash
python3 training/train_crse.py \
  --profile local_mini \
  --data data/crse/mini_dataset.jsonl \
  --device cpu
```

### Issue: "CUDA out of memory"

**Solution**: Reduce `batch_groups`:

```bash
python3 training/train_crse.py \
  --profile gpu_full \
  --data data/crse/full_dataset.jsonl \
  --batch-groups 16  # Instead of 32
```

### Issue: Training loss not decreasing

**Possible causes:**
1. Dataset too small (need at least 50+ anchor groups)
2. Learning rate too high or too low
3. All attacks are trivial (filter them with `--skip-trivial` in dataset builder)

---

## Questions?

Open an issue on [GitHub](https://github.com/HrxuAlbert/FBA_ENCODER/issues) or contact:  
📧 2614067X@student.gla.ac.uk

