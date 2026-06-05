# CRSE / FBA Encoder

**CRSE / FBA Encoder** is a robust semantic encoder designed for Byzantine-resilient multi-agent agreement. It aims to map semantically consistent honest proposals close to each other while separating adversarial or semantically poisoned proposals in the embedding space.

This repository supports my broader research on **trustworthy multi-agent AI systems**, **semantic commitment**, and **Byzantine-resilient collaboration**, where multiple agents may submit natural-language or structured proposals under disagreement, uncertainty, or adversarial manipulation.

## Motivation

Modern multi-agent AI systems increasingly rely on natural-language proposals, explanations, and evidence-grounded judgments. However, standard semantic encoders may fail to distinguish between honest semantic variations and Byzantine semantic attacks, such as polarity flips, misleading evidence use, hallucinated causal relations, or adversarially shifted claims.

This project explores contrastive training strategies for improving semantic separation between honest and Byzantine proposals, with the goal of supporting downstream mechanisms such as fuzzy Byzantine agreement, certified semantic commitment, and selective commitment in LLM-agent collaboration.

## Key Features

* Transformer-based semantic encoding for proposal-level representation.
* Contrastive learning for honest-honest alignment and honest-Byzantine separation.
* Support for semantic attack patterns such as polarity flip, semantic shift, and misleading claim transformation.
* Designed as a research module for Byzantine-resilient semantic agreement and trustworthy multi-agent AI systems.

## Research Context

This repository is part of a broader research agenda on protocol-level trust mechanisms for multi-agent AI collaboration, including:

* Byzantine-resilient semantic agreement;
* semantic robustness under adversarial disagreement;
* evidence-aware and certificate-based commitment;
* selective commitment and safe-abort mechanisms for LLM-agent systems.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/HrxuAlbert/FBA_ENCODER.git
cd FBA_ENCODER

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+

---

## 🚀 Quick Start

### 1. Prepare Your Data

CRSE training requires a JSONL dataset where each line is an "anchor group":

```json
{
  "anchor": {"text": "Original paragraph...", "source_id": "id1"},
  "positives": [
    {"text": "Faithful paraphrase...", "id": "id1_P1"}
  ],
  "byzantine": [
    {"text": "Semantic attack...", "id": "id1_B1", "attack_type": "B1_POLARITY_FLIP", "budget": "mild"}
  ],
  "stats": {"n_positives": 1, "n_byzantine": 12}
}
```

Use `data_preparation/build_dataset.py` to construct this from separate files:

```bash
python3 data_preparation/build_dataset.py \
  --anchors path/to/source.jsonl \
  --paraphrases path/to/paraphrases.jsonl \
  --attacks path/to/attacks.jsonl \
  --output data/crse/dataset.jsonl
```

### 2. Train the Model

#### Local Training (Mac/CPU)

```bash
python3 training/train_crse.py \
  --profile local_mini \
  --data data/crse/dataset.jsonl \
  --output-dir checkpoints/crse_local
```

#### GPU Training (CUDA)

```bash
python3 training/train_crse.py \
  --profile gpu_full \
  --data data/crse/dataset.jsonl \
  --output-dir checkpoints/crse_gpu \
  --batch-groups 32 \
  --epochs 10
```

### 3. Use the Trained Model

```python
import torch
from crse.model import CRSEModel

# Load model
model = CRSEModel(projection_dim=512)
model.load_state_dict(torch.load("checkpoints/crse_gpu/best_model.pt"))
model.eval()

# Encode texts
texts = ["This is a test.", "Another sentence."]
embeddings = model.encode(texts, normalize=True)

print(embeddings.shape)  # (2, 512)
```

---

## 📂 Repository Structure

```
FBA_ENCODER/
├── crse/                      # CRSE model core
│   ├── __init__.py
│   ├── model.py               # CRSEModel definition
│   └── config.py              # Training configurations
│
├── training/                  # Training scripts
│   ├── train_crse.py          # Main training script
│   ├── dataset.py             # Dataset loader
│   ├── loss.py                # Contrastive loss
│   └── evaluation.py          # Geometry evaluation
│
├── data_preparation/          # Data preprocessing tools
│   └── build_dataset.py       # Build CRSE dataset
│
├── scripts/                   # Helper scripts
│   ├── train_local.sh         # Local training script
│   └── train_gpu.sh           # GPU training script
│
├── docs/                      # Documentation
│   ├── TRAINING_GUIDE.md      # Detailed training guide
│   ├── DATA_PREPARATION.md    # Data preparation guide
│   └── QUICKSTART.md          # Quick start tutorial
│
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── .gitignore                 # Git ignore rules
```

---

## 🔬 Training Strategy

CRSE uses **anchor-centered contrastive learning**:

1. **Anchor groups**: Each training sample consists of:
   - 1 anchor text (original)
   - N positives (faithful paraphrases)
   - M negatives (Byzantine attacks)

2. **Contrastive loss**: InfoNCE loss that:
   - Pulls anchor and honest paraphrases together (high cosine similarity)
   - Pushes anchor and Byzantine attacks apart (low cosine similarity)

3. **Byzantine attacks**: Four attack types with three intensity levels:
   - `B1_POLARITY_FLIP`: Reverse factual claims
   - `B2_EVIDENCE_OMISSION`: Remove caveats/limitations
   - `B3_FAKE_CAUSALITY`: Introduce false causal links
   - `B4_ON_TOPIC_HALLUCINATION`: Add plausible but false details
   - Budgets: `mild`, `medium`, `strong`

4. **Evaluation metric**: Separation = cos(anchor, para) - cos(anchor, byz)

---

## 📊 Training Profiles

| Profile      | Device    | Batch Groups | Epochs | Projection Dim | Use Case                |
|--------------|-----------|--------------|--------|----------------|-------------------------|
| `local_mini` | MPS/CPU   | 8            | 5      | 256            | Quick local testing     |
| `gpu_full`   | CUDA      | 32           | 10     | 512            | Full GPU training       |

Customize with command-line arguments:

```bash
python3 training/train_crse.py \
  --profile gpu_full \
  --data data/crse/dataset.jsonl \
  --batch-groups 64 \
  --epochs 15 \
  --lr 1e-5
```

---

## 📖 Documentation

- **[Quick Start Tutorial](docs/QUICKSTART.md)**: Step-by-step beginner's guide
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Detailed training instructions
- **[Data Preparation Guide](docs/DATA_PREPARATION.md)**: How to prepare training data
- **[Dataset Specification](docs/DATASET.md)**: Complete dataset format and requirements

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Author**: Haoran Xu  
**Email**: 2614067X@student.gla.ac.uk  
**Institution**: University of Glasgow  
**GitHub**: [@HrxuAlbert](https://github.com/HrxuAlbert)

---

## 🔗 Related Projects

- **RAP-BFT**: Byzantine Fault-Tolerant consensus using CRSE  
  https://github.com/HrxuAlbert/RAP-BFT

---

## ⭐ Citation

If you use CRSE in your research, please cite:

```bibtex
@misc{crse2025,
  author = {Xu, Haoran},
  title = {CRSE: Certified Robust Semantic Encoder},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HrxuAlbert/FBA_ENCODER}
}
```

---

## 🙏 Acknowledgments

- Built on [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
- Inspired by contrastive learning literature in NLP
- Byzantine attack taxonomy designed for distributed consensus systems

