# Data Preparation Guide

Learn how to prepare training data for CRSE from source texts, paraphrases, and Byzantine attacks.

---

## Overview

CRSE training requires three types of data:

1. **Anchor texts**: Original factual paragraphs (e.g., from Wikipedia)
2. **Positives**: Faithful paraphrases of anchors (honest rewrites)
3. **Negatives**: Byzantine attacks (subtle semantic corruptions)

These are combined into a unified JSONL dataset using `data_preparation/build_dataset.py`.

---

## Data Pipeline

```
Source Texts (anchors)
      ↓
      ├─→ Generate Paraphrases (LLM) → paraphrases.jsonl
      ├─→ Generate Attacks (LLM) → attacks.jsonl
      ↓
Build CRSE Dataset → crse_dataset.jsonl
      ↓
Train CRSE Model
```

---

## Step 1: Prepare Source Texts

### Format

Create a JSONL file where each line is a source paragraph:

```json
{"id": "source_001", "text": "Machine learning is a subset of AI that enables systems to learn from data...", "title": "Machine Learning", "topic": "science_tech"}
{"id": "source_002", "text": "Neural networks consist of layers...", "title": "Neural Networks", "topic": "science_tech"}
```

### Requirements

- **Length**: 50-500 words per paragraph (optimal: 100-200 words)
- **Domain**: Factual, encyclopedic content (science, technology, history, etc.)
- **Quality**: Well-written, grammatical, coherent
- **Quantity**: At least 50+ unique anchors for meaningful training

### Example Sources

- Wikipedia articles (e.g., science & technology topics)
- Academic abstracts
- News articles (factual sections)
- Technical documentation

---

## Step 2: Generate Paraphrases

### Option A: Use LLM API (Recommended)

CRSE benefits from high-quality **faithful paraphrases** that preserve factual content and causal structure.

**Requirements:**
- OpenAI API key (`export OPENAI_API_KEY=your-key`)
- 2-4 paraphrases per anchor

**Example prompt:**

```
You are a scientific writer. Generate 2 faithful paraphrases of the following paragraph.

Requirements:
- Preserve all factual content, causal relations, and conclusions
- Use different wording and sentence structures
- Do NOT add new facts or remove important caveats
- Remain encyclopedic and grammatical

Original paragraph:
{text}

Output as JSON:
{
  "paraphrases": [
    {"text": "First paraphrase...", "explanation": "..."},
    {"text": "Second paraphrase...", "explanation": "..."}
  ]
}
```

**Output format** (`paraphrases.jsonl`):

```json
{"source_id": "source_001", "paraphrases": [{"id": "source_001_P1", "text": "ML, a branch of AI, allows systems to learn...", "explanation": "..."}, {"id": "source_001_P2", "text": "As a subset of artificial intelligence, ML...", "explanation": "..."}]}
```

### Option B: Manual Paraphrasing

For small datasets, manually write 1-2 paraphrases per anchor.

**Guidelines:**
- Preserve all factual claims
- Do NOT invert conclusions
- Do NOT remove caveats or limitations
- Keep similar length and complexity

---

## Step 3: Generate Byzantine Attacks

### Attack Types

CRSE is trained to detect **four types of subtle semantic attacks**:

| Attack Type                  | Definition                                                                 | Example                                  |
|------------------------------|---------------------------------------------------------------------------|------------------------------------------|
| **B1_POLARITY_FLIP**         | Reverse factual claims or conclusions                                      | "X increases Y" → "X decreases Y"        |
| **B2_EVIDENCE_OMISSION**     | Remove caveats, conditions, or limitations                                 | Remove "in most cases" or "under X condition" |
| **B3_FAKE_CAUSALITY**        | Introduce false causal links or turn correlations into causation           | "X correlates with Y" → "X causes Y"     |
| **B4_ON_TOPIC_HALLUCINATION**| Add plausible but false details (dates, numbers, institutions, results)    | Invent "Stanford 2022 study found..."    |

### Budget Levels

| Budget   | Intensity                                      | Example                                  |
|----------|-----------------------------------------------|------------------------------------------|
| `mild`   | Modify ONE key claim                          | Flip one conclusion                      |
| `medium` | Modify MULTIPLE claims or parts of reasoning  | Flip 2-3 statements                      |
| `strong` | Apply attack aggressively                     | Multiple instances of corruption pattern |

### Option A: Use LLM API (Recommended)

**Requirements:**
- OpenAI API key
- 12 attacks per anchor (4 types × 3 budgets)

**Example prompt:**

```
You are an adversarial editor acting as a Byzantine node.

Generate 12 Byzantine attack variations of this paragraph:
- 4 attack types: B1_POLARITY_FLIP, B2_EVIDENCE_OMISSION, B3_FAKE_CAUSALITY, B4_ON_TOPIC_HALLUCINATION
- 3 budgets: mild, medium, strong
- All combinations: 4 × 3 = 12 attacks

[Detailed type definitions...]

Original paragraph:
{text}

Output as JSON:
{
  "B1_POLARITY_FLIP": {
    "mild": {"byzantine_text": "...", "explanation": "..."},
    "medium": {"byzantine_text": "...", "explanation": "..."},
    "strong": {"byzantine_text": "...", "explanation": "..."}
  },
  ...
}
```

**Output format** (`attacks.jsonl`):

```json
{"source_id": "source_001", "attack_type": "B1_POLARITY_FLIP", "budget": "mild", "byzantine_text": "Machine learning is not related to AI...", "is_trivial": false}
{"source_id": "source_001", "attack_type": "B1_POLARITY_FLIP", "budget": "medium", "byzantine_text": "...", "is_trivial": false}
...
```

### Option B: Manual Attack Generation

For small datasets, manually create 2-4 attacks per anchor.

**Focus on:**
- Subtlety (attacks should be plausible, not obviously wrong)
- Semantic corruption (not topical divergence)
- Diversity (cover different attack types)

---

## Step 4: Build CRSE Dataset

Combine anchors, paraphrases, and attacks into a unified dataset:

```bash
python3 data_preparation/build_dataset.py \
  --anchors data/source/anchors.jsonl \
  --paraphrases data/source/paraphrases.jsonl \
  --attacks data/source/attacks.jsonl \
  --output data/crse/crse_dataset.jsonl \
  --min-positives 1 \
  --min-byzantine 2 \
  --skip-trivial \
  --verbose
```

**Arguments:**
- `--min-positives`: Minimum paraphrases per anchor (default: 1)
- `--min-byzantine`: Minimum attacks per anchor (default: 1)
- `--skip-trivial`: Filter out attacks marked as `is_trivial=True`
- `--max-anchors`: Keep only N random anchors (for mini datasets)
- `--subset-seed`: Random seed for subset sampling

**Output** (`crse_dataset.jsonl`):

```json
{"anchor": {"text": "...", "source_id": "source_001"}, "positives": [...], "byzantine": [...], "stats": {"n_positives": 2, "n_byzantine": 12}}
```

---

## Data Quality Checks

Before training, verify your dataset:

### 1. Paraphrase Quality

Check that paraphrases preserve factual content:

```python
import json

with open("data/crse/crse_dataset.jsonl") as f:
    for line in f:
        record = json.loads(line)
        anchor = record['anchor']['text']
        for para in record['positives']:
            print(f"Anchor: {anchor[:100]}...")
            print(f"Para:   {para['text'][:100]}...")
            input("OK? (Press Enter)")
```

### 2. Attack Diversity

Check attack type distribution:

```bash
grep -o '"attack_type":"[^"]*"' data/crse/crse_dataset.jsonl | sort | uniq -c
```

Expected output:

```
  150 "attack_type":"B1_POLARITY_FLIP"
  150 "attack_type":"B2_EVIDENCE_OMISSION"
  150 "attack_type":"B3_FAKE_CAUSALITY"
  150 "attack_type":"B4_ON_TOPIC_HALLUCINATION"
```

### 3. Attack Non-Triviality

Ensure attacks are subtle:

```bash
# Count trivial attacks
grep -c '"is_trivial":true' data/source/attacks.jsonl

# Should be <10% of total
```

---

## Example Workflow

### Full Pipeline (100 anchors)

```bash
# 1. Prepare anchors
# (Manual or scripted from Wikipedia, etc.)
echo '{"id": "ml1", "text": "Machine learning...", "topic": "science"}' > anchors.jsonl

# 2. Generate paraphrases (requires OpenAI API)
# (Use your own script or LLM prompting tool)
python3 generate_paraphrases.py \
  --input anchors.jsonl \
  --output paraphrases.jsonl \
  --per-sample 2

# 3. Generate attacks (requires OpenAI API)
python3 generate_attacks.py \
  --input anchors.jsonl \
  --output attacks.jsonl

# 4. Build CRSE dataset
python3 data_preparation/build_dataset.py \
  --anchors anchors.jsonl \
  --paraphrases paraphrases.jsonl \
  --attacks attacks.jsonl \
  --output data/crse/dataset_100.jsonl

# 5. Verify
wc -l data/crse/dataset_100.jsonl  # Should output: 100
```

---

## Tips

### For Quick Testing

Create a **mini dataset** (10-20 anchors):

```bash
python3 data_preparation/build_dataset.py \
  ... \
  --max-anchors 20 \
  --subset-seed 42
```

### For Production Models

Use **300-1000 anchors** with:
- 2-4 paraphrases per anchor
- 8-12 attacks per anchor (2-3 per attack type)
- Diverse topics and writing styles

### LLM Best Practices

- Use **GPT-4o-mini** or **GPT-4** for generation
- Set `temperature=0.7` for diversity
- Use structured output (`response_format={"type": "json_object"}`)
- Manually audit 10-20 samples before scaling up

---

## Questions?

Open an issue on [GitHub](https://github.com/HrxuAlbert/FBA_ENCODER/issues)

