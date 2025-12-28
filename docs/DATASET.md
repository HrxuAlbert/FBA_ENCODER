# CRSE Dataset Specification

Complete specification of the CRSE training dataset format, structure, and requirements.

---

## Overview

CRSE training uses a **structured JSONL dataset** where each line represents one **anchor group** consisting of:
- 1 anchor text (original source)
- N positive samples (honest paraphrases)
- M negative samples (Byzantine attacks)

**File format**: JSONL (newline-delimited JSON)  
**Encoding**: UTF-8  
**Extension**: `.jsonl`

---

## Dataset Structure

### File Format

```jsonl
{"anchor": {...}, "positives": [...], "byzantine": [...], "stats": {...}}
{"anchor": {...}, "positives": [...], "byzantine": [...], "stats": {...}}
...
```

Each line is a complete JSON object representing one anchor group.

### Anchor Group Schema

```json
{
  "anchor": {
    "text": "Original paragraph from Wikipedia or other source...",
    "source_id": "unique_anchor_id",
    "title": "Article Title",
    "topic": "science_tech",
    "tier": "featured",
    "page_id": "12345"
  },
  "positives": [
    {
      "id": "unique_anchor_id_P1",
      "text": "Faithful paraphrase that preserves factual content...",
      "source": "paraphrase"
    },
    {
      "id": "unique_anchor_id_P2",
      "text": "Another faithful paraphrase with different wording...",
      "source": "paraphrase"
    }
  ],
  "byzantine": [
    {
      "id": "unique_anchor_id_B1_mild",
      "text": "Byzantine attack text with subtle semantic corruption...",
      "attack_type": "B1_POLARITY_FLIP",
      "budget": "mild",
      "is_trivial": false,
      "cos_sim": 0.8234,
      "angle_deg": 34.56
    },
    ...
  ],
  "stats": {
    "n_positives": 2,
    "n_byzantine": 12,
    "byz_per_attack_type": {
      "B1_POLARITY_FLIP": 3,
      "B2_EVIDENCE_OMISSION": 3,
      "B3_FAKE_CAUSALITY": 3,
      "B4_ON_TOPIC_HALLUCINATION": 3
    },
    "byz_per_budget": {
      "mild": 4,
      "medium": 4,
      "strong": 4
    }
  }
}
```

---

## Field Descriptions

### Anchor Object

| Field       | Type   | Required | Description                                      |
|-------------|--------|----------|--------------------------------------------------|
| `text`      | string | ✅       | Original source paragraph (50-500 words)         |
| `source_id` | string | ✅       | Unique identifier for this anchor                |
| `title`     | string | ❌       | Article/document title                           |
| `topic`     | string | ❌       | Topic category (e.g., "science_tech")            |
| `tier`      | string | ❌       | Quality tier (e.g., "featured", "standard")      |
| `page_id`   | string | ❌       | Source page ID (e.g., Wikipedia page ID)         |

### Positive Object

| Field    | Type   | Required | Description                                      |
|----------|--------|----------|--------------------------------------------------|
| `id`     | string | ✅       | Unique identifier (typically `{source_id}_P{n}`) |
| `text`   | string | ✅       | Faithful paraphrase preserving factual content   |
| `source` | string | ❌       | Source type (e.g., "paraphrase")                 |

**Requirements for positives:**
- Must preserve all factual claims and conclusions
- Must NOT add new information not present in anchor
- Must NOT remove important caveats or limitations
- Should use different wording and sentence structures

### Byzantine Object

| Field         | Type    | Required | Description                                   |
|---------------|---------|----------|-----------------------------------------------|
| `id`          | string  | ✅       | Unique identifier (e.g., `{source_id}_B1_mild`) |
| `text`        | string  | ✅       | Attack text with semantic corruption          |
| `attack_type` | string  | ✅       | Attack type (B1/B2/B3/B4)                     |
| `budget`      | string  | ✅       | Intensity level (mild/medium/strong)          |
| `is_trivial`  | boolean | ❌       | Whether attack is trivially detectable        |
| `cos_sim`     | float   | ❌       | Cosine similarity to anchor (baseline encoder)|
| `angle_deg`   | float   | ❌       | Angular distance in degrees                   |

**Attack types:**

1. **B1_POLARITY_FLIP**: Reverse factual claims or conclusions
   - Example: "X increases Y" → "X decreases Y"
   
2. **B2_EVIDENCE_OMISSION**: Remove caveats, conditions, or limitations
   - Example: Remove "under specific conditions" or "in most cases"
   
3. **B3_FAKE_CAUSALITY**: Introduce false causal links
   - Example: "X correlates with Y" → "X causes Y"
   
4. **B4_ON_TOPIC_HALLUCINATION**: Add plausible but false details
   - Example: Invent "Stanford 2022 study found..."

**Budget levels:**
- `mild`: Modify ONE key claim
- `medium`: Modify MULTIPLE claims or reasoning parts
- `strong`: Apply attack aggressively with multiple instances

### Stats Object

| Field                  | Type   | Description                                   |
|------------------------|--------|-----------------------------------------------|
| `n_positives`          | int    | Number of positive samples in this group      |
| `n_byzantine`          | int    | Number of Byzantine samples in this group     |
| `byz_per_attack_type`  | dict   | Count of attacks per type (B1/B2/B3/B4)       |
| `byz_per_budget`       | dict   | Count of attacks per budget (mild/medium/strong) |

---

## Dataset Sizes

### Recommended Sizes

| Purpose              | Anchor Groups | Positives/Anchor | Byzantines/Anchor | Total Samples |
|----------------------|---------------|------------------|-------------------|---------------|
| **Quick Testing**    | 10-20         | 1-2              | 2-4               | ~60-100       |
| **Local Training**   | 50-100        | 2                | 8-12              | ~550-1200     |
| **Full Training**    | 300-1000      | 2-4              | 8-12              | ~3300-14000   |
| **Production**       | 1000+         | 2-4              | 12                | 14000+        |

### Example Statistics

From a typical CRSE dataset:

```
Total anchor groups:              500
Average positives per anchor:     2.3
Average byzantines per anchor:    11.8

Total positive samples:           1,150
Total byzantine samples:          5,900

Attack type distribution:
  B1_POLARITY_FLIP:              1,475 (25%)
  B2_EVIDENCE_OMISSION:          1,475 (25%)
  B3_FAKE_CAUSALITY:             1,475 (25%)
  B4_ON_TOPIC_HALLUCINATION:     1,475 (25%)

Budget distribution:
  mild:                          1,967 (33%)
  medium:                        1,967 (33%)
  strong:                        1,966 (33%)

Average text length:
  Anchor:                        128.5 words
  Positive:                      131.2 words
  Byzantine:                     132.8 words
```

---

## Quality Requirements

### Anchor Texts

- ✅ **Length**: 50-500 words (optimal: 100-200)
- ✅ **Language**: Fluent, grammatical English
- ✅ **Domain**: Factual, encyclopedic content
- ✅ **Quality**: Well-written, coherent paragraphs
- ✅ **Diversity**: Cover multiple topics and writing styles

### Positive Samples (Paraphrases)

- ✅ **Faithfulness**: Preserve all factual claims
- ✅ **Completeness**: Do NOT omit important information
- ✅ **Consistency**: Do NOT contradict the anchor
- ✅ **Diversity**: Use different wording and structures
- ✅ **Naturalness**: Fluent and grammatical

### Byzantine Samples (Attacks)

- ✅ **Subtlety**: Not obviously wrong or off-topic
- ✅ **Semantic**: Corruption in meaning, not topic
- ✅ **Plausibility**: Sound believable to casual readers
- ✅ **Diversity**: Cover all attack types and budgets
- ✅ **Non-triviality**: Not detectable by simple keyword matching

---

## Validation Checklist

Before using a dataset for training, verify:

### 1. Format Validation

```bash
# Check valid JSON
python3 -c "import json; [json.loads(l) for l in open('dataset.jsonl')]"

# Count anchor groups
wc -l dataset.jsonl
```

### 2. Field Validation

```python
import json

with open('dataset.jsonl') as f:
    for i, line in enumerate(f, 1):
        record = json.loads(line)
        
        # Required fields
        assert 'anchor' in record
        assert 'positives' in record
        assert 'byzantine' in record
        assert 'stats' in record
        
        # Anchor required fields
        assert 'text' in record['anchor']
        assert 'source_id' in record['anchor']
        
        # Non-empty lists
        assert len(record['positives']) > 0
        assert len(record['byzantine']) > 0
        
        print(f"✅ Line {i}: Valid")
```

### 3. Quality Checks

```python
import json
import numpy as np

text_lengths = []
n_positives = []
n_byzantine = []

with open('dataset.jsonl') as f:
    for line in f:
        record = json.loads(line)
        text_lengths.append(len(record['anchor']['text'].split()))
        n_positives.append(record['stats']['n_positives'])
        n_byzantine.append(record['stats']['n_byzantine'])

print(f"Anchor text length: {np.mean(text_lengths):.1f} ± {np.std(text_lengths):.1f} words")
print(f"Positives per anchor: {np.mean(n_positives):.1f} ± {np.std(n_positives):.1f}")
print(f"Byzantines per anchor: {np.mean(n_byzantine):.1f} ± {np.std(n_byzantine):.1f}")
```

### 4. Attack Distribution

```bash
# Check attack type distribution
grep -o '"attack_type":"[^"]*"' dataset.jsonl | sort | uniq -c

# Check budget distribution
grep -o '"budget":"[^"]*"' dataset.jsonl | sort | uniq -c
```

---

## Building Your Dataset

Use `data_preparation/build_dataset.py` to combine separate files:

```bash
python3 data_preparation/build_dataset.py \
  --anchors data/source/anchors.jsonl \
  --paraphrases data/source/paraphrases.jsonl \
  --attacks data/source/attacks.jsonl \
  --output data/crse/dataset.jsonl \
  --min-positives 1 \
  --min-byzantine 2 \
  --skip-trivial \
  --verbose
```

**Flags:**
- `--min-positives N`: Require at least N paraphrases per anchor
- `--min-byzantine N`: Require at least N attacks per anchor
- `--skip-trivial`: Filter out trivial attacks (`is_trivial=True`)
- `--max-anchors N`: Keep only N random anchors (for mini datasets)

---

## Example Mini Dataset

For quick testing, create a minimal dataset:

```json
{"anchor": {"text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.", "source_id": "ml_001", "title": "Machine Learning"}, "positives": [{"id": "ml_001_P1", "text": "ML, a branch of AI, allows systems to learn from data automatically.", "source": "paraphrase"}], "byzantine": [{"id": "ml_001_B1_strong", "text": "Machine learning is completely unrelated to artificial intelligence.", "attack_type": "B1_POLARITY_FLIP", "budget": "strong", "is_trivial": false}], "stats": {"n_positives": 1, "n_byzantine": 1}}
```

Save as `mini_dataset.jsonl` and use for training:

```bash
python3 training/train_crse.py \
  --profile local_mini \
  --data mini_dataset.jsonl \
  --epochs 2 \
  --batch-groups 2
```

---

## Further Reading

- **[Data Preparation Guide](DATA_PREPARATION.md)**: How to prepare source data
- **[Training Guide](TRAINING_GUIDE.md)**: How to train CRSE models
- **[Quick Start](QUICKSTART.md)**: Step-by-step tutorial

---

## Questions?

Open an issue on [GitHub](https://github.com/HrxuAlbert/FBA_ENCODER/issues)

