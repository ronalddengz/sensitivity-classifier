# PII/PHI and Quasi-Identifier Detection System

A Python-based privacy risk analysis toolkit that detects **Personally Identifiable Information (PII)** and **Quasi-Identifiers (QI)** in text, calculates weighted risk scores, and outputs structured JSON reports.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [E-Score (Explicit PII Detection)](#e-score-explicit-pii-detection)
  - [Q-Score (Quasi-Identifier Detection)](#q-score-quasi-identifier-detection)
- [Installation](#installation)
- [Usage](#usage)
- [Output Format](#output-format)
- [Configuration](#configuration)
- [Architecture](#architecture)

---

## Overview

This system provides two complementary analysis modules:

| Module | Purpose | Technology |
|--------|---------|------------|
| **E-Score** | Detects explicit PII (SSNs, emails, names, etc.) | Presidio + spaCy NLP |
| **Q-Score** | Detects quasi-identifiers and calculates re-identification risk | Pattern matching + Census API |

Together, they provide a comprehensive privacy risk assessment for any text input.

---

## How It Works

### E-Score (Explicit PII Detection)

The E-Score module uses [Microsoft Presidio](https://github.com/microsoft/presidio) with a spaCy NLP backend to detect explicit identifiers.

#### Detection Flow

```
Input Text
    │
    ▼
┌─────────────────────────────────┐
│  Presidio Analyzer Engine       │
│  (spaCy en_core_web_lg model)   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Entity Recognition             │
│  • Named Entity Recognition     │
│  • Pattern Matching (SSN, etc.) │
│  • Checksum Validation          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Scoring Pipeline               │
│  1. Get Presidio confidence     │
│  2. Apply sensitivity weight    │
│  3. Calculate weighted_score    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  E-Score Calculation            │
│  E = 0.6 × max(scores) +        │
│      0.4 × avg(scores)          │
│  (capped at 1.0)                │
└─────────────────────────────────┘
    │
    ▼
JSON Output
```

#### Sensitivity Weights

Each entity type has a predefined sensitivity weight (0.0 to 1.0):

| Category | Entity Types | Weight |
|----------|--------------|--------|
| **Direct Identifiers** | SSN, Passport, Driver License | 1.0 |
| **Financial** | Bank Account, Credit Card, ITIN | 0.95 |
| **Strong Identifiers** | Person Name, Email | 0.85-0.90 |
| **Contact Info** | Phone, IP Address | 0.75-0.80 |
| **Location/Temporal** | Location, Date/Time | 0.50-0.60 |
| **Other** | Organization, URL | 0.40-0.50 |

#### E-Score Formula

```
weighted_score = presidio_confidence × sensitivity_weight

E-Score = min(0.6 × max(weighted_scores) + 0.4 × mean(weighted_scores), 1.0)
```

The formula prioritizes the highest-risk entity (60% weight) while considering overall PII density (40% weight).

---

### Q-Score (Quasi-Identifier Detection)

The Q-Score module identifies quasi-identifiers and estimates re-identification risk using **k-anonymity** principles with real population data.

#### What are Quasi-Identifiers?

Quasi-identifiers are attributes that aren't directly identifying alone but can uniquely identify individuals when combined:

- Age + Gender + ZIP Code + Rare Disease → Potentially unique person

#### Detection Flow

```
Input Text
    │
    ▼
┌─────────────────────────────────┐
│  QI Extractor                   │
│  • Regex patterns (age, ZIP)    │
│  • Keyword matching (diseases)  │
│  • State name recognition       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Population Data Lookup         │
│  • Census API (demographics)    │
│  • Disease prevalence database  │
│  • BLS occupation statistics    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Joint Probability Calculation  │
│  P(joint) = P(age) × P(gender)  │
│           × P(location) × ...   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Expected k Calculation         │
│  E[k] = US_Population × P(joint)│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Q-Score Calculation            │
│  if E[k] ≥ k_threshold: 0.0     │
│  else: 1 - (E[k] / k_threshold) │
└─────────────────────────────────┘
    │
    ▼
JSON Output
```

#### Population Data Sources

| QI Type | Data Source | Method |
|---------|-------------|--------|
| **Age** | Census ACS Table B01001 | API call for age distribution |
| **Gender** | Census Bureau | Static 50.8% F / 49.2% M |
| **Location** | Census ACS by State FIPS | API call for state population |
| **ZIP Code** | Census ACS ZCTA | API call for ZIP population |
| **Disease** | Orphanet + CDC | Local prevalence database |
| **Occupation** | Bureau of Labor Statistics | Local employment database |

#### Q-Score Formula

```
P(joint) = ∏ P(qi)  for all detected quasi-identifiers

E[k] = 331,900,000 × P(joint)   # Expected equivalence class size

Q-Score = 
  0.0                           if E[k] ≥ k_threshold (default: 5)
  1.0 - (E[k] / k_threshold)    if E[k] < k_threshold
```

#### Risk Interpretation

| E[k] Value | Q-Score | Risk Level | Meaning |
|------------|---------|------------|---------|
| E[k] < 1 | ~1.0 | **CRITICAL** | Likely uniquely identifying |
| 1 ≤ E[k] < 5 | 0.6-0.8 | **HIGH** | Small anonymity set |
| 5 ≤ E[k] < 20 | 0.0-0.4 | **MEDIUM** | Moderate protection |
| E[k] ≥ 20 | 0.0 | **LOW** | Good k-anonymity |

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Step 1: Install Dependencies

```bash
pip install presidio-analyzer presidio-anonymizer spacy python-dotenv requests
```

### Step 2: Download spaCy Model

```bash
python -m spacy download en_core_web_lg
```

### Step 3: (Optional) Set Up Census API Key

For accurate population data, get a free API key from [Census Bureau](https://api.census.gov/data/key_signup.html):

```bash
# Create .env file
echo "CENSUS_API_KEY=your_key_here" > .env
```

The system works without an API key but may hit rate limits or use fallback estimates.

---

## Usage

### Basic Usage

```python
from pii_detector import detect_pii
from qscore import calculate_qscore

text = """
Patient Jane Doe, 45-year-old female from Wyoming.
Diagnosed with Ehlers-Danlos syndrome.
SSN: 123-45-6789, Email: jane.doe@email.com
ZIP: 82001
"""

# Detect explicit PII
pii_result = detect_pii(text)
print(f"E-Score: {pii_result.e_score}")
print(pii_result.to_json())

# Detect quasi-identifiers
qi_report = calculate_qscore(text)
print(f"Q-Score: {qi_report['q_score']}")
print(f"Expected k: {qi_report['expected_k']}")
```

### Save Results to Files

```python
# Save PII analysis
detect_pii(text, output_path="pii_analysis.json")

# Save Q-Score analysis
calculate_qscore(text, output_path="qscore_analysis.json")
```

### Command Line

```bash
# Run PII detection
python pii_detector.py

# Run Q-Score calculation
python qscore.py
```

---

## Output Format

### E-Score Output (pii_result.json)

```json
{
  "text_length": 156,
  "timestamp": "2024-01-15T10:30:00.000000",
  "e_score": 0.8234,
  "entity_count": 4,
  "entities": [
    {
      "text": "Jane Doe",
      "entity_type": "PERSON",
      "start": 8,
      "end": 16,
      "confidence": 0.95,
      "weight": 0.90,
      "weighted_score": 0.855,
      "context": "Patient Jane Doe, 45-year-old..."
    },
    {
      "text": "123-45-6789",
      "entity_type": "US_SSN",
      "start": 89,
      "end": 100,
      "confidence": 1.0,
      "weight": 1.0,
      "weighted_score": 1.0,
      "context": "...syndrome. SSN: 123-45-6789, Email: jane..."
    }
  ]
}
```

### Q-Score Output (qscore_result.json)

```json
{
  "q_score": 0.9847,
  "expected_k": 0.076,
  "risk_level": "HIGH",
  "k_threshold": 5,
  "detected_quasi_identifiers": [
    {
      "type": "age",
      "raw_value": "45-year-old",
      "normalized_value": "45",
      "confidence": 0.9,
      "position": {"start": 22, "end": 33},
      "frequency": {
        "probability": 0.0128,
        "population_count": 4248320,
        "source": "Census ACS B01001"
      }
    },
    {
      "type": "gender",
      "raw_value": "female",
      "normalized_value": "female",
      "confidence": 0.95,
      "frequency": {
        "probability": 0.508,
        "population_count": 168605200,
        "source": "Census Bureau"
      }
    },
    {
      "type": "location",
      "raw_value": "Wyoming",
      "normalized_value": "Wyoming",
      "confidence": 0.85,
      "frequency": {
        "probability": 0.00174,
        "population_count": 576851,
        "source": "Census ACS"
      }
    },
    {
      "type": "disease",
      "raw_value": "Ehlers-Danlos syndrome",
      "normalized_value": "ehlers-danlos syndrome",
      "confidence": 0.95,
      "frequency": {
        "probability": 0.0002,
        "population_count": 66380,
        "source": "Orphanet/CDC prevalence data"
      }
    }
  ],
  "explanation": "Detected 4 quasi-identifier(s):\n  - age: '45' (P ≈ 1.28e-02, ~4,248,320 people)\n  - gender: 'female' (P ≈ 5.08e-01, ~168,605,200 people)\n  - location: 'Wyoming' (P ≈ 1.74e-03, ~576,851 people)\n  - disease: 'ehlers-danlos syndrome' (P ≈ 2.00e-04, ~66,380 people)\n\nExpected equivalence class size E[k]: 0.08\n⚠️  E[k] < 1: Likely UNIQUELY IDENTIFYING"
}
```

---

## Configuration

### E-Score Configuration

Edit `SENSITIVITY_WEIGHTS` dictionary to adjust entity weights:

```python
SENSITIVITY_WEIGHTS = {
    "US_SSN": 1.0,        # Highest sensitivity
    "PERSON": 0.90,
    "EMAIL_ADDRESS": 0.85,
    # ... add custom weights
}

DEFAULT_WEIGHT = 0.50     # For unrecognized entity types
```

Adjust detection threshold in `PIIDetector`:

```python
detector = PIIDetector(
    score_threshold=0.4,   # Minimum Presidio confidence
    context_window=50      # Characters of context to include
)
```

### Q-Score Configuration

Adjust k-anonymity threshold:

```python
analyzer = QScoreAnalyzer(k_threshold=5)  # Default: 5
```

Add custom diseases or occupations:

```python
# In DiseaseDataSource.DISEASE_PREVALENCE
"custom_condition": 1/50000,  # prevalence rate

# In OccupationDataSource.OCCUPATION_EMPLOYMENT  
"custom_job": 150000,  # employment count
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Text                           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│     PII Detector        │     │      Q-Score Analyzer       │
│  ┌───────────────────┐  │     │  ┌───────────────────────┐  │
│  │ Presidio Engine   │  │     │  │   QI Extractor        │  │
│  │ + spaCy NLP       │  │     │  │   (Regex + Keywords)  │  │
│  └───────────────────┘  │     │  └───────────────────────┘  │
│           │             │     │             │               │
│           ▼             │     │             ▼               │
│  ┌───────────────────┐  │     │  ┌───────────────────────┐  │
│  │ Weight Lookup     │  │     │  │  Population Lookup    │  │
│  │ (Sensitivity Map) │  │     │  │  • Census API         │  │
│  └───────────────────┘  │     │  │  • Disease DB         │  │
│           │             │     │  │  • Occupation DB      │  │
│           ▼             │     │  └───────────────────────┘  │
│  ┌───────────────────┐  │     │             │               │
│  │ E-Score Calc      │  │     │             ▼               │
│  │ 0.6×max + 0.4×avg │  │     │  ┌───────────────────────┐  │
│  └───────────────────┘  │     │  │  Q-Score Calc         │  │
└─────────────────────────┘     │  │  E[k] → Risk Score    │  │
              │                 │  └───────────────────────┘  │
              │                 └─────────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   PIIAnalysisResult     │     │      QScoreResult           │
│   • entities[]          │     │      • q_score              │
│   • e_score             │     │      • expected_k           │
│   • to_json()           │     │      • detected_qis[]       │
└─────────────────────────┘     └─────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    ┌─────────────────┐
                    │   JSON Output   │
                    └─────────────────┘
```

---

## Key Concepts

### Why Two Scores?

| Score | Detects | Example | Risk Type |
|-------|---------|---------|-----------|
| **E-Score** | Direct identifiers | SSN, Name, Email | Direct identification |
| **Q-Score** | Indirect identifiers | Age + ZIP + Disease | Re-identification via linkage |

### k-Anonymity Explained

k-anonymity ensures that any individual in a dataset is indistinguishable from at least k-1 other individuals based on quasi-identifiers.

- **E[k] = 100**: ~100 people share this QI combination → Low risk
- **E[k] = 5**: ~5 people share this combination → Borderline
- **E[k] = 0.5**: Less than 1 person expected → Uniquely identifying

### Independence Assumption

The Q-Score calculation assumes quasi-identifiers are statistically independent:

```
P(45yo ∩ female ∩ Wyoming ∩ EDS) = P(45yo) × P(female) × P(Wyoming) × P(EDS)
```

This is a simplification—real-world correlations may exist (e.g., certain diseases correlate with age).

---

## Limitations

1. **English only**: spaCy model and patterns are English-focused
2. **US-centric**: Census data and patterns are US-specific
3. **Independence assumption**: QI correlations not modeled
4. **Pattern coverage**: Not all QI patterns are captured
5. **API rate limits**: Census API may throttle without key

---

## License

MIT License