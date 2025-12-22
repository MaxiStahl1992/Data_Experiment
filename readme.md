# Political Polarization Analysis: Reddit & News Media

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Measuring and forecasting political polarization in online discourse using NLP and time series analysis**

## Overview

This repository contains a complete pipeline for analyzing political polarization across social media (Reddit) and traditional news media. The pipeline processes raw text data through topic classification, stance detection, polarization measurement, and forecasting.

**Research Period:** September-October 2016 (US Presidential Election)

**Platforms:**

- Reddit (political subreddits)
- News articles (US outlets via Common Crawl)

**Topics Analyzed:**

- Climate Change
- Donald Trump
- Gun Control
- Immigration
- Vaccination

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 8GB+ RAM recommended
- GPU optional (speeds up deep learning models)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Data_Experiment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Getting Started

**Start with the overview notebook:**

```bash
jupyter notebook 00_project_overview.ipynb
```

This notebook provides:

- Complete pipeline walkthrough
- Data statistics and visualizations
- Key findings and challenges
- Thesis direction recommendations

## 📊 Pipeline Overview

```
Raw Text Data
    ↓
[1] Filtering & Cleaning
    ↓ (Silver Data)
[2] Topic Classification (Zero-shot BART)
    ↓
[3] Stance Detection (Fine-tuned BERT) ⚠️ BOTTLENECK
    ↓
[4] Polarization Measurement (Esteban-Ray Index)
    ↓
[5] Time Series Forecasting (ETS, TFT, Chronos)
    ↓
Analysis & Visualization
```

### Key Components

#### 1. Data Collection & Filtering

- **Reddit:** Pushshift archives, filtered for political subreddits
- **News:** Common Crawl News dataset, US outlets
- **Filters:** Language detection, spam removal, deduplication, minimum length

#### 2. Topic Classification

- **Model:** Facebook BART-large-mnli (zero-shot)
- **Method:** Classify texts into 5 political topics
- **Threshold:** Confidence > 0.5

#### 3. Stance Detection ⚠️

- **Model:** BERT fine-tuned on stance detection datasets
- **Output:** FAVOR, AGAINST, NEUTRAL
- **Challenge:** High uncertainty, needs improvement (see below)

#### 4. Polarization Measurement

- **Index:** Esteban-Ray polarization (from economics literature)
- **Metrics:** ER polarization, bipolarity, balance, extremism
- **Granularity:** Daily per topic per platform

#### 5. Forecasting

- **Models:** ETS (classical), TFT (deep learning), Chronos (foundation model)
- **Horizon:** 14 days ahead
- **Input:** 30+ days historical polarization

## 📁 Repository Structure

```
Data_Experiment/
│
├── 00_project_overview.ipynb       ← START HERE: Complete overview
├── README.md                       ← This file
├── requirements.txt                ← Python dependencies
│
├── data/                           ← Processed datasets
│   ├── 00_raw/                     ← Original data (not in repo)
│   ├── 01_silver/                  ← Filtered & cleaned
│   │   ├── reddit/                 ← reddit_silver.parquet
│   │   └── news/                   ← news_silver.parquet
│   ├── 02_topics/                  ← Topic-classified
│   ├── 03_stance/                  ← Stance-detected
│   ├── 04_polarization/            ← Daily polarization indices
│   └── 05_forecasting/             ← Forecast results
│
├── notebooks/                      ← Analysis notebooks
│   ├── 00_environment_check.ipynb  ← Verify setup
│   ├── 01_data_validation_plan.ipynb
│   │
│   ├── reddit/                     ← Reddit-specific processing
│   │   ├── 10_reddit_download_sep_oct_2016.ipynb
│   │   ├── 11_reddit_extract_filter_silver.ipynb
│   │   ├── 12_reddit_thread_pseudodocs_gold.ipynb
│   │   └── 13_reddit_dataset_qc_report.ipynb
│   │
│   ├── news/                       ← News-specific processing
│   │   ├── 20_news_hf_stream_sep_oct_2016.ipynb
│   │   ├── 21_news_filter_dedup_sample_silver.ipynb
│   │   └── 22_news_dataset_qc_report.ipynb
│   │
│   ├── forecasting/                ← Time series forecasting
│   │   ├── 30_ets_forecasting.ipynb         ← Exponential Smoothing
│   │   ├── 31_tft_forecasting.ipynb         ← Temporal Fusion Transformer
│   │   └── 32_chronos_forecasting.ipynb     ← Foundation model (zero-shot)
│   │
│   └── shared/                     ← Cross-platform analysis
│       ├── 90_shared_schema_checks.ipynb
│       └── 91_shared_determinism_checks.ipynb
│
├── src/                            ← Reusable Python modules
│   └── thesis_pipeline/
│       ├── io/                     ← File I/O utilities
│       ├── news/                   ← News processing
│       ├── reddit/                 ← Reddit processing
│       └── qc/                     ← Quality control
│
├── reports/                        ← Generated reports
│   └── data_validation/
│       └── 2016-09_2016-10/
│           ├── reddit/
│           └── news/
│
└── artefacts/                      ← Metadata & configs
    └── run_metadata/
```

## 🔬 Key Findings

### Data Statistics

| Metric          | Reddit               | News                 |
| --------------- | -------------------- | -------------------- |
| **Total Texts** | ~500K comments       | ~50K articles        |
| **With Topics** | ~250K (50%)          | ~25K (50%)           |
| **With Stance** | ~100K (40%)          | ~10K (40%)           |
| **Date Range**  | Sep 1 - Oct 31, 2016 | Sep 1 - Oct 31, 2016 |

### Polarization Patterns

- **Donald Trump:** Highest polarization on both platforms (~0.08-0.12)
- **Climate Change:** Moderate polarization, trending up (~0.04-0.06)
- **Reddit > News:** Reddit consistently more polarized
- **Temporal Spikes:** Debates and scandals cause polarization surges

### Forecasting Results

| Model       | MAE   | RMSE  | sMAPE | Notes                       |
| ----------- | ----- | ----- | ----- | --------------------------- |
| **ETS**     | 0.024 | 0.029 | 65%   | Baseline, simple            |
| **TFT**     | 0.019 | 0.023 | 52%   | ⚠️ Overfits (straight line) |
| **Chronos** | 0.026 | 0.031 | 72%   | Zero-shot, no training      |

**Critical Issue:** TFT has best metrics but doesn't learn temporal patterns—only fits mean due to limited training data (47 days insufficient).

## ⚠️ Known Issues & Limitations

### 1. Stance Detection (CRITICAL BOTTLENECK)

**Problem:** Current stance model has high uncertainty

- 60-70% classified as NEUTRAL (may indicate model confusion)
- Trained on Twitter data, not political text
- Topic ambiguity (e.g., "Trump" = person or policies?)

**Impact:** Incorrect stance → Wrong polarization → Meaningless forecasting

**What Needs Fixing:**

- [ ] Human annotation of 1000+ political texts
- [ ] Fine-tune on annotated political corpus
- [ ] Add confidence thresholding
- [ ] Validate against ground truth

### 2. Limited Training Data

**Problem:** Only 2 months of data (Sep-Oct 2016)

- TFT requires 6+ months to learn patterns
- Currently defaults to mean prediction
- Can't capture long-term trends

**Solutions:**

- Extend data collection to full election cycle
- Add external features (polls, events)
- Use simpler models for limited data

### 3. News Data Sparsity

**Problem:** Daily news articles insufficient for many topics

- Climate change: Only ~10 articles/day
- Vaccination: <5 articles/day
- Can't compute reliable daily polarization

**Solutions:**

- Aggregate to weekly granularity
- Focus analysis on Reddit (denser data)
- Or expand news sources

## 🎯 Future Directions

### Option A: Improve Stance Pipeline (Recommended)

**Focus:** Fix stance detection to make polarization reliable

**Tasks:**

1. Create annotated stance dataset (political texts)
2. Fine-tune BERT/RoBERTa on annotations
3. Validate polarization indices
4. Extend to more topics/periods

**Pros:** Builds on existing work, clear path
**Cons:** Labor-intensive annotation

### Option B: Network-Based Polarization (Alternative)

**Focus:** Measure polarization from interaction networks, not stance

**Approach:**

- Build user-user networks from Reddit replies
- Detect communities (Louvain algorithm)
- Measure polarization as modularity (community separation)

**Advantages:**

- **No stance detection needed** (uses behavioral data)
- More robust (network structure less noisy)
- Captures echo chambers directly

**Data Requirements:**

- ✅ Reddit has reply chains (`parent_id`)
- ❌ News lacks interaction data

**Feasibility:** High—Reddit data already structured for this

### Option C: Hybrid (Most Ambitious)

Combine stance-based polarization + network analysis for richest insights

## 📚 Dependencies

Key libraries:

- **Data:** `pandas`, `pyarrow`, `datasets`
- **NLP:** `transformers`, `torch`, `spacy`
- **Forecasting:** `statsmodels`, `pytorch-forecasting`, `chronos`
- **Viz:** `matplotlib`, `seaborn`, `plotly`

See `requirements.txt` for complete list.

