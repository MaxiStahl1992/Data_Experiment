# Notebook 18: Reddit Stance Detection - Workflow Guide

## Overview

**Purpose:** Systematically compare stance detection models and apply the best one to full Reddit dataset.

**Input Files:**

- `data/02_topics/reddit/embeddings/comments_expanded_with_topics.parquet`
- `data/02_topics/reddit/embeddings/submissions_expanded_with_topics.parquet`

**Output Files:**

- `data/03_stance/reddit/comments_with_stance.parquet`
- `data/03_stance/reddit/submissions_with_stance.parquet`
- Model comparison metrics and visualizations

---

## Workflow Steps

### 1. Setup & Data Loading (Cells 1-2)

- Load topic-enhanced comments and submissions
- Verify data structure and topic distribution

### 2. Create Test Set (Cell 3)

- Sample ~100 texts stratified by topic
- Mix of comments and submissions
- Used for model comparison

### 3. Define Models to Compare (Cell 4)

Four models tested:

1. **RoBERTa-MNLI** - Baseline (stable, well-tested)
2. **DeBERTa-v3-Multi** - Stronger multi-task NLI
3. **DeBERTa-v3-CrossEncoder** - Cross-encoder architecture
4. **BART-MNLI** - Zero-shot baseline

### 4. Manual Annotation (Cells 5-6)

**CRITICAL:** Must annotate test set with ground truth labels

**Options:**

- **Interactive widget:** Run cell 6 for in-notebook annotation
- **Export/import:** Save test set to CSV, annotate in Excel, reload

**Labels:**

- `FAVOUR` - Text supports/agrees with topic
- `AGAINST` - Text opposes/disagrees with topic
- `NEUTRAL` - No clear stance or balanced view

**Minimum:** ~50 samples for reasonable evaluation
**Recommended:** ~100 samples for reliable metrics

### 5. Run Model Comparison (Cell 7)

- Automatically tests all 4 models on annotated test set
- Calculates metrics for each model
- Takes ~10-30 minutes depending on test set size

**Metrics Calculated:**

- Accuracy
- Macro-F1 (primary criterion)
- Per-class F1 (FAVOUR/AGAINST/NEUTRAL)
- Cohen's Kappa (reliability measure)
- Confusion matrices

### 6. Compare Performance (Cell 8)

- View metrics summary table
- Visualize comparison plots:
  - Overall metrics comparison
  - Per-class F1 scores
  - Best model confusion matrix
  - Ranking summary

### 7. Select Best Model (Cell 9)

- Automatic selection based on Macro-F1
- Shows selected model and performance
- Saves selection to JSON

### 8-10. Full Dataset Processing

**Cell 9:** Comments stance detection (~1-3 hours)
**Cell 10:** Submissions stance detection (~30-60 minutes)

Applies best model to entire datasets with:

- Batch processing (efficient)
- Progress tracking
- Confidence scores saved

### 11. Save Results (Cell 11)

Saves to `data/03_stance/reddit/`:

- `comments_with_stance.parquet`
- `submissions_with_stance.parquet`
- `stance_detection_metadata.json`

### 12-13. Quality Analysis

- Stance distribution visualizations
- Confidence analysis
- Topic-level stance patterns
- Summary statistics

---

## Expected Outputs

### 1. Model Comparison Results

```
data/03_stance/reddit/
├── model_comparison_metrics.csv       # Metrics for all models
├── model_comparison.png               # Visualization
├── model_selection.json               # Best model info
└── test_set_annotations.parquet       # Annotated test set
```

### 2. Stance-Enhanced Datasets

```
data/03_stance/reddit/
├── comments_with_stance.parquet       # Comments + stance columns
├── submissions_with_stance.parquet    # Submissions + stance columns
└── stance_detection_metadata.json     # Metadata and stats
```

### 3. Quality Analysis

```
data/03_stance/reddit/
└── stance_quality_analysis.png        # Distribution visualizations
```

---

## New Columns Added

Both output datasets will have these additional columns:

| Column                | Type  | Description                               |
| --------------------- | ----- | ----------------------------------------- |
| `stance`              | str   | Predicted stance (FAVOUR/AGAINST/NEUTRAL) |
| `stance_confidence`   | float | Model confidence (0-1)                    |
| `stance_prob_favour`  | float | Probability of FAVOUR                     |
| `stance_prob_against` | float | Probability of AGAINST                    |
| `stance_prob_neutral` | float | Probability of NEUTRAL                    |

---

## Decision Criteria

**Model selection prioritizes:**

1. **Macro-F1** (primary) - Balanced performance across all stances
2. **Per-class F1** - Individual stance detection quality
3. **Cohen's Kappa** - Reliability/consistency

**Acceptable baseline:**

- Macro-F1 ≥ 0.60
- Cohen's Kappa ≥ 0.70

**If insufficient:** Can fine-tune on VAST dataset (~60k stance examples)

---

## Reusability for News

The same framework will be used for news stance detection:

1. Load news topic-enhanced data
2. Use same test set annotation approach
3. Potentially use same best model or re-compare
4. Apply to news corpus
5. Save to `data/03_stance/news/`

**Key advantage:** All utilities are platform-agnostic and reusable!

---

## Troubleshooting

### Issue: Models fail to load

**Solution:** Check GPU/MPS memory, try smaller batch size

### Issue: All predictions are NEUTRAL

**Solution:** Check confidence threshold (default 0.04 is intentionally low)

### Issue: Low F1 scores (<0.5)

**Solutions:**

1. Increase test set size (more annotations)
2. Check annotation quality (inter-annotator agreement)
3. Try fine-tuning on VAST dataset
4. Adjust confidence threshold

### Issue: Out of memory during full dataset processing

**Solutions:**

1. Reduce batch size (from 32 to 16 or 8)
2. Process in chunks with multiple runs
3. Use CPU instead of GPU (slower but more memory)

---

## Time Estimates

| Step                                | Time          | Notes                    |
| ----------------------------------- | ------------- | ------------------------ |
| Manual annotation (100 samples)     | 30-60 min     | Depends on familiarity   |
| Model comparison (4 models)         | 10-30 min     | Depends on test set size |
| Comments processing (~550k rows)    | 1-3 hours     | Batch size 32            |
| Submissions processing (~150k rows) | 30-60 min     | Batch size 32            |
| **Total**                           | **2-5 hours** | Plus annotation time     |

---

## Next Steps After Completion

1. **Validate quality:** Manual inspection of ~50-100 samples per stance
2. **Threshold optimization:** Tune confidence threshold if needed
3. **Apply to news:** Run same pipeline on news data
4. **Polarization analysis:** Use stance + topics for Notebook 19
5. **Temporal analysis:** Track stance evolution over time

---

## References

**Utilities Used:**

- `thesis_pipeline.stance.model.ImprovedNLIStanceModel` - NLI-based stance detection
- `thesis_pipeline.stance.comparison.StanceModelComparison` - Model comparison framework
- `thesis_pipeline.stance.comparison.evaluate_stance_model` - Metrics calculation

**Model Sources:**

- RoBERTa-MNLI: `roberta-large-mnli`
- DeBERTa-v3-Multi: `microsoft/deberta-v3-base-mnli-fever-anli-ling-wanli`
- DeBERTa-v3-CrossEncoder: `cross-encoder/nli-deberta-v3-base`
- BART-MNLI: `facebook/bart-large-mnli`
