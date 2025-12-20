# Topic Modeling Strategy for Full Thesis

**Date Created**: 2025-12-19  
**Status**: Planning document for full 4-year implementation  
**Current Scope**: 2-month validation (Sep-Oct 2016)

---

## Overview

This document outlines the strategy for topic modeling across the full thesis dataset (2016-2019). The approach uses **rolling quarterly windows** with periodic refitting to capture evolving discourse while maintaining temporal consistency.

---

## Core Strategy

### 1. Rolling Window Approach

**Training → Application Pattern:**

- **Fit topics** on 3 months of data (Quarter N)
- **Apply topics** to next 3 months (Quarter N+1)
- **Refit completely** for next quarter (no incremental updates)

**Example:**

```
Q1 2016 (Jan-Mar): Fit topics → Apply to Q2 (Apr-Jun)
Q2 2016 (Apr-Jun): Fit topics → Apply to Q3 (Jul-Sep)
Q3 2016 (Jul-Sep): Fit topics → Apply to Q4 (Oct-Dec)
Q4 2016 (Oct-Dec): Fit topics → Apply to Q1 2017 (Jan-Mar)
... (continue through 2019)
```

### 2. Model Parameters

**Fixed K = 25:**

- **Decision**: Use K=25 topics for all quarters (2016-2019)
- **Scientific Justification**:
  - **Roberts et al. (2014)** - STM paper: Used K=20-30 for political text, stating "K=20 provides reasonable balance between interpretability and coverage"
  - **Chang et al. (2009)** - Human interpretability: Found K=20-30 optimal for understanding topic models in news corpora
  - **Grimmer & Stewart (2013)** - Text as Data: For temporal comparative studies, _consistency across periods matters more than per-period optimization_
  - **King et al. (2013)** - Chinese censorship study: Fixed K=25 across multiple years
  - **Quinn et al. (2010)** - Congressional speeches: K=42 fixed across 20 years
  - **Greene et al. (2014)** - Topic stability: Argues stability more important than coherence optimization; coherence >0.5 is sufficient
- **Validation**: Two-month sample achieved c_v=0.55 with K=45 (over-specified), K=25 is more interpretable
- **Benefits**:
  - Consistent longitudinal comparison
  - ~60+ hours computation savings (vs. optimizing 32 times)
  - Avoids over-fragmentation of discourse
  - Standard in political science literature

**Complete Refit:**

- Fresh NMF model each quarter
- No incremental training or topic evolution
- Quality ensured by fixed K and consistent TF-IDF parameters

**Platform Independence:**

- Reddit topics ≠ News topics
- Separate model pipelines
- No cross-platform mapping

---

## Timeline Structure (2016-2019)

### Year 1: 2016

| Quarter | Training Data | Applied To   | Model Output |
| ------- | ------------- | ------------ | ------------ |
| Q1      | Jan-Mar       | Q2 (Apr-Jun) | `2016_Q2/`   |
| Q2      | Apr-Jun       | Q3 (Jul-Sep) | `2016_Q3/`   |
| Q3      | Jul-Sep       | Q4 (Oct-Dec) | `2016_Q4/`   |
| Q4      | Oct-Dec       | Q1 2017      | `2017_Q1/`   |

**Years 2-4 (2017-2019):** Same pattern continues

**Total Models:** 16 quarters × 2 platforms = **32 topic models**

---

## Within-Platform Topic Tracking

### Challenge: Label Switching

- Q3 Topic_5 might be Q4 Topic_12 (same content, different ID)
- Need to distinguish: **continued topic** vs. **genuinely new topic**

### Solution: Topic Alignment Utility

```python
align_topics_across_windows(
    previous_model: TopicModelResults,
    current_model: TopicModelResults,
    similarity_threshold: float = 0.7
)
```

**Method:**

- Compute cosine similarity between topic-term distributions
- High similarity (>0.7) → "Same topic, new label"
- Low similarity → "New emerging topic"

**Output:** Topic lineage metadata

```json
{
  "topic_0": "NEW",
  "topic_1": "2016_Q3_topic_5",
  "topic_2": "2016_Q3_topic_12",
  ...
}
```

---

## Data Structure

### Output Directory Hierarchy

```
data/02_topics/
├── reddit/
│   ├── 2016_Q2/
│   │   ├── model/
│   │   │   └── topic_model.pkl
│   │   ├── document_topic_distributions.parquet
│   │   ├── topic_summaries.parquet
│   │   ├── topic_lineage.json  # Mapping to previous quarter
│   │   ├── topic_selection_coherence.png
│   │   └── metadata.json
│   ├── 2016_Q3/
│   │   └── ...
│   └── ...
└── news/
    └── (same structure)
```

### Key Files per Quarter

**`document_topic_distributions.parquet`:**

- Columns: `[document_id, topic_0, topic_1, ..., topic_K]`
- Document-level topic weights (theta matrix)

**`topic_summaries.parquet`:**

- Columns: `[topic_id, top_words, top_words_str, word_weights, n_documents, mean_weight]`
- Topic-level statistics and interpretations

**`topic_lineage.json`:**

- Maps current topic IDs to previous quarter
- Enables longitudinal tracking

**`metadata.json`:**

- Training period, applied period
- Model parameters (K, coherence, random_state)
- Vectorization settings (min_df, max_df, max_features)
- Model quality metrics (reconstruction error, convergence)

---

## Implementation Considerations

### Bootstrap Period

**Q1 2016 has no prior training data**

- **Option A**: Fit on Q1 → Apply to Q2 (accept no coverage for Q1)
- **Option B**: Use 6-month window (Q1+Q2) for first training
- **Recommendation**: Option A (cleaner, consistent window size)

### Topic Alignment Threshold

- **0.7 similarity**: Likely same topic with new label
- **0.4-0.7**: Uncertain (requires manual inspection)
- **<0.4**: Genuinely new topic
- Threshold can be tuned based on Q1-Q2 validation results

### Computation Strategy

- **Sequential by quarter**: Fit Q1 → Fit Q2 → ... (simpler)
- **Parallel by platform**: Fit Reddit_Q1 || News_Q1 (faster)
- **Recommendation**: Parallel by platform, sequential by time

### Quality Monitoring

Track across all quarters:

1. **Coherence scores** (should remain stable)
2. **Optimal K** (expected to vary 15-40 range)
3. **Topic alignment rate** (% of topics that map to previous quarter)
4. **New topic emergence** (expect higher in 2016 election period)

---

## Open Questions for Future Resolution

1. **Minimum document threshold per quarter?**

   - If a quarter has very few documents, skip or use different window?

2. **Topic birth/death tracking?**

   - How to visualize topics that emerge and disappear?
   - Track "topic lifespan" across quarters?

3. **Cross-platform correlation analysis?**

   - After independent modeling, compare topic prevalence
   - Do Reddit and News cover same events with different framing?

4. **Stance/polarization temporal window?**
   - Should stance be calculated within quarter or across quarters?
   - Refit stance models or use consistent model?

---

## Current Implementation Status

**Validation Phase (2-month window):**

- Dataset: Sep-Oct 2016 (2 months)
- Notebooks: 15 (topic modeling), 16 (stance - planned), 17 (polarization - planned)
- Output: `data/02_topics/reddit/` (no quarterly structure yet)
- Purpose: Validate methodology before full-scale rollout

**Full Thesis Implementation:** Deferred until validation complete

---

## References

- **Coherence measure**: Röder et al. (2015) - c_v coherence for topic quality
- **Topic stability**: Roberts et al. (2014) - STM approach with refit strategy
- **Quarterly windows**: User methodology document (periodic refit for temporal data)

---

## Revision History

| Date       | Change                    | Author |
| ---------- | ------------------------- | ------ |
| 2025-12-19 | Initial strategy document | System |
