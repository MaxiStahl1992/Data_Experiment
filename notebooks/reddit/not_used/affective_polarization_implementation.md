# Affective Polarization Implementation Summary

## Overview

This document summarizes the shift from stance-based to affective-polarization-based measurement of political polarization, following your research plan.

---

## Why Move from Stance to Affective Polarization?

### Problems with Stance Detection

1. **Conceptual mismatch:** Cannot take "stance" toward broad topic categories like "Infrastructure"
2. **Forced claims:** Converting topics to claims ("Government should invest in infrastructure") felt artificial
3. **Not capturing hostility:** Stance measures agreement/disagreement, not emotional negativity

### Advantages of Affective Polarization

1. **Theoretically grounded:** Aligns with contemporary political science research
2. **Captures hostility:** Focuses on negative affect toward political out-groups (what polarization IS)
3. **Ordinal scale:** Measures escalation from incivility → intolerance → violence
4. **Topic-aware:** Uses existing topic assignments to identify political targets
5. **More relevant:** Measures delegitimization and dehumanization, key to democratic health

---

## Implementation Structure

### Files Created

1. **`notebooks/reddit/18_sentiment_detetction.ipynb`**

   - Main analysis notebook (typo in filename - "detetction" vs "detection")
   - 9 sections: Setup → Codebook → Annotation → Modeling → Analysis
   - Interactive annotation widget
   - Model comparison framework
   - Full dataset processing pipeline

2. **`notebooks/reddit/affective_polarization_codebook.md`**
   - Formal 4-level annotation guide
   - Decision tree for consistent coding
   - 30+ examples with rationales
   - Quality control guidelines
   - Based on PHOS framework and contemporary research

### Pipeline Stages

```
Topic-Enhanced Data (existing)
         ↓
Create Test Set (stratified by topic)
         ↓
Manual Annotation (100-200 samples, 4-level scale)
         ↓
Model Comparison (hate speech, sentiment, emotion, NLI)
         ↓
Select Best Model (Macro-F1 + Quadratic Kappa)
         ↓
Apply to Full Dataset (~700k text units)
         ↓
Generate Polarization Scores
         ↓
Validation & Analysis
```

---

## The 4-Level Scale

| Level | Name        | Definition                               | Typical % |
| ----- | ----------- | ---------------------------------------- | --------- |
| **0** | None        | No political target or neutral           | 40-60%    |
| **1** | Adversarial | Insults, ridicule (legitimate opponents) | 30-40%    |
| **2** | Intolerant  | Enemies, traitors, exclusionary          | 10-20%    |
| **3** | Belligerent | Violence, harm, elimination              | 2-5%      |

### Key Innovation

- **Level 0** vs **1-3**: Separates neutral from polarized
- **1** vs **2**: Separates incivility from delegitimization (critical distinction!)
- **2** vs **3**: Separates intolerance from violence advocacy

This captures the escalation ladder that matters for democratic discourse.

---

## Annotation Decision Tree

```
1. Is there a political target?
   (party, politician, voters, ideological group)
   NO → Label 0
   YES ↓

2. Is there affective evaluation?
   (emotional, not just policy discussion)
   NO → Label 0
   YES ↓

3. How hostile?
   - Insults, mockery → 1
   - Enemies, threats, exclusion → 2
   - Violence, harm → 3
```

---

## Current Status

### ✅ Completed

- [x] Notebook structure created (9 sections)
- [x] Formal codebook written (8 pages, 30+ examples)
- [x] Test set creation logic (stratified by topic)
- [x] Interactive annotation widget
- [x] Evaluation framework (Macro-F1, Quadratic Kappa)
- [x] Model comparison scaffold

### ⏳ In Progress

- [ ] **Annotation** (need 100-200 samples)
  - Use interactive widget in notebook
  - Or export to CSV for external annotation
  - Target: 3-5 hours of annotation work

### 📋 To Do (After Annotation)

- [ ] Implement model wrappers for 4 approaches:
  1. Hate speech model (Facebook RoBERTa)
  2. Sentiment model (Twitter RoBERTa)
  3. Emotion model (DistilRoBERTa)
  4. Zero-shot NLI (DeBERTa)
- [ ] Run model comparison
- [ ] Select best model based on Macro-F1
- [ ] Fine-tune on annotated data (optional)
- [ ] Apply to full Reddit dataset
- [ ] Validate with manual sample
- [ ] Generate polarization scores

---

## How Topics Are Used

### Critical Insight

Topics help identify **political targets** but aren't themselves polarization indicators.

```
Text: "MAGA idiots keep falling for obvious lies"
Topic: "Presidential Election" (from earlier pipeline)
  ↓
Political target: MAGA supporters ✓
Affective evaluation: "idiots", "falling for lies" ✓
Hostility level: Insults, ridicule → Level 1 (Adversarial)
```

The topic tells us **what political issue**, affective polarization tells us **how hostile toward whom**.

---

## Models to Compare

### 1. Hate Speech Model

- **Model:** `facebook/roberta-hate-speech-dynabench-r4-target`
- **Rationale:** Hate speech overlaps with Levels 2-3 (intolerance/belligerence)
- **Mapping:** Hate → 2-3, Non-hate → 0-1

### 2. Sentiment Model

- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Rationale:** Negative sentiment indicates affective polarization
- **Mapping:** Negative + political → 1-3, Positive/Neutral → 0

### 3. Emotion Model

- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Rationale:** Anger, disgust indicate hostility
- **Mapping:** Anger/Disgust → 1-2, Fear → 2, Neutral → 0

### 4. Zero-Shot NLI

- **Model:** `microsoft/deberta-v3-base-mnli-fever-anli-ling-wanli`
- **Rationale:** Test explicit hypotheses for each level
- **Hypotheses:**
  - "This text is neutral about political groups" → 0
  - "This text insults political opponents" → 1
  - "This text treats opponents as enemies" → 2
  - "This text supports violence against opponents" → 3

**Expected winner:** Likely the NLI approach or emotion model, as they can distinguish fine-grained hostility levels.

---

## Metrics for Evaluation

### Primary Metric: Macro-F1

- Averages F1 across all 4 classes
- Handles class imbalance (Levels 2-3 are rare)
- Standard for multi-class problems

### Secondary Metric: Quadratic Kappa

- Weights disagreements by distance (1→2 better than 1→3)
- Respects ordinal nature of scale
- Target: κ > 0.70

### Per-Class Metrics

- Especially important for rare classes (2, 3)
- Need good recall for Level 3 (belligerence) to capture extreme cases
- Need good precision for Level 0 to avoid false positives

---

## From Labels to Polarization Scores

### Individual Text Score

```python
score = predicted_label / 3  # Normalized 0-1
# OR
score = sum(prob_i * level_i for i, level_i in enumerate([0,1,2,3]))
```

### Aggregate Scores

**By Topic:**

```python
topic_polarization = comments.groupby('topic_label')['polarization_score'].mean()
# "Presidential Election" → 0.45 (high)
# "Infrastructure Policy" → 0.22 (moderate)
```

**By Time:**

```python
daily_polarization = comments.groupby('date')['polarization_score'].mean()
# Track escalation toward election day
```

**By Subreddit:**

```python
subreddit_polarization = comments.groupby('subreddit')['polarization_score'].mean()
# r/The_Donald → 0.62 (very high)
# r/NeutralPolitics → 0.15 (low)
```

---

## Integration with Existing Pipeline

### Data Flow

```
00_raw/
  ├── news/
  └── reddit/
         ↓
01_silver/reddit/
  (filtered, deduplicated)
         ↓
02_topics/reddit/embeddings/
  ├── comments_expanded_with_topics.parquet
  └── submissions_expanded_with_topics.parquet
         ↓
04_affective_polarization/reddit/  ← NEW
  ├── test_set_annotations.parquet
  ├── comments_with_polarization.parquet
  └── submissions_with_polarization.parquet
         ↓
reports/affective_polarization/
  ├── model_comparison.png
  ├── polarization_by_topic.png
  └── temporal_trends.png
```

### Schema Addition

Existing columns remain; we add:

- `affective_polarization_label`: 0-3 (integer)
- `affective_polarization_score`: 0-1 (float)
- `polarization_confidence`: Model confidence
- `polarization_probs`: [prob_0, prob_1, prob_2, prob_3]

---

## Next Immediate Steps

### 1. Start Annotation (NOW)

```python
# In notebook cell
annotate_interactive(test_set, start_idx=0)
```

**Tips:**

- Annotate in 30-60 minute sessions (content is draining)
- Use keyboard if possible (faster than clicking)
- Track cases you're uncertain about
- Aim for 20-30 samples per session
- Save frequently

### 2. Quality Check (After 50 samples)

```python
# Check distribution
test_set['affective_polarization_label'].value_counts(normalize=True)

# Expected (roughly):
# 0: 50%
# 1: 35%
# 2: 12%
# 3: 3%
```

If very different, review your application of the codebook.

### 3. Double-Code Sample (Optional but Recommended)

- Have second coder annotate 20-30 of your samples
- Calculate inter-rater reliability (Quadratic Kappa)
- Target: κ > 0.70
- Review disagreements to clarify codebook

---

## Research Contribution

### Methodological Innovation

1. **Topic-aware polarization measurement**

   - Uses topic modeling to identify political targets
   - Measures hostility toward specific issues
   - More granular than generic toxicity detection

2. **Ordinal scale captures escalation**

   - Not just "polarized vs not"
   - Tracks progression: incivility → intolerance → violence
   - Critical for understanding democratic backsliding

3. **Reproducible framework**
   - Formal codebook enables replication
   - Multiple models compared systematically
   - Validation protocol included

### Theoretical Alignment

- Follows "put the affect into affective polarization" movement
- Aligns with PHOS framework (adversarial vs antagonistic)
- Captures delegitimization (Levitsky & Ziblatt's warning signs)
- Measures dehumanization (moral disengagement indicator)

---

## Troubleshooting

### "Annotation is too hard / uncertain"

**Solution:** Refer to decision tree in codebook

- Start with: Is there a political target?
- Then: Is there emotion?
- Finally: How hostile?
- When uncertain: Default to lower level (conservative)

### "Model comparison fails"

**Solution:** Likely need more annotations

- 4-class problem needs more data than binary
- Minimum 100, target 150-200
- Ensure all 4 classes have ≥10 examples

### "Level 3 is too rare"

**Solution:** This is expected and OK

- Level 3 is naturally rare (~2-5%)
- Can over-sample Level 3 candidates using keywords
- Or collapse 2+3 for modeling, then separate with thresholds

---

## Timeline to Completion

| Stage                | Time          | When             |
| -------------------- | ------------- | ---------------- |
| Annotation           | 3-5 hrs       | NOW → 1-2 days   |
| Model implementation | 2-3 hrs       | After annotation |
| Comparison           | 1 hr          | Same day         |
| Full processing      | 2-4 hrs       | Compute time     |
| Validation           | 2-3 hrs       | Next day         |
| **Total**            | **10-17 hrs** | **2-4 days**     |

**Critical path:** Annotation is the bottleneck. Everything else can proceed quickly once you have 100-150 annotated samples.

---

## Questions?

- **Codebook ambiguity:** Review examples in codebook, add new examples as needed
- **Model implementation:** Can adapt from stance detection code (similar structure)
- **Evaluation strategy:** Macro-F1 is standard, Quadratic Kappa is best for ordinal
- **Next steps:** Start annotating! Everything else depends on having labeled data.

---

**Last Updated:** December 20, 2025
**Status:** Notebook + codebook ready, awaiting annotation
**Next Action:** Begin annotation using interactive widget
