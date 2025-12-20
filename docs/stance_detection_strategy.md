# Stance Detection Strategy

**Date Created**: 2025-12-20  
**Last Updated**: 2025-12-20  
**Status**: Baseline implementation (validation phase)  
**Current Scope**: Sep-Oct 2016 (433k Reddit threads)

---

## Overview

This document outlines the strategy for stance detection across comments and submissions relative to their assigned political topics. Stance detection determines whether a text expresses **FAVOUR**, **AGAINST**, or **NEUTRAL** sentiment toward each assigned topic.

---

## 1. Stance Task Definition

For stance unit u (comment or submission) and topic target τ_k:

$$
\text{stance}(u,\tau_k)\in\{\text{FAVOUR},\text{AGAINST},\text{NEUTRAL}\}
$$

**Definitions:**

- **FAVOUR**: Text explicitly or implicitly supports, agrees with, or promotes the topic
- **AGAINST**: Text explicitly or implicitly opposes, criticizes, or argues against the topic
- **NEUTRAL**: Text discusses the topic without clear support or opposition, OR insufficient confidence to assign polarized stance

This follows the standard stance detection formulation (text + target → favour/against/neutral) established in SemEval-2016 Task 6.

**Citation:**

> Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., & Cherry, C. (2016). SemEval-2016 Task 6: Detecting Stance in Tweets. In _Proceedings of SemEval_ (pp. 31-41).

---

## 2. Current Approach: Zero-Shot NLI-Based Stance Detection

### 2.1 Method

**Model**: DeBERTa-v3-base-mnli (or similar NLI transformer)

**Approach**: Map Natural Language Inference (entailment/contradiction) to stance labels

**Process**:

1. For each (text, topic) pair, formulate two NLI hypothesis tests:
   - Hypothesis 1: "This text supports [topic]"
   - Hypothesis 2: "This text opposes [topic]"
2. Get entailment probabilities for each hypothesis
3. Map to stance:
   - High entailment for H1 → **FAVOUR**
   - High entailment for H2 → **AGAINST**
   - Low confidence for both → **NEUTRAL**

**Rationale**:

- Zero-shot approach requires no task-specific training data
- NLI models are strong at semantic reasoning
- Validates feasibility before investing in fine-tuned models
- Fast enough for validation (compared to generative models)

**Scientific Basis:**

- Allaway & McKeown (2020): Zero-shot stance detection using NLI models
- Entailment as proxy for stance has been validated in multiple studies

**Citations:**

> Allaway, E., & McKeown, K. (2020). Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations. In _Proceedings of EMNLP_ (pp. 8913-8931).

### 2.2 Target Representation τ_k

**Critical Adaptation from Original Plan:**

**Original Plan (STM-based):**

- Use top 10-15 STM terms as target representation
- Example: τ_k = "healthcare insurance ACA reform costs public medical"
- Rationale: Works for discovered, term-based topics

**Our Implementation (Supervised CAP-based):**

- Use topic label + description from predefined taxonomy
- Example: τ_k = "Healthcare Policy: Health insurance, ACA/Obamacare, medical costs, public health"
- Rationale: We use supervised topics, not discovered STM terms

**Why This Adaptation?**

1. **Consistency**: Our topics are predefined (CAP taxonomy), not discovered via STM
2. **Interpretability**: Topic labels are more semantically rich than term lists
3. **Validity**: CAP descriptions validated across political science literature
4. **NLI Compatibility**: NLI models trained on natural language, not term bags

**Implementation:**

```python
target = f"{topic_label}: {topic_description}"
hypothesis_favour = f"This text supports {topic_label.lower()}"
hypothesis_against = f"This text opposes {topic_label.lower()}"
```

### 2.3 Confidence Filtering

**Threshold-Based Rejection:**

Stance model outputs probabilities:

$$
\mathbf{p}(u,\tau_k)=(p_{\text{FAVOUR}}, p_{\text{AGAINST}}, p_{\text{NEUTRAL}})
$$

A confidence filter is applied:

- Assign **FAVOUR** if favour_score > confidence_threshold AND favour_score > against_score
- Assign **AGAINST** if against_score > confidence_threshold AND against_score > favour_score
- Otherwise assign **NEUTRAL** (includes low-confidence cases)

**Current Threshold**: c = 0.5 (tunable based on validation)

**Rationale**:

- Avoids assigning polarized stance when model is uncertain
- NEUTRAL becomes "catch-all" for unclear cases (defensible choice)
- Threshold can be optimized on held-out validation set

**Scientific Basis:**

- Selective classification / abstention frameworks (Geifman & El-Yaniv, 2017)
- Confidence-based rejection improves precision at cost of recall

**Citations:**

> Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. In _NIPS_ (pp. 4878-4887).

---

## 3. Multi-Label Structure

**Key Design Decision:** Each text can be assigned to **multiple topics** (multi-label), and therefore can have **different stances** for different topics.

**Example:**

- Text: "Trump's healthcare cuts will hurt veterans"
- Topics assigned: [Healthcare Policy, Presidential Politics, Defense & Military]
- Stances:
  - Healthcare Policy: **AGAINST** (opposes healthcare cuts)
  - Presidential Politics: **AGAINST** (criticism of Trump)
  - Defense & Military: **FAVOUR** (supports veterans)

**Implementation:**

- Expand dataset: 1 text with 3 topics → 3 rows
- Each row contains: (text, topic, stance, confidence)
- Same text can appear multiple times with different stances

**Scientific Justification:**

- Political discourse is naturally multi-faceted (Read et al., 2011)
- Single stance per text would lose information
- Matches our multi-label topic classification approach

---

## 4. Output Structure

### 4.1 Submission-Level Dataset

**File**: `reddit_submissions_topic_stance.parquet`

**Schema**:

```
submission_id_text (str): Submission ID
submission_id (str): Thread ID (link_id_clean)
created_utc (float): Unix timestamp
text (str): Submission title
topic_id (int): Topic ID (0-19)
topic_label (str): Topic name
stance (str): FAVOUR/AGAINST/NEUTRAL
stance_confidence (float): Confidence score (0-1)
favour_prob (float): Probability of FAVOUR
against_prob (float): Probability of AGAINST
neutral_prob (float): Probability of NEUTRAL
```

**Multi-Label Example:**

- Submission X assigned 3 topics → 3 rows in dataset
- Each row has potentially different stance

### 4.2 Comment-Level Dataset

**File**: `reddit_comments_topic_stance.parquet`

**Schema**:

```
comment_id (str): Comment ID
submission_id (str): Parent submission ID (for thread reference)
created_utc (float): Unix timestamp
text (str): Comment body
topic_id (int): Topic ID (0-19)
topic_label (str): Topic name
stance (str): FAVOUR/AGAINST/NEUTRAL
stance_confidence (float): Confidence score (0-1)
favour_prob (float): Probability of FAVOUR
against_prob (float): Probability of AGAINST
neutral_prob (float): Probability of NEUTRAL
```

**Purpose**:

- Enables thread-level stance analysis (submission + all comments)
- Allows tracking stance evolution within discussions
- Supports polarization metrics (stance disagreement within threads)

---

## 5. Validation & Quality Assurance

### 5.1 Manual Validation Plan

**Stratified Sampling:**

- Sample 50-100 cases per stance (FAVOUR, AGAINST, NEUTRAL)
- Stratify across:
  - Topics (ensure all 20 topics represented)
  - Confidence levels (high vs. medium vs. low)
  - Content type (submissions vs. comments)
  - Time period (Sep vs. Oct 2016)

**Annotation Protocol:**

- Two independent annotators per sample
- Annotation guidelines based on SemEval-2016 Task 6
- Measure inter-annotator agreement (Cohen's κ)
- Report macro-F1 on {FAVOUR, AGAINST, NEUTRAL}

**Quality Metrics:**

- Macro-F1 (equal weight to each class)
- Per-class precision & recall
- Confusion matrix
- Agreement with human annotations

### 5.2 Threshold Optimization

**Approach**: Tune confidence threshold c on validation set

**Objective**: Maximize macro-F1 subject to minimum coverage constraint

- Coverage: % of cases assigned FAVOUR or AGAINST (not NEUTRAL)
- Trade-off: Higher threshold → higher precision, lower coverage

**Expected Range**: c ∈ [0.4, 0.7]

### 5.3 Success Criteria

**Acceptable baseline performance:**

- Macro-F1 ≥ 0.60 (60% accuracy across stances)
- Inter-annotator agreement κ ≥ 0.70 (substantial agreement)
- Coverage ≥ 50% (at least half of texts get polarized stance)

**If baseline insufficient:**

- Switch to Option 2: Fine-tuned VAST model (RoBERTa-base)
- Requires pre-trained VAST weights or fine-tuning on stance datasets
- Expected improvement: +10-15 points macro-F1

---

## 6. Alternative Approach: VAST Fine-Tuned Model (Option 2)

**If zero-shot baseline proves insufficient**, implement fine-tuned stance detection:

### 6.1 Model

**RoBERTa-base** fine-tuned on **VAST** (Variability-Aware Stance Target) dataset

**VAST Dataset:**

- 13,000+ stance-annotated texts
- Open-target stance (generalizes beyond fixed targets)
- Multiple domains (news, social media, debates)

**Citations:**

> Allaway, E., & McKeown, K. (2020). Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations. In _Proceedings of EMNLP_.

### 6.2 Target Representation

**Generalized Topic Representation:**

- Combine topic label + description + keywords
- Example: τ_k = "Healthcare Policy. Topics: health insurance, ACA, Obamacare, medical costs, public health"

**Process:**

- Fine-tune RoBERTa on VAST dataset
- At inference: concatenate [CLS] text [SEP] target [SEP]
- Classify into {FAVOUR, AGAINST, NEUTRAL}

### 6.3 Expected Performance

**Literature benchmarks:**

- VAST model: macro-F1 = 0.65-0.72 on held-out targets
- Fine-tuned RoBERTa: +10-15 points over zero-shot
- Target-aware models outperform target-agnostic by ~8 points

### 6.4 Implementation Requirements

**If needed:**

1. Obtain VAST dataset (publicly available)
2. Find pre-trained VAST checkpoint OR fine-tune RoBERTa
3. Adapt target representation for CAP topics
4. Validate on manual annotation sample
5. Re-run notebook 17 with new model

---

## 7. Computational Considerations

### 7.1 Performance (Zero-Shot Baseline)

**Current Implementation:**

- Model: DeBERTa-v3-base-mnli (~140M parameters)
- Device: Apple Silicon MPS
- Batch size: 16
- Processing speed: ~20-30 texts/second (two forward passes per text)

**Scalability:**

- Sep-Oct 2016 (433k threads, avg 1.27 topics): ~550k (text, topic) pairs
- Estimated time: 5-8 hours (acceptable for validation)
- Full corpus (50M docs, 1.27 topics): ~63M pairs → 600-900 hours
  - Solution: Batch processing overnight/weekend, or use GPU cluster

### 7.2 Memory Optimization

**Strategies:**

- Batch processing: 16 texts at a time (prevents OOM)
- Text truncation: 512 characters (sufficient for stance)
- Streaming: Process in chunks, save incrementally
- Checkpointing: Resume from last saved batch if interrupted

### 7.3 Production Considerations

**For full thesis (2016-2019):**

- Parallelize by month: Process 48 months independently
- Use distributed computing: Multiple GPUs/nodes
- Cache embeddings: If switching to VAST, reuse text embeddings
- Monitor quality: Track stance distribution over time for drift

---

## 8. Integration with Topic Classification

**Pipeline Flow:**

1. **Topic Classification** (notebook 15d):

   - Input: Thread pseudo-documents
   - Output: Multi-label topic assignments (1-5 topics per thread)
   - File: `thread_pseudodocs_with_supervised_topics_multilabel.parquet`

2. **Stance Detection** (notebook 17):

   - Input: Individual comments/submissions + assigned topics
   - Process: Expand to (text, topic) pairs → detect stance
   - Output: Two datasets (submissions, comments) with stance per topic

3. **Polarization Metrics** (notebook 18, planned):
   - Input: Comment-level stance data
   - Process: Aggregate within threads to measure disagreement
   - Metrics: Stance entropy, disagreement rate, polarization index

**Data Flow:**

```
thread_pseudodocs (topics)
    → merge with gold_data (individual texts)
    → expand to (text, topic) pairs
    → stance detection
    → submissions_topic_stance.parquet
    → comments_topic_stance.parquet
    → polarization analysis
```

---

## 9. Scientific Justification for Thesis

### 9.1 Methodological Defense

**Why Zero-Shot NLI (Baseline)?**

1. **Speed**: Faster than fine-tuning for validation
2. **No Training Data**: Avoids annotation burden
3. **Strong Performance**: NLI models excel at semantic reasoning
4. **Defensible**: Validated in Allaway & McKeown (2020)
5. **Baseline**: If insufficient, upgrade to VAST (clear progression)

**Why Not Generative Models (GPT, etc.)?**

1. **Speed**: 10-100x slower than classification
2. **Cost**: API costs for 63M pairs prohibitive
3. **Consistency**: Deterministic classification preferred
4. **Overkill**: Stance is simple 3-class problem, not generation task

**Why Target-Aware (vs. Target-Agnostic)?**

1. **Accuracy**: +8 points F1 in literature (Allaway & McKeown, 2020)
2. **Interpretability**: Stance is inherently relative to target
3. **Multi-Label**: Same text has different stances for different topics

### 9.2 Addressing Potential Criticisms

**_"Isn't NLI a proxy, not real stance?"_**

- Yes, but validated proxy (Allaway & McKeown, 2020)
- Entailment correlates strongly with stance agreement
- Manual validation will measure true stance accuracy
- If insufficient, we upgrade to fine-tuned VAST

**_"Zero-shot might be too weak"_**

- Agree—baseline only, not final method
- Literature suggests 0.60-0.65 F1 achievable
- If validation shows <0.60, switch to VAST (Option 2)
- Clear upgrade path documented

**_"Why not fine-tune on political stance data?"_**

- VAST already includes political texts
- Limited political stance datasets available (SemEval has 5 targets)
- Transfer learning from VAST better than training from scratch
- Option 2 includes VAST if needed

**_"Confidence threshold seems arbitrary"_**

- c=0.5 is starting point, will be tuned on validation set
- Optimization based on macro-F1 maximization
- Literature suggests c ∈ [0.4, 0.7] optimal range
- Threshold selection will be justified empirically

**_"What about annotation subjectivity?"_**

- Acknowledged: Stance is inherently subjective
- Inter-annotator agreement (κ) will be reported
- Multiple annotators reduce individual bias
- Model stance is deterministic (reproducible)
- Subjectivity exists in any stance annotation (human or model)

### 9.3 Contribution Clarity

**Stance detection is infrastructure, not contribution:**

- Thesis contribution: Polarization dynamics, not stance modeling
- Stance is measurement layer (like using a thermometer, not inventing one)
- Validated approach from literature (SemEval, VAST, NLI)
- Quality assurance ensures measurement validity

**Documented Decisions:**

1. Zero-shot NLI baseline (speed, validation)
2. Target representation adapted for supervised topics (CAP vs. STM)
3. Confidence filtering with tunable threshold (c=0.5 initial)
4. Multi-label structure (matches topic classification)
5. Clear upgrade path if baseline insufficient (VAST fine-tuned)

---

## 10. Current Status & Next Steps

### 10.1 Completed

✅ Stance detection notebook (17_reddit_stance_detection.ipynb) created  
✅ Zero-shot NLI approach implemented  
✅ Multi-label structure preserved (one row per text-topic pair)  
✅ Two output datasets created (submissions, comments)  
✅ Confidence filtering applied  
✅ Initial sample inspection performed

### 10.2 Pending

⏳ Manual validation (50-100 samples per stance)  
⏳ Inter-annotator agreement measurement  
⏳ Threshold optimization (tune c on validation set)  
⏳ Quality metrics calculation (macro-F1, precision, recall)  
⏳ Decision: Keep zero-shot OR upgrade to VAST

### 10.3 Future Work (Full Corpus)

📋 Apply stance detection to full 2016-2019 dataset  
📋 Implement VAST model if zero-shot insufficient  
📋 Monitor temporal stability of stance distribution  
📋 Integrate with polarization analysis (notebook 18)

---

## 11. References

### Stance Detection Foundations

- Mohammad et al. (2016). "SemEval-2016 Task 6: Detecting Stance in Tweets." _Proceedings of SemEval_.
- Küçük & Can (2020). "Stance Detection: A Survey." _ACM Computing Surveys_.

### Zero-Shot & NLI Approaches

- Allaway & McKeown (2020). "Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations." _Proceedings of EMNLP_.
- Schiller et al. (2021). "Stance Detection Benchmark: How Robust Is Your Stance Detection?" _KI - Künstliche Intelligenz_.

### VAST & Target-Aware Models

- Allaway & McKeown (2020). "VAST: VAriability-Aware Stance Target." _EMNLP_.
- Augenstein et al. (2016). "Stance Detection with Bidirectional Conditional Encoding." _Proceedings of EMNLP_.

### Confidence Filtering & Selective Classification

- Geifman & El-Yaniv (2017). "Selective Classification for Deep Neural Networks." _NIPS_.
- Ren et al. (2018). "Learning to Reweight Examples for Robust Deep Learning." _ICML_.

### Multi-Label Learning

- Read et al. (2011). "Classifier chains for multi-label classification." _Machine Learning_.
- Tsoumakas & Katakis (2007). "Multi-label classification: An overview." _International Journal of Data Warehousing and Mining_.

---

## Revision History

| Date       | Change                              | Author |
| ---------- | ----------------------------------- | ------ |
| 2025-12-20 | Initial strategy document created   | System |
| 2025-12-20 | Adapted target representation (CAP) | System |
| 2025-12-20 | Documented zero-shot baseline       | System |
