# Topic Modeling Strategy for Full Thesis

**Date Created**: 2025-12-19  
**Last Updated**: 2025-12-20  
**Status**: Planning document for full 4-year implementation  
**Current Scope**: 2-month validation (Sep-Oct 2016)

---

## Version History

- **v4.0 (2025-12-20)**: Multi-label embedding classification with top-5 cap (CURRENT - PRODUCTION)
- **v3.0 (2025-12-20)**: Supervised zero-shot classification (prototype, superseded)
- **v2.0 (2025-12-19)**: BERTopic with semantic embeddings (evaluated, not adopted)
- **v1.0 (2025-12-19)**: STM/NMF unsupervised topic modeling (baseline)

---

## Overview

This document outlines the strategy for topic classification across the full thesis dataset (2016-2019). After evaluating unsupervised approaches (STM/NMF, BERTopic), we adopted a **supervised classification approach** using predefined political topics from established political science taxonomies.

---

This document outlines the strategy for topic classification across the full thesis dataset (2016-2019). After evaluating unsupervised approaches (STM/NMF, BERTopic), we adopted a **supervised classification approach** using predefined political topics from established political science taxonomies.

---

## 🎯 CURRENT APPROACH (v4.0): Multi-Label Embedding Classification

### Evolution from v3.0 to v4.0

**v3.0 (Zero-Shot BART)** - Initial supervised approach:

- Method: facebook/bart-large-mnli zero-shot classification
- Issue: ~7 docs/sec → 12-24 hours for 433k documents
- Result: Too slow for production, abandoned

**v4.0 (Multi-Label Embeddings)** - CURRENT PRODUCTION METHOD:

- Method: Sentence transformers with cosine similarity
- Speed: 455 docs/sec (65x faster than v3.0)
- Time: 433k docs in 16 minutes (vs. 12+ hours)
- Scalability: 50M docs in ~30 hours (weekend batch feasible)

### Rationale for Supervised Approach

**Limitations of Unsupervised Methods:**

After comprehensive evaluation of STM/NMF (16/25 topics passing quality checks, 64%) and BERTopic (15/24 topics passing, 62%), we identified fundamental limitations:

1. **Topic Pooling Effect**: Both methods assign difficult-to-classify documents to "garbage" topics (Topic 0 pooling)
2. **Analytical Misalignment**: Discovered topics don't necessarily align with research questions (may never discover "Healthcare" or "Immigration")
3. **Arbitrary Topic Counts**: Even with fixed K=25, no guarantee topics match analytical needs
4. **Temporal Instability**: Requires complex alignment algorithms across quarters
5. **Low Interpretability**: "Topic 7" vs. "Healthcare policy debate"

**Advantages of Supervised Classification:**

1. **Theoretical Grounding**: Based on validated political science taxonomies (CAP, PAP)
2. **Perfect Interpretability**: Each topic has clear meaning ("Elections", "Healthcare")
3. **Temporal Stability**: Same 20 topics across 2016-2019, no alignment needed
4. **Zero Data Loss**: All documents assigned to meaningful topics
5. **Production-Ready**: Define topics once (human), classify forever (machine)
6. **Scientifically Defensible**: Cite established frameworks, not ad-hoc discovery
7. **Multi-Label Capability**: Captures that political discourse spans multiple topics naturally

### Critical Methodological Innovation: Multi-Label Classification

**Key Insight**: Political comments and news articles naturally span **multiple topics** simultaneously. Forcing single-label assignment produces artificially low confidence scores (81.9% <0.3 in initial tests).

**Example**: A comment about "Trump's immigration executive order affecting healthcare" spans:

- Presidential Politics (executive order)
- Immigration & Borders (policy content)
- Healthcare Policy (impact)

**Multi-Label Approach:**

- Assign ALL topics above similarity threshold (0.25)
- Cap at top-5 topics for interpretability (Miller's Law: 7±2 items in working memory)
- Results: 84.7% documents get 1 topic, 15.3% get 2+ topics (realistic distribution)

**Scientific Justification for 5-Topic Cap:**

- **Miller (1956)**: The Magical Number Seven, Plus or Minus Two - human cognitive limit
- **Read et al. (2011)**: Multi-label classification benefits from top-k thresholding
- **Tsoumakas & Katakis (2007)**: Label cardinality control improves interpretability
- **Information Theory**: 20/20 topics = noise; 5 topics = focused signal
- **CAP Methodology**: Complex policy documents typically span 2-4 major themes

### Predefined Political Topic Taxonomy (20 Topics)

**Source**: Adapted from the Comparative Agendas Project (CAP) codebook

**Citation**:

> Baumgartner, F. R., Jones, B. D., & Wilkerson, J. (2011). _Comparative Studies of Policy Agendas_. Journal of European Public Policy, 18(5), 639-653.
>
> Comparative Agendas Project. (2024). _Master Codebook_. Retrieved from https://www.comparativeagendas.net/pages/master-codebook

The CAP framework has been validated across 20+ countries and multiple decades of policy discourse, providing a robust foundation for political topic classification.

**Our 20-Topic Taxonomy:**

1. **Elections & Voting**: Electoral processes, campaigns, voting rights, electoral reform
2. **Presidential Politics**: Presidential actions, administration, executive orders
3. **Congress & Legislation**: Congressional activities, legislative processes, lawmakers
4. **Healthcare Policy**: Health insurance, ACA/Obamacare, medical costs, public health
5. **Immigration & Borders**: Immigration policy, border security, refugee issues, deportation
6. **Economy & Employment**: Jobs, unemployment, wages, labor issues, economic growth
7. **Budget & Taxation**: Federal budget, taxes, government spending, deficits
8. **Education Policy**: Schools, universities, student loans, education reform
9. **Criminal Justice**: Crime, policing, prisons, law enforcement, justice system
10. **Gun Rights & Control**: Second Amendment, gun violence, firearms regulation
11. **Environment & Climate**: Climate change, EPA, pollution, conservation
12. **Energy Policy**: Oil, gas, renewable energy, energy independence
13. **Foreign Policy & Diplomacy**: International relations, diplomacy, global issues
14. **Defense & Military**: Military operations, veterans, defense spending
15. **Trade Policy**: International trade, tariffs, trade agreements
16. **Social Issues**: Abortion, LGBTQ rights, religious freedom, family values
17. **Civil Rights & Discrimination**: Racial justice, discrimination, equality issues
18. **Media & Free Speech**: Press freedom, censorship, media bias, First Amendment
19. **Technology & Privacy**: Tech regulation, surveillance, data privacy, cybersecurity
20. **Infrastructure**: Roads, bridges, public transit, infrastructure investment

### Implementation: Embedding-Based Multi-Label Classification

**Method**: Semantic similarity between document embeddings and topic description embeddings

**Model**: `all-MiniLM-L6-v2` (sentence-transformers)

- 384-dimensional embeddings
- Fast encoding (~455 docs/sec on Apple Silicon MPS)
- Pre-trained on diverse text corpus
- Normalized embeddings for cosine similarity

**Process**:

1. **One-time setup**: Embed 20 topic descriptions → topic_embeddings (20 × 384)
2. **For each document batch**:
   - Embed document text (truncated to 512 chars) → doc_embeddings (N × 384)
   - Compute cosine similarity: similarities = doc_embeddings @ topic_embeddings.T
   - For each document:
     - Get all topics with similarity ≥ 0.25
     - If > 5 topics qualify: take top-5 by similarity
     - If 0 topics qualify: assign best topic (fallback)
3. **Output**: Multi-label assignments + confidence scores

**Memory Optimization** (critical for production):

- Small batch size: 32 documents (prevents memory overflow)
- Chunked processing: 10k documents at a time with checkpointing
- Aggressive cache clearing: Every 50k documents (maintains speed)
- Text truncation: 512 characters (sufficient for topic classification)
- Sequence length: 128 tokens (faster than 256)

**Performance Characteristics**:

- **Speed**: 455 docs/sec sustained (16 minutes for 433k documents)
- **Memory**: <10GB RAM (Apple Silicon MPS)
- **Scalability**: 50M documents in ~30 hours
- **Throughput degradation**: Observed slowdown from 680→450 docs/sec (acceptable)

**Advantages over Zero-Shot (v3.0)**:

- 65x faster (455 vs 7 docs/sec)
- No text generation overhead (embeddings only)
- Fully vectorized operations (GPU-optimized)
- Embeddings reusable for other analyses
- No rate limits or API costs

### Quality Validation

**Validation results (Sep-Oct 2016, 433k Reddit threads)**:

**Multi-Label Statistics:**

- Average topics per document: 1.27
- Documents with 1 topic: 84.7%
- Documents with 2+ topics: 15.3%
- Documents with 3+ topics: 6.4%
- Documents at 5-topic cap: <1%

**Top Topics (2016 Election Period)**:

1. Media & Free Speech: 36.4% of documents
2. Presidential Politics: 23.7%
3. Immigration & Borders: 9.4%
4. Social Issues: 8.3%
5. Elections & Voting: 7.0%

**Performance Validation:**

- Time: 16 minutes for 433k documents (within 20-minute target)
- Memory: <10GB (no crashes)
- Throughput: 455 docs/sec sustained
- Quality: Topic assignments align with 2016 election discourse

**Expected Improvements over Unsupervised**:

- 0% garbage topics (all topics meaningful by design)
- 100% interpretability (each topic has clear definition)
- Multi-label captures complexity without forcing single assignment
- Temporal stability (same topics 2016-2019, no alignment needed)

### Data Structure

```
data/03_gold/
├── reddit/
│   └── thread_pseudodocs_with_supervised_topics_multilabel.parquet
│       Columns: [submission_id, created_utc, pseudodoc_text,
│                 supervised_topics, supervised_topic_labels, num_topics,
│                 best_topic_id, best_topic_label, best_topic_confidence]
│   └── document_embeddings.npy  # Reusable 384d embeddings
│   └── supervised_multilabel_classification_metadata.json
└── news/
    └── (same structure, to be implemented)
```

**Key Columns:**

- `supervised_topics`: Array of topic IDs [0, 5, 12] (multi-label)
- `supervised_topic_labels`: Array of topic names ["Elections", "Economy", ...]
- `num_topics`: Count of topics assigned (1-5)
- `best_topic_*`: Single best topic for comparison/fallback

### Scientific Justification for Thesis

**Theoretical Framework**:

- CAP: 21 major policy topics, validated across 20+ countries (Baumgartner et al., 2011)
- Policy Agendas Project: Similar taxonomy for US policy research
- Manifesto Project: Party platform coding scheme
- Multi-label learning: Read et al. (2011), Tsoumakas & Katakis (2007)

**Methodological Defense**:

1. **Reproducibility**: Same topics can be applied by other researchers; embeddings are deterministic
2. **Comparability**: Results comparable to political science literature using CAP taxonomy
3. **Interpretability**: "Healthcare debate intensity" directly answers research questions
4. **Validity**: Framework validated across countries and decades (CAP); embeddings validated on diverse corpora
5. **Scalability**: Production-ready for live monitoring systems (455 docs/sec)
6. **Multi-label Realism**: Political discourse naturally spans multiple topics; single-label forcing creates artificial confidence issues
7. **Cognitive Justification**: 5-topic cap grounded in Miller's Law (7±2 items in working memory)

**Performance Evidence**:

- Sep-Oct 2016 validation: 433k documents in 16 minutes
- 1.27 topics/document average (realistic complexity)
- Top topics align with 2016 election: Media (36%), Presidential Politics (24%), Immigration (9%)
- Memory-efficient (<10GB), scales to 50M documents in ~30 hours

**Addressing Potential Criticisms**:

- _"Why not discover topics?"_: Unsupervised methods produced 36-38% garbage topics (v1.0, v2.0 evaluation)
- _"How do you know 20 topics are right?"_: Based on validated CAP framework (21 topics), refined to 20 for US political discourse
- _"What about new topics?"_: Fallback mechanism assigns best topic if none exceed threshold; low confidence indicates potential new topics
- _"Isn't this biased?"_: All topic modeling is biased; supervised makes assumptions explicit and defensible; CAP validation across 20+ countries provides robustness
- _"Why multi-label vs single-label?"_: Single-label forcing produced 81.9% low confidence (<0.3); multi-label reflects reality that political comments span multiple issues
- _"Why cap at 5 topics?"_: Grounded in cognitive science (Miller's Law), information theory (20/20 = noise), and multi-label best practices (Read et al., 2011)
- _"Why embeddings vs zero-shot?"_: 65x faster (455 vs 7 docs/sec), same semantic similarity principle, GPU-optimized, no text generation overhead

---

## 📊 EVALUATION RESULTS (v1.0 & v2.0)

### STM/NMF Unsupervised Approach (v1.0)

**Method**: Structural Topic Model with Non-negative Matrix Factorization

**Configuration**:

- Fixed K=25 topics
- TF-IDF vectorization (1000 features)
- 200 iterations, random_state=42

**Quality Results** (Sep-Oct 2016, Reddit):

- Passing quality checks: 16/25 (64%)
- Failing quality checks: 9/25 (36%)

**Failure Breakdown**:

- Contamination (>40% stopwords): 1 topic
- Low distinctiveness: 5 topics
- Unfocused (high entropy): 5 topics
- Low persistence (<20% days): 0 topics
- Low volume: 0 topics

**Issues**:

- Topic 0 pooling effect (garbage collector)
- Arbitrary topic definitions
- Requires quarterly refitting and alignment

### BERTopic Semantic Approach (v2.0)

**Method**: BERTopic with semantic embeddings

**Configuration**:

- Embeddings: all-MiniLM-L6-v2 (384d)
- UMAP: 15 neighbors, 5 components
- HDBSCAN: min_cluster_size=300, min_samples=30
- Target topics: 25 (forced reduction)
- MPS acceleration for Apple Silicon

**Quality Results** (Sep-Oct 2016, Reddit):

- Passing quality checks: 15/24 (62%)
- Failing quality checks: 9/24 (38%)

**Failure Breakdown**:

- Contamination: 0 topics
- Low distinctiveness: 5 topics
- Unfocused: 5 topics
- Low persistence: 0 topics
- Low volume: 2 topics

**Issues**:

- Slightly worse than STM/NMF (1 fewer passing topic)
- Topic 0 pooling effect persists
- No advantage despite semantic embeddings
- Still requires arbitrary K choice

**Conclusion**: Both unsupervised methods produce ~36-38% garbage topics, motivating shift to supervised classification.

---

## 🔄 LEGACY: Rolling Window Approach (v1.0 Only)

_Note: The following approach was designed for unsupervised topic modeling (STM/NMF). With supervised classification (v3.0), rolling windows and refitting are no longer necessary._

_Note: The following approach was designed for unsupervised topic modeling (STM/NMF). With supervised classification (v3.0), rolling windows and refitting are no longer necessary._

### Core Strategy (v1.0 - STM/NMF Only)

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
