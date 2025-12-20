# Affective Polarization Pipeline

## Quick Start

### 1. Read the Codebook

📖 **[affective_polarization_codebook.md](./affective_polarization_codebook.md)**

This is your annotation guide. It defines:

- 4-level scale (None, Adversarial, Intolerant, Belligerent)
- Decision tree for consistent coding
- 30+ examples with rationales
- Quality control guidelines

**Time:** 20-30 minutes to read and internalize

---

### 2. Open the Notebook

📓 **[18_sentiment_detetction.ipynb](./18_sentiment_detetction.ipynb)**

Run cells 1-6:

- Loads topic-enhanced Reddit data
- Creates test set (stratified by topic)
- Shows annotation interface

**Time:** 2-3 minutes to run cells

---

### 3. Start Annotating

🏷️ **Interactive Widget**

```python
# In notebook cell 8
annotate_interactive(test_set, start_idx=0)
```

For each text:

1. Is there a political target? (party, voters, ideological group)
2. Is there emotional language? (not just policy talk)
3. How hostile? (0=none, 1=insults, 2=enemies, 3=violence)

**Target:** 100-150 annotated samples
**Time:** 3-5 hours (break into sessions)

---

### 4. After Annotation Complete

⚙️ **Model Comparison**

The notebook will:

- Test 4 different models
- Evaluate with Macro-F1 and Quadratic Kappa
- Select best model
- Apply to full dataset (~700k comments)

**Time:** 3-4 hours (mostly compute)

---

## Files in This Directory

| File                                       | Purpose                                       |
| ------------------------------------------ | --------------------------------------------- |
| `affective_polarization_codebook.md`       | **Formal annotation guide** (read first!)     |
| `affective_polarization_implementation.md` | **Detailed implementation notes** (reference) |
| `18_sentiment_detetction.ipynb`            | **Main notebook** (run this)                  |
| `xx_reddit_stance_detection.ipynb`         | Old stance approach (archived)                |

---

## Key Concepts

### Affective Polarization

Negative affect and hostility toward **political out-groups**:

- Not just disagreement → **emotional negativity**
- Not just any insults → **toward political groups**
- Not just toxicity → **delegitimization and dehumanization**

### The 4 Levels

| Level | Name        | Example                                  |
| ----- | ----------- | ---------------------------------------- |
| 0     | None        | "Republicans won the House."             |
| 1     | Adversarial | "Democrats are clueless."                |
| 2     | Intolerant  | "These leftists are enemies of America." |
| 3     | Belligerent | "Lock them all up."                      |

### Why This Matters

- **Democratic health:** Delegitimization → erosion of norms
- **Escalation tracking:** From incivility to violence
- **Topic-specific:** Measure polarization by political issue
- **Temporal:** Track changes over election cycle

---

## Expected Output

### Data Files

```
data/04_affective_polarization/reddit/
├── test_set_annotations.parquet          # Your annotations
├── comments_with_polarization.parquet    # ~550k rows with scores
└── submissions_with_polarization.parquet # ~150k rows with scores
```

### New Columns

- `affective_polarization_label`: 0-3 (category)
- `affective_polarization_score`: 0-1 (continuous)
- `polarization_confidence`: Model confidence
- `polarization_probs`: [P(0), P(1), P(2), P(3)]

### Analysis Possibilities

- **By topic:** Which issues are most polarized?
- **Over time:** Did polarization increase toward election?
- **By subreddit:** Community-level hostility
- **By user:** Individual polarization trajectories

---

## Troubleshooting

### "I'm not sure how to code this text"

→ Follow the decision tree in the codebook
→ When uncertain, default to lower level
→ Note the case for review

### "My label distribution seems off"

Expected (roughly):

- Level 0: 40-60%
- Level 1: 30-40%
- Level 2: 10-20%
- Level 3: 2-5%

### "Annotation is taking too long"

→ You should average 1-2 minutes per text
→ Break into multiple sessions
→ Skip truly ambiguous cases initially

### "I need more examples"

→ See codebook section 5 (30+ examples)
→ Add your own examples as you annotate
→ Share difficult cases for discussion

---

## Timeline

| Day   | Activity                     | Hours |
| ----- | ---------------------------- | ----- |
| Day 1 | Read codebook, annotate 50   | 2-3   |
| Day 2 | Annotate 50-100 more         | 2-3   |
| Day 3 | Model comparison & selection | 2-3   |
| Day 4 | Full processing & validation | 3-4   |

**Total:** 9-13 hours over 3-4 days

---

## Next Steps

1. ✅ **NOW:** Read [affective_polarization_codebook.md](./affective_polarization_codebook.md)
2. ✅ **TODAY:** Start annotating (target 30-50 samples)
3. ✅ **TOMORROW:** Continue annotation (reach 100+)
4. ⏳ **NEXT:** Run model comparison
5. ⏳ **FINALLY:** Process full dataset

---

## Questions?

- **Codebook questions:** See codebook section 4 (borderline cases)
- **Technical issues:** Check notebook cell comments
- **Methodology:** See [affective_polarization_implementation.md](./affective_polarization_implementation.md)

---

**Status:** ✅ Ready to start annotation
**Critical Next Step:** Begin annotating test set using the codebook
**Estimated Time to First Results:** 2-4 days
