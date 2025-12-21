# 15b Implementation Plan: Hybrid Thread-Aware Topic Assignment

## Overview

Notebook 15b needs to be **completely rewritten** to implement the validated hybrid approach from 15a on the full corpus.

**Current 15b**: Uses old pseudo-document approach (NOT hybrid thread-aware)
**Required 15b**: Implement two-stage hybrid filtering validated in 15a

## Key Differences from Current 15b

### Current Approach (WRONG)
- Loads thread pseudo-documents
- Embeds pseudo-documents
- Assigns topics to pseudo-documents
- No comment-level filtering

### Required Approach (CORRECT - from 15a)
1. **Thread-Level**: Assign topics to submissions (title + selftext)
2. **Comment-Level**: Filter individual comments in topic threads

## Implementation Steps

### Part 1: Thread-Level Assignment (Submissions)

**Input**: Gold submissions from `data/01_corpus/02_gold/reddit/submissions/`

**Process**:
1. Load all submissions
2. Embed submission text (title + selftext)
3. Calculate similarity to 5 topic embeddings
4. Assign `thread_topic` if max_similarity >= 0.4
5. Save: `submissions_with_topics.parquet`

**Expected Results** (from 15a):
- ~8.5% of submissions will have topics
- High quality topic assignments

### Part 2: Comment-Level Filtering

**Input**: Gold comments from `data/01_corpus/02_gold/reddit/comments/`

**Process**:
1. Load all comments
2. Merge `thread_topic` from submissions
3. Filter: Keep only comments where submission has `thread_topic`
4. Embed remaining comments
5. Calculate comment similarity to its thread's topic
6. Keep comment if similarity >= 0.4
7. Save: `comments_with_topics.parquet`

**Expected Results** (from 15a):
- ~50-70% of comments in topic threads will pass
- ~3-5% overall retention of all comments
- High quality topic-relevant comments

## Required Code Structure

```python
# 1. Load topic definitions and embeddings (from 15a outputs)
# 2. Load ALL submissions from gold data
# 3. Embed submissions (title + selftext)
# 4. Calculate submission similarities
# 5. Assign thread_topic (threshold = 0.4)
# 6. Save submissions_with_topics.parquet
# 7. Load ALL comments from gold data
# 8. Filter to comments in topic threads
# 9. Embed filtered comments
# 10. Calculate comment similarities
# 11. Apply hybrid filter (comment sim >= 0.4 for its thread's topic)
# 12. Save comments_with_topics.parquet
```

## Data Sources

### Input
- `data/01_corpus/02_gold/reddit/submissions/*.parquet` - All submissions
- `data/01_corpus/02_gold/reddit/comments/*.parquet` - All comments  
- `data/02_topics/reddit/topic_definitions.json` - From 15a
- `data/02_topics/reddit/topic_embeddings.npy` - From 15a
- `data/02_topics/reddit/topic_ids.json` - From 15a

### Output
- `data/02_topics/reddit/submissions_with_topics.parquet` - All submissions with thread_topic
- `data/02_topics/reddit/comments_with_topics.parquet` - Filtered comments with topics

## Key Configuration

```python
THRESHOLD = 0.4  # Validated in 15a
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Same as 15a
```

## Expected Runtime

- Submission embedding: ~10-15 minutes
- Comment embedding: ~20-30 minutes
- Total: ~35-50 minutes

## Quality Checks

After processing, verify:
1. All comments have `thread_topic`
2. All comments have `parent_id` preserved
3. Similarity scores reasonable (mean ~0.5-0.6 for passing comments)
4. Topic distribution reasonable across 5 topics
5. Manual review of sample comments shows high quality

## Next Steps

1. **Backup current 15b** (optional - if you want to keep old approach)
2. **Clear 15b completely**
3. **Implement new hybrid approach** following structure from this plan
4. **Test on small subset first** (optional - to verify logic)
5. **Run on full corpus**
6. **Verify outputs**
7. **Proceed to 16a** (stance annotation sampling)

## Important Notes

- Do NOT use pseudo-documents (that was the old approach)
- Do NOT use thread_pseudodocs.parquet
- DO use raw gold submissions and comments
- DO implement two-stage filtering (submission + comment)
- DO preserve parent_id for all comments
- DO use exact same model and threshold as 15a

## Validation

Compare results with 15a expectations:
- Submission topic rate: ~8.5%
- Comment pass rate: ~50-70% of those in topic threads
- Overall comment retention: ~3-5%

If significantly different, investigate discrepancy.
