# Reddit Data Pipeline Architecture

## Overview

This pipeline processes Reddit data from two separate raw sources into a unified gold layer for downstream analysis (Topic → Stance → Polarization).

## Data Sources

### 1. Politosphere Comments (Zenodo)

- **Files**: `comments_2016-09.bz2`, `comments_2016-10.bz2`
- **Content**: ~8.8M comments from 605 political subreddits
- **Format**: bz2-compressed JSON lines
- **Coverage**: September-October 2016
- **Notebook**: `10_reddit_download_sep_oct_2016.ipynb`

### 2. Pushshift Submissions (Reddit Archives)

- **Files**: `RS_2016-09.zst`, `RS_2016-10.zst`
- **Content**: Submission posts (initiating threads)
- **Format**: zstandard-compressed JSON lines
- **Coverage**: All of Reddit for Sep-Oct 2016
- **Notebook**: `10a_reddit_extract_submissions.ipynb`
- **Filter**: Extract only submissions from 605 political subreddits

## Pipeline Stages

### Stage 1: Raw Data (00_raw/)

```
10_reddit_download_sep_oct_2016.ipynb
├─ Downloads politosphere comments from Zenodo
└─ Output: data/01_corpus/00_raw/reddit/politosphere_2016-09_2016-10/comments_*.bz2

Raw Pushshift archives:
└─ data/01_corpus/00_raw/reddit/politosphere_2016-09_2016-10/RS_2016-*.zst
```

### Stage 2: Silver Data (01_silver/)

```
11_reddit_extract_filter_silver.ipynb
├─ Processes politosphere comments
├─ Parses JSON, filters by date, validates schema
└─ Output: data/01_corpus/01_silver/reddit/comments/reddit_comment_YYYY-MM-DD.parquet

11a_reddit_extract_submissions_silver.ipynb
├─ Extracts and filters submissions from Pushshift archives
├─ Filters by 605 political subreddits (subreddits.txt)
├─ Filters by date (Sep-Oct 2016)
└─ Output: data/01_corpus/01_silver/reddit/submissions/YYYY-MM-DD.parquet
```

### Stage 3: Gold Data (02_gold/)

```
12_reddit_thread_pseudodocs_gold.ipynb
├─ Loads comment silver + submission silver
├─ Merges on thread_id (comment.link_id == submission.id)
├─ Adds submission title/selftext as thread context
└─ Output: data/01_corpus/02_gold/reddit/reddit_gold_YYYY-MM-DD.parquet
```

### Stage 4: Corpus Preparation (03_qa/)

```
14_reddit_corpus_prep_topics.ipynb
├─ Groups comments by thread_id
├─ Uses actual submission as thread initiator (not earliest comment)
├─ Creates thread pseudo-documents for topic modeling
└─ Outputs:
    ├─ thread_pseudodocs.parquet (for NMF/LDA)
    ├─ comment_thread_map.parquet (comment → thread lookup)
    └─ thread_metadata.parquet (stats per thread)
```

## Key Design Principles

### 1. Reproducibility

- All processing starts from raw sources (no circular dependencies)
- No loading from gold layer to filter raw data
- Each stage is independently reproducible

### 2. Subreddit Filtering

- **Filter file**: `data/01_corpus/00_raw/reddit/subreddits.txt`
- **Contains**: 605 political subreddits from politosphere dataset
- **Usage**: Filter Pushshift submissions by subreddit membership
- **Rationale**: Ensures submissions match comment coverage

### 3. Date Range

- **Period**: September 1 - October 31, 2016 (61 days)
- **Rationale**: Presidential election period (2016 election)
- **Implementation**: Filter by `created_utc` timestamp

### 4. Thread Structure

- **thread_id**: Submission ID (without t3\_ prefix)
- **Submissions**: Posts that initiate threads (t3\_ prefix)
- **Comments**: Replies to submissions (t1\_ prefix)
- **link_id**: Comment field pointing to parent submission (includes t3\_ prefix)

## Data Model

### Reddit ID Prefixes

- `t1_`: Comment
- `t3_`: Submission (thread initiator)
- `t5_`: Subreddit

### Thread Relationships

```
Submission (t3_abc123) = thread_id: abc123
├─ Comment 1 (t1_xyz456) → link_id: t3_abc123
├─ Comment 2 (t1_xyz789) → link_id: t3_abc123
└─ Comment 3 (t1_xyz012) → link_id: t3_abc123
```

### Gold Layer Schema

```python
{
    'comment_id': str,           # t1_ comment ID
    'thread_id': str,            # Submission ID (no prefix)
    'body': str,                 # Comment text
    'author': str,               # Comment author
    'created_utc': int,          # Comment timestamp
    'submission_title': str,     # From submission join
    'submission_selftext': str,  # From submission join
    'subreddit': str,            # Subreddit name
    ...
}
```

## File Paths

```
data/01_corpus/
├── 00_raw/reddit/
│   ├── subreddits.txt                                    # 605 political subreddits
│   └── politosphere_2016-09_2016-10/
│       ├── comments_2016-09.bz2                         # Politosphere comments
│       ├── comments_2016-10.bz2
│       ├── RS_2016-09.zst                               # Pushshift submissions
│       └── RS_2016-10.zst
│
├── 01_silver/reddit/
│   ├── comments/
│   │   └── reddit_comment_YYYY-MM-DD.parquet            # Cleaned comments
│   └── submissions/
│       └── YYYY-MM-DD.parquet                           # Extracted submissions (61 files)
│
├── 02_gold/reddit/
│   └── reddit_gold_YYYY-MM-DD.parquet                   # Merged threads
│
└── 03_qa/
    ├── thread_pseudodocs.parquet                         # For topic modeling
    ├── comment_thread_map.parquet                        # Comment lookups
    └── thread_metadata.parquet                           # Thread statistics
```

## Dependencies

### Python Packages

- `zstandard`: Decompress Pushshift .zst files
- `pandas`: Data processing
- `pyarrow`: Parquet I/O
- `tqdm`: Progress bars

### Installation

```bash
pip install zstandard pandas pyarrow tqdm
```

## Next Steps

1. ✅ **Notebook 11a**: Extract submissions (COMPLETE - produces silver layer)
2. ⏳ **Notebook 12**: Update to merge submissions with comments
3. ⏳ **Notebook 14**: Fix to use real submission data

## Notes

- Original notebook 14 incorrectly used earliest comment as submission
- Submissions were missing from politosphere dataset entirely
- Decision made to download Pushshift archives for scientific validity
- Architecture ensures no circular dependencies between processing stages
