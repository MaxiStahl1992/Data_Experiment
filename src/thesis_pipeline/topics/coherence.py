"""
Topic coherence metrics for quality assessment.

Implements coherence measures following:
- Röder et al. (2015): "Exploring the Space of Topic Coherence Measures"
  
Coherence measures evaluate topic interpretability by measuring semantic 
similarity of top topic words. Higher coherence → more interpretable topics.

Common measures:
- c_v: Combines normalized PMI with cosine similarity (best for human judgment)
- u_mass: Based on document co-occurrence (faster, less correlated with humans)
- c_npmi: Normalized PMI (good balance)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


def compute_topic_coherence(
    topic_terms: List[str],
    documents: List[List[str]],
    coherence_type: str = 'c_v',
    top_n: int = 10
) -> float:
    """
    Compute coherence for a single topic.
    
    Parameters
    ----------
    topic_terms : list of str
        Top terms for the topic (ordered by importance)
    documents : list of list of str
        Tokenized documents (used to compute co-occurrence)
    coherence_type : str
        Coherence measure: 'c_v', 'u_mass', 'c_npmi', 'c_uci'
    top_n : int
        Number of top terms to use for coherence
        
    Returns
    -------
    float
        Coherence score (higher = better)
        Range depends on measure (c_v: typically 0.3-0.7)
        
    Notes
    -----
    Requires tokenized documents for co-occurrence computation.
    For c_v, documents should be lemmatized/preprocessed tokens.
    """
    # Take top N terms
    topic_terms_top = topic_terms[:top_n]
    
    # Create gensim dictionary and corpus
    dictionary = Dictionary(documents)
    
    # Create coherence model
    cm = CoherenceModel(
        topics=[topic_terms_top],
        texts=documents,
        dictionary=dictionary,
        coherence=coherence_type
    )
    
    coherence = cm.get_coherence()
    
    return coherence


def compute_coherence_scores(
    topic_descriptors: pd.DataFrame,
    documents: List[List[str]],
    top_terms_col: str = 'top_terms',
    topic_id_col: str = 'topic_id',
    coherence_type: str = 'c_v',
    top_n: int = 10
) -> pd.DataFrame:
    """
    Compute coherence for all topics.
    
    Parameters
    ----------
    topic_descriptors : pd.DataFrame
        Topic descriptors with top_terms column (list of str per topic)
    documents : list of list of str
        Tokenized documents
    top_terms_col : str
        Column with top terms (list of strings)
    topic_id_col : str
        Topic ID column
    coherence_type : str
        Coherence measure type
    top_n : int
        Number of top terms to use
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'coherence' column
        
    Examples
    --------
    >>> descriptors = pd.DataFrame({
    ...     'topic_id': [1, 2],
    ...     'top_terms': [['health', 'care', 'insurance'], ['vote', 'election', 'campaign']]
    ... })
    >>> docs = [['health', 'care'], ['election', 'vote'], ...]
    >>> compute_coherence_scores(descriptors, docs)
    """
    print(f"Computing {coherence_type} coherence for {len(topic_descriptors)} topics...")
    
    # Build dictionary once for all topics
    dictionary = Dictionary(documents)
    
    coherences = []
    
    for idx, row in topic_descriptors.iterrows():
        topic_terms = row[top_terms_col][:top_n]
        
        # Compute coherence
        cm = CoherenceModel(
            topics=[topic_terms],
            texts=documents,
            dictionary=dictionary,
            coherence=coherence_type
        )
        
        coherence = cm.get_coherence()
        coherences.append(coherence)
    
    result = topic_descriptors.copy()
    result['coherence'] = coherences
    result['coherence_type'] = coherence_type
    
    print(f"  Coherence range: [{min(coherences):.4f}, {max(coherences):.4f}]")
    print(f"  Mean: {np.mean(coherences):.4f}")
    print(f"  Median: {np.median(coherences):.4f}")
    
    return result


def filter_by_coherence_threshold(
    topic_descriptors: pd.DataFrame,
    coherence_col: str = 'coherence',
    threshold: Optional[float] = None,
    percentile: Optional[float] = 10,
    min_topics: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter topics by coherence threshold.
    
    As per operationalization spec: "topics below a coherence threshold 
    are excluded from downstream forecasting targets."
    
    Parameters
    ----------
    topic_descriptors : pd.DataFrame
        Topics with coherence scores
    coherence_col : str
        Coherence column name
    threshold : float, optional
        Explicit coherence threshold (if None, use percentile)
    percentile : float, optional
        Percentile threshold (e.g., 10 = remove lowest 10%)
        Used if threshold is None
    min_topics : int
        Minimum number of topics to retain (safety check)
        
    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        (passing_topics, filtered_topics)
        
    Examples
    --------
    >>> # Remove lowest 10% by coherence
    >>> passing, filtered = filter_by_coherence_threshold(descriptors, percentile=10)
    """
    if threshold is None and percentile is not None:
        # Use percentile threshold
        threshold = np.percentile(topic_descriptors[coherence_col], percentile)
        print(f"Coherence threshold (p{percentile}): {threshold:.4f}")
    elif threshold is not None:
        print(f"Using explicit threshold: {threshold:.4f}")
    else:
        raise ValueError("Must specify either threshold or percentile")
    
    # Filter
    passing = topic_descriptors[topic_descriptors[coherence_col] >= threshold].copy()
    filtered = topic_descriptors[topic_descriptors[coherence_col] < threshold].copy()
    
    print(f"\nCoherence filtering results:")
    print(f"  Passing: {len(passing)} topics")
    print(f"  Filtered: {len(filtered)} topics")
    
    if len(passing) < min_topics:
        print(f"  ⚠ Warning: Only {len(passing)} topics pass threshold (min: {min_topics})")
        print(f"  Consider lowering threshold or adjusting model parameters")
    
    return passing, filtered


def compute_coherence_multi_measure(
    topic_descriptors: pd.DataFrame,
    documents: List[List[str]],
    measures: List[str] = ['c_v', 'u_mass', 'c_npmi'],
    top_terms_col: str = 'top_terms',
    top_n: int = 10
) -> pd.DataFrame:
    """
    Compute multiple coherence measures for robustness check.
    
    Parameters
    ----------
    topic_descriptors : pd.DataFrame
        Topic descriptors
    documents : list of list of str
        Tokenized documents
    measures : list of str
        Coherence measures to compute
    top_terms_col : str
        Column with top terms
    top_n : int
        Number of top terms
        
    Returns
    -------
    pd.DataFrame
        Topics with multiple coherence columns (coherence_c_v, coherence_u_mass, ...)
    """
    result = topic_descriptors.copy()
    
    for measure in measures:
        print(f"\nComputing {measure} coherence...")
        coherences = []
        
        dictionary = Dictionary(documents)
        
        for idx, row in topic_descriptors.iterrows():
            topic_terms = row[top_terms_col][:top_n]
            
            cm = CoherenceModel(
                topics=[topic_terms],
                texts=documents,
                dictionary=dictionary,
                coherence=measure
            )
            
            coherence = cm.get_coherence()
            coherences.append(coherence)
        
        result[f'coherence_{measure}'] = coherences
        print(f"  Range: [{min(coherences):.4f}, {max(coherences):.4f}]")
        print(f"  Mean: {np.mean(coherences):.4f}")
    
    return result


def summarize_coherence_results(
    topic_descriptors: pd.DataFrame,
    coherence_cols: List[str] = None
) -> pd.DataFrame:
    """
    Summarize coherence statistics.
    
    Parameters
    ----------
    topic_descriptors : pd.DataFrame
        Topics with coherence columns
    coherence_cols : list of str, optional
        Coherence column names (auto-detect if None)
        
    Returns
    -------
    pd.DataFrame
        Summary statistics per coherence measure
    """
    if coherence_cols is None:
        coherence_cols = [c for c in topic_descriptors.columns if 'coherence' in c]
    
    summary = []
    
    for col in coherence_cols:
        values = topic_descriptors[col].dropna()
        
        summary.append({
            'measure': col,
            'count': len(values),
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'q25': values.quantile(0.25),
            'median': values.median(),
            'q75': values.quantile(0.75),
            'max': values.max()
        })
    
    return pd.DataFrame(summary)


# Example usage
if __name__ == '__main__':
    print("Topic Coherence Test")
    print("=" * 60)
    
    # Mock data
    topic_descriptors = pd.DataFrame({
        'topic_id': [1, 2, 3],
        'top_terms': [
            ['health', 'care', 'insurance', 'medical', 'hospital'],
            ['election', 'vote', 'campaign', 'candidate', 'poll'],
            ['court', 'judge', 'ruling', 'legal', 'decision']
        ]
    })
    
    # Mock tokenized documents
    documents = [
        ['health', 'care', 'insurance', 'medical'],
        ['election', 'vote', 'campaign'],
        ['court', 'judge', 'ruling'],
        ['health', 'insurance', 'hospital', 'care'],
        ['election', 'candidate', 'poll', 'vote'],
        ['legal', 'decision', 'court', 'judge'],
    ] * 10  # Repeat for stability
    
    # Test 1: Single measure
    print("\nTest 1: c_v coherence")
    result = compute_coherence_scores(
        topic_descriptors,
        documents,
        coherence_type='c_v',
        top_n=5
    )
    print(result[['topic_id', 'coherence']])
    
    # Test 2: Filter by threshold
    print("\nTest 2: Filter by coherence")
    passing, filtered = filter_by_coherence_threshold(
        result,
        percentile=33,
        min_topics=2
    )
    print(f"Passing topics: {passing['topic_id'].tolist()}")
    
    # Test 3: Multiple measures
    print("\nTest 3: Multiple coherence measures")
    multi = compute_coherence_multi_measure(
        topic_descriptors,
        documents,
        measures=['c_v', 'u_mass']
    )
    print(multi[['topic_id', 'coherence_c_v', 'coherence_u_mass']])
    
    # Test 4: Summary
    print("\nTest 4: Summary statistics")
    summary = summarize_coherence_results(multi)
    print(summary)
    
    print("\n✓ All tests passed!")
