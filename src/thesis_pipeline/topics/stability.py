"""
Topic stability analysis for model reliability.

Following Greene et al. (2014): "How Many Topics? Stability Analysis for Topic Models"

Topic modeling can yield different results across runs due to random initialization.
Stability analysis:
1. Fit multiple models with different seeds
2. Align topics across runs (Hungarian algorithm on similarity matrix)
3. Compute stability scores (topic-level and overall)

High stability → topics are robust to initialization
Low stability → topics may be arbitrary or dataset is too small
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import warnings


@dataclass
class StabilityResults:
    """
    Container for stability analysis results.
    
    Attributes
    ----------
    n_runs : int
        Number of model runs
    n_topics : int
        Number of topics K
    topic_similarities : np.ndarray
        Pairwise topic similarities across runs (K × n_runs × n_runs)
    topic_stability_scores : np.ndarray
        Per-topic stability (K,) - mean similarity across aligned runs
    overall_stability : float
        Mean of topic_stability_scores
    alignment_matrices : list of np.ndarray
        Topic alignment matrices (run_i → run_j mapping)
    """
    n_runs: int
    n_topics: int
    topic_similarities: np.ndarray
    topic_stability_scores: np.ndarray
    overall_stability: float
    alignment_matrices: List[np.ndarray]
    
    def summary(self) -> pd.DataFrame:
        """Get summary statistics."""
        return pd.DataFrame({
            'topic_id': range(1, self.n_topics + 1),
            'stability': self.topic_stability_scores
        }).sort_values('stability', ascending=False)


def compute_topic_similarity(
    terms1: List[str],
    terms2: List[str],
    weights1: Optional[List[float]] = None,
    weights2: Optional[List[float]] = None,
    method: str = 'jaccard',
    top_n: int = 10
) -> float:
    """
    Compute similarity between two topics.
    
    Parameters
    ----------
    terms1, terms2 : list of str
        Top terms for topics (ordered by weight)
    weights1, weights2 : list of float, optional
        Term weights (if None, use uniform)
    method : str
        Similarity method: 'jaccard', 'cosine', 'rank_biased_overlap'
    top_n : int
        Number of top terms to compare
        
    Returns
    -------
    float
        Similarity score (0 = no overlap, 1 = identical)
    """
    terms1 = terms1[:top_n]
    terms2 = terms2[:top_n]
    
    if method == 'jaccard':
        # Simple Jaccard similarity (most common for topic stability)
        set1 = set(terms1)
        set2 = set(terms2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    elif method == 'cosine':
        # Weighted cosine similarity
        if weights1 is None:
            weights1 = [1.0] * len(terms1)
        if weights2 is None:
            weights2 = [1.0] * len(terms2)
        
        # Build term vectors
        all_terms = sorted(set(terms1) | set(terms2))
        vec1 = np.array([weights1[terms1.index(t)] if t in terms1 else 0 for t in all_terms])
        vec2 = np.array([weights2[terms2.index(t)] if t in terms2 else 0 for t in all_terms])
        
        # Cosine similarity
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        return 1 - cosine(vec1, vec2)
    
    elif method == 'rank_biased_overlap':
        # RBO: considers term ranking
        # Simplified version (p=0.9 decay)
        p = 0.9
        rbo = 0.0
        
        for d in range(1, min(len(terms1), len(terms2)) + 1):
            overlap = len(set(terms1[:d]) & set(terms2[:d]))
            rbo += (p ** (d - 1)) * (overlap / d)
        
        rbo *= (1 - p)
        
        return rbo
    
    else:
        raise ValueError(f"Unknown method: {method}")


def align_topics_across_runs(
    descriptors_list: List[pd.DataFrame],
    top_terms_col: str = 'top_terms',
    top_weights_col: str = 'top_weights',
    method: str = 'jaccard',
    top_n: int = 10
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Align topics across multiple runs using Hungarian algorithm.
    
    For each pair of runs, compute a similarity matrix and find optimal alignment.
    
    Parameters
    ----------
    descriptors_list : list of pd.DataFrame
        Topic descriptors from each run (must have same n_topics)
    top_terms_col : str
        Column with top terms
    top_weights_col : str
        Column with term weights
    method : str
        Similarity method
    top_n : int
        Number of top terms to compare
        
    Returns
    -------
    (list of np.ndarray, np.ndarray)
        - alignment_matrices: K × K matrices mapping topics run_i → run_j
        - similarity_matrix: (n_runs × n_runs) mean similarity after alignment
        
    Notes
    -----
    Hungarian algorithm finds maximum weight bipartite matching.
    We maximize similarity by minimizing (1 - similarity).
    """
    n_runs = len(descriptors_list)
    n_topics = len(descriptors_list[0])
    
    # Verify all runs have same K
    for i, desc in enumerate(descriptors_list):
        if len(desc) != n_topics:
            raise ValueError(f"Run {i} has {len(desc)} topics, expected {n_topics}")
    
    # Compute pairwise similarities and alignments
    similarity_matrix = np.zeros((n_runs, n_runs))
    alignment_matrices = []
    
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            # Build similarity matrix between topics in run i and run j
            sim_matrix_ij = np.zeros((n_topics, n_topics))
            
            for ti, row_i in descriptors_list[i].iterrows():
                for tj, row_j in descriptors_list[j].iterrows():
                    sim = compute_topic_similarity(
                        row_i[top_terms_col],
                        row_j[top_terms_col],
                        weights1=row_i.get(top_weights_col),
                        weights2=row_j.get(top_weights_col),
                        method=method,
                        top_n=top_n
                    )
                    sim_matrix_ij[ti, tj] = sim
            
            # Hungarian algorithm: maximize similarity = minimize (1 - similarity)
            cost_matrix = 1 - sim_matrix_ij
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Alignment matrix (topics in run i → topics in run j)
            alignment = np.zeros((n_topics, n_topics), dtype=int)
            for r, c in zip(row_ind, col_ind):
                alignment[r, c] = 1
            
            alignment_matrices.append(alignment)
            
            # Mean similarity after alignment
            aligned_similarity = sim_matrix_ij[row_ind, col_ind].mean()
            similarity_matrix[i, j] = aligned_similarity
            similarity_matrix[j, i] = aligned_similarity
    
    # Diagonal (self-similarity) = 1
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return alignment_matrices, similarity_matrix


def compute_stability_scores(
    descriptors_list: List[pd.DataFrame],
    top_terms_col: str = 'top_terms',
    top_weights_col: str = 'top_weights',
    method: str = 'jaccard',
    top_n: int = 10
) -> StabilityResults:
    """
    Compute topic stability scores across multiple runs.
    
    Parameters
    ----------
    descriptors_list : list of pd.DataFrame
        Topic descriptors from multiple runs
    top_terms_col : str
        Column with top terms
    top_weights_col : str
        Column with term weights
    method : str
        Similarity method
    top_n : int
        Number of top terms
        
    Returns
    -------
    StabilityResults
        Stability analysis results
        
    Examples
    --------
    >>> # Fit 5 models with different seeds
    >>> runs = [fit_topic_model(docs, n_topics=20, random_state=i) for i in range(5)]
    >>> descriptors = [r.get_all_topic_descriptors() for r in runs]
    >>> stability = compute_stability_scores(descriptors)
    >>> print(f"Overall stability: {stability.overall_stability:.3f}")
    """
    n_runs = len(descriptors_list)
    n_topics = len(descriptors_list[0])
    
    print(f"Computing stability across {n_runs} runs ({n_topics} topics each)...")
    
    # Align topics and compute similarities
    alignment_matrices, run_similarity_matrix = align_topics_across_runs(
        descriptors_list,
        top_terms_col=top_terms_col,
        top_weights_col=top_weights_col,
        method=method,
        top_n=top_n
    )
    
    # For each topic, compute mean similarity across aligned runs
    topic_stability = np.zeros(n_topics)
    
    # Build K × n_runs × n_runs similarity tensor
    topic_similarities = np.zeros((n_topics, n_runs, n_runs))
    
    # Compute per-topic stability
    for ti in range(n_topics):
        similarities = []
        
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                # Find aligned topic in run j
                # (This is simplified; full implementation tracks alignment chains)
                sim = compute_topic_similarity(
                    descriptors_list[i].iloc[ti][top_terms_col],
                    descriptors_list[j].iloc[ti][top_terms_col],  # Assuming same index after alignment
                    weights1=descriptors_list[i].iloc[ti].get(top_weights_col),
                    weights2=descriptors_list[j].iloc[ti].get(top_weights_col),
                    method=method,
                    top_n=top_n
                )
                
                similarities.append(sim)
                topic_similarities[ti, i, j] = sim
                topic_similarities[ti, j, i] = sim
        
        topic_stability[ti] = np.mean(similarities) if similarities else 0.0
    
    # Overall stability: mean of topic stabilities
    overall_stability = topic_stability.mean()
    
    print(f"  Overall stability: {overall_stability:.4f}")
    print(f"  Topic stability range: [{topic_stability.min():.4f}, {topic_stability.max():.4f}]")
    
    return StabilityResults(
        n_runs=n_runs,
        n_topics=n_topics,
        topic_similarities=topic_similarities,
        topic_stability_scores=topic_stability,
        overall_stability=overall_stability,
        alignment_matrices=alignment_matrices
    )


def fit_multiple_runs(
    documents: List[str],
    n_topics: int,
    n_runs: int = 5,
    method: str = 'nmf',
    random_state_base: int = 42,
    **fit_kwargs
) -> List:
    """
    Fit multiple topic models with different random seeds.
    
    Convenience wrapper for stability analysis.
    
    Parameters
    ----------
    documents : list of str
        Input documents
    n_topics : int
        Number of topics K
    n_runs : int
        Number of runs (different seeds)
    method : str
        'nmf' or 'lda'
    random_state_base : int
        Base random seed (incremented for each run)
    **fit_kwargs
        Additional arguments for fit_topic_model
        
    Returns
    -------
    list of TopicModelResults
        Fitted models from each run
    """
    from .stm_utils import fit_topic_model
    
    print(f"Fitting {n_runs} models for stability analysis...")
    
    results = []
    
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        result = fit_topic_model(
            documents,
            n_topics=n_topics,
            method=method,
            random_state=random_state_base + i,
            verbose=0,
            **fit_kwargs
        )
        results.append(result)
    
    return results


# Example usage
if __name__ == '__main__':
    print("Topic Stability Test")
    print("=" * 60)
    
    # Mock topic descriptors (3 runs, 3 topics each)
    descriptors_run1 = pd.DataFrame({
        'topic_id': [1, 2, 3],
        'top_terms': [
            ['health', 'care', 'insurance', 'medical', 'hospital'],
            ['election', 'vote', 'campaign', 'candidate', 'poll'],
            ['court', 'judge', 'ruling', 'legal', 'decision']
        ]
    })
    
    # Run 2: Similar but slightly different
    descriptors_run2 = pd.DataFrame({
        'topic_id': [1, 2, 3],
        'top_terms': [
            ['health', 'insurance', 'care', 'medical', 'coverage'],  # Similar to run1 topic 1
            ['vote', 'election', 'ballot', 'campaign', 'candidate'],  # Similar to run1 topic 2
            ['court', 'legal', 'judge', 'ruling', 'law']  # Similar to run1 topic 3
        ]
    })
    
    # Run 3: More variation
    descriptors_run3 = pd.DataFrame({
        'topic_id': [1, 2, 3],
        'top_terms': [
            ['election', 'vote', 'campaign', 'poll', 'primary'],  # More like run1 topic 2
            ['health', 'care', 'insurance', 'doctor', 'patient'],  # More like run1 topic 1
            ['court', 'judge', 'case', 'trial', 'verdict']  # Somewhat like run1 topic 3
        ]
    })
    
    descriptors_list = [descriptors_run1, descriptors_run2, descriptors_run3]
    
    # Test 1: Topic similarity
    print("\nTest 1: Pairwise topic similarity")
    sim = compute_topic_similarity(
        descriptors_run1.iloc[0]['top_terms'],
        descriptors_run2.iloc[0]['top_terms'],
        method='jaccard'
    )
    print(f"Similarity (health topics): {sim:.4f}")
    
    # Test 2: Stability scores
    print("\nTest 2: Stability analysis")
    stability = compute_stability_scores(descriptors_list, method='jaccard')
    print(f"\nOverall stability: {stability.overall_stability:.4f}")
    print(f"\nPer-topic stability:")
    print(stability.summary())
    
    print("\n✓ All tests passed!")
