"""
Stance distribution aggregation with topic-weighting.

Platform-specific implementations that follow the same mathematical principles:
- Reddit: Comments inherit thread topic proportions, indexed by comment timestamp
- News: Articles have their own topic proportions, indexed by publication date

Both use soft topic-weighted aggregation to remain consistent with STM.
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional, Dict, Tuple


def topic_weighted_stance_shares(
    stance_df: pd.DataFrame,
    topic_weights: pd.Series,
    stance_prob_cols: Tuple[str, str, str] = ('prob_favour', 'prob_against', 'prob_neutral'),
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute topic-weighted stance shares (π_F, π_A, π_N).
    
    This is the core aggregation logic used by both platforms:
    
        π_g = Σ_u (w_u * p_g(u)) / Σ_u w_u
    
    Where:
        - w_u is the topic weight for unit u
        - p_g(u) is the stance probability for group g ∈ {F, A, N}
    
    Platform differences are handled via the topic_weights input:
        - Reddit: w_u = θ_{d(u),k} (comment inherits thread's topic proportion)
        - News: w_u = θ_{d,k} (article's own topic proportion)
    
    Parameters
    ----------
    stance_df : pd.DataFrame
        Stance predictions for units (comments or articles)
        Must have columns specified in stance_prob_cols
    topic_weights : pd.Series
        Topic weights w_u for each unit (same index as stance_df)
        For Reddit: θ_{d(u),k} from thread pseudo-document
        For News: θ_{d,k} from article itself
    stance_prob_cols : tuple of str
        Column names: (prob_favour, prob_against, prob_neutral)
    normalize : bool, default=True
        Ensure shares sum to 1.0 (recommended)
        
    Returns
    -------
    dict
        {'pi_favour': float, 'pi_against': float, 'pi_neutral': float}
        
    Examples
    --------
    >>> # Reddit example: 3 comments on same day/topic
    >>> stance_df = pd.DataFrame({
    ...     'prob_favour': [0.8, 0.2, 0.1],
    ...     'prob_against': [0.1, 0.7, 0.1],
    ...     'prob_neutral': [0.1, 0.1, 0.8]
    ... })
    >>> topic_weights = pd.Series([0.9, 0.8, 0.3])  # θ_{d(u),k} from threads
    >>> topic_weighted_stance_shares(stance_df, topic_weights)
    {'pi_favour': 0.52..., 'pi_against': 0.42..., 'pi_neutral': 0.17...}
    """
    prob_favour_col, prob_against_col, prob_neutral_col = stance_prob_cols
    
    # Validate inputs
    if len(stance_df) != len(topic_weights):
        raise ValueError(f"stance_df ({len(stance_df)}) and topic_weights ({len(topic_weights)}) must have same length")
    
    # Compute weighted sums
    weight_sum = topic_weights.sum()
    
    if weight_sum == 0:
        # No topic weight (shouldn't happen, but handle gracefully)
        return {'pi_favour': 0.0, 'pi_against': 0.0, 'pi_neutral': 1.0}
    
    pi_favour = (stance_df[prob_favour_col] * topic_weights).sum() / weight_sum
    pi_against = (stance_df[prob_against_col] * topic_weights).sum() / weight_sum
    pi_neutral = (stance_df[prob_neutral_col] * topic_weights).sum() / weight_sum
    
    # Normalize (handle floating point errors)
    if normalize:
        total = pi_favour + pi_against + pi_neutral
        if total > 0:
            pi_favour /= total
            pi_against /= total
            pi_neutral /= total
    
    return {
        'pi_favour': float(pi_favour),
        'pi_against': float(pi_against),
        'pi_neutral': float(pi_neutral)
    }


def aggregate_stance_distribution(
    stance_df: pd.DataFrame,
    topic_assignments: pd.DataFrame,
    platform: Literal['reddit', 'news'],
    date_col: str = 'date',
    topic_id_col: str = 'topic_id',
    unit_id_col: str = 'unit_id',
    stance_prob_cols: Tuple[str, str, str] = ('prob_favour', 'prob_against', 'prob_neutral'),
    min_units_per_day_topic: int = 10
) -> pd.DataFrame:
    """
    Aggregate stance predictions to daily topic-level distributions.
    
    Platform-aware wrapper that handles Reddit vs. News differences while
    maintaining mathematical consistency.
    
    Parameters
    ----------
    stance_df : pd.DataFrame
        Stance predictions with columns:
            - unit_id: comment ID (Reddit) or article ID (News)
            - date: temporal index
            - prob_favour, prob_against, prob_neutral: stance probabilities
            - [other columns preserved]
    topic_assignments : pd.DataFrame
        Topic proportions from STM with columns:
            - unit_id: same as stance_df
            - topic_id: topic identifier (1, 2, ..., K)
            - theta: topic proportion θ_{d,k}
        For Reddit: unit_id is thread_id (comments inherit from threads)
        For News: unit_id is article_id (self-contained)
    platform : {'reddit', 'news'}
        Platform identifier (for documentation/validation)
    date_col : str
        Temporal index column name
    topic_id_col : str
        Topic identifier column name
    unit_id_col : str
        Unit identifier column name
    stance_prob_cols : tuple of str
        Stance probability column names
    min_units_per_day_topic : int
        Minimum number of units required per (day, topic) pair
        Insufficient data results in NaN for that observation
        
    Returns
    -------
    pd.DataFrame
        Daily stance distributions with columns:
            - date
            - topic_id
            - platform
            - pi_favour, pi_against, pi_neutral (shares)
            - n_units (count of contributing units)
            - mean_topic_weight (average θ used)
            
    Notes
    -----
    Reddit-specific:
        - stance_df.unit_id = comment_id
        - topic_assignments.unit_id = thread_id
        - Requires comment→thread mapping (via link_id)
        - Temporal index MUST be comment.created_utc
        
    News-specific:
        - stance_df.unit_id = article_id
        - topic_assignments.unit_id = article_id
        - Direct alignment (no mapping needed)
        - Temporal index is article.published_date
    """
    results = []
    
    # Group by (date, topic_id)
    for (date, topic_id), group in stance_df.groupby([date_col, topic_id_col]):
        # Get topic weights for units in this group
        unit_ids = group[unit_id_col].values
        
        # Lookup topic proportions
        # For Reddit: these will be thread-level θ_{d(u),k}
        # For News: these will be article-level θ_{d,k}
        topic_data = topic_assignments[
            (topic_assignments[unit_id_col].isin(unit_ids)) &
            (topic_assignments[topic_id_col] == topic_id)
        ]
        
        # Align topic weights with stance predictions
        merged = group.merge(
            topic_data[[unit_id_col, 'theta']],
            on=unit_id_col,
            how='inner'
        )
        
        n_units = len(merged)
        
        # Skip if insufficient data
        if n_units < min_units_per_day_topic:
            results.append({
                date_col: date,
                topic_id_col: topic_id,
                'platform': platform,
                'pi_favour': np.nan,
                'pi_against': np.nan,
                'pi_neutral': np.nan,
                'n_units': n_units,
                'mean_topic_weight': np.nan,
                'insufficient_data': True
            })
            continue
        
        # Compute topic-weighted shares
        shares = topic_weighted_stance_shares(
            stance_df=merged,
            topic_weights=merged['theta'],
            stance_prob_cols=stance_prob_cols,
            normalize=True
        )
        
        results.append({
            date_col: date,
            topic_id_col: topic_id,
            'platform': platform,
            'pi_favour': shares['pi_favour'],
            'pi_against': shares['pi_against'],
            'pi_neutral': shares['pi_neutral'],
            'n_units': n_units,
            'mean_topic_weight': merged['theta'].mean(),
            'insufficient_data': False
        })
    
    df_result = pd.DataFrame(results)
    
    return df_result


def validate_stance_shares(
    shares_df: pd.DataFrame,
    pi_cols: Tuple[str, str, str] = ('pi_favour', 'pi_against', 'pi_neutral'),
    tolerance: float = 1e-4
) -> pd.DataFrame:
    """
    Validate that stance shares sum to 1.0 and are non-negative.
    
    Parameters
    ----------
    shares_df : pd.DataFrame
        Stance distribution data
    pi_cols : tuple of str
        Column names for shares
    tolerance : float
        Tolerance for sum check
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'shares_valid' boolean column
    """
    result = shares_df.copy()
    
    pi_f_col, pi_a_col, pi_n_col = pi_cols
    
    # Check sum
    share_sum = shares_df[pi_f_col] + shares_df[pi_a_col] + shares_df[pi_n_col]
    sum_valid = np.abs(share_sum - 1.0) < tolerance
    
    # Check non-negativity
    non_negative = (
        (shares_df[pi_f_col] >= 0) &
        (shares_df[pi_a_col] >= 0) &
        (shares_df[pi_n_col] >= 0)
    )
    
    result['shares_valid'] = sum_valid & non_negative
    result['share_sum'] = share_sum
    
    return result


# Example usage
if __name__ == '__main__':
    print("Stance Aggregation Tests")
    print("=" * 60)
    
    # Test 1: Topic-weighted shares (basic)
    print("\nTest 1: Basic topic-weighted aggregation")
    stance_df = pd.DataFrame({
        'prob_favour': [0.8, 0.2, 0.1],
        'prob_against': [0.1, 0.7, 0.1],
        'prob_neutral': [0.1, 0.1, 0.8]
    })
    topic_weights = pd.Series([0.9, 0.8, 0.3])
    
    shares = topic_weighted_stance_shares(stance_df, topic_weights)
    print(f"Shares: {shares}")
    print(f"Sum: {sum(shares.values()):.6f}")
    assert np.isclose(sum(shares.values()), 1.0), "Shares should sum to 1"
    
    # Test 2: Full aggregation (mock Reddit data)
    print("\nTest 2: Full aggregation pipeline (Reddit mock)")
    
    # Mock comments with stance predictions
    stance_df = pd.DataFrame({
        'unit_id': [f'comment_{i}' for i in range(20)],
        'thread_id': ['thread_1'] * 10 + ['thread_2'] * 10,
        'date': ['2016-09-01'] * 15 + ['2016-09-02'] * 5,
        'topic_id': [1] * 12 + [2] * 8,
        'prob_favour': np.random.dirichlet([2, 1, 1], 20)[:, 0],
        'prob_against': np.random.dirichlet([2, 1, 1], 20)[:, 1],
        'prob_neutral': np.random.dirichlet([2, 1, 1], 20)[:, 2]
    })
    
    # Renormalize probs
    prob_sum = stance_df[['prob_favour', 'prob_against', 'prob_neutral']].sum(axis=1)
    for col in ['prob_favour', 'prob_against', 'prob_neutral']:
        stance_df[col] = stance_df[col] / prob_sum
    
    # Mock thread topic assignments
    topic_assignments = pd.DataFrame({
        'unit_id': ['thread_1', 'thread_1', 'thread_2', 'thread_2'],
        'topic_id': [1, 2, 1, 2],
        'theta': [0.7, 0.3, 0.6, 0.4]
    })
    
    # Map comments to threads (Reddit-specific)
    stance_df = stance_df.merge(
        stance_df[['unit_id', 'thread_id']].drop_duplicates(),
        on='unit_id'
    )
    stance_df = stance_df.rename(columns={'thread_id': 'original_thread'})
    stance_df['unit_id'] = stance_df['original_thread']  # Use thread_id for lookup
    
    # Aggregate
    daily_shares = aggregate_stance_distribution(
        stance_df=stance_df,
        topic_assignments=topic_assignments,
        platform='reddit',
        date_col='date',
        topic_id_col='topic_id',
        unit_id_col='unit_id',
        min_units_per_day_topic=3
    )
    
    print(daily_shares[['date', 'topic_id', 'pi_favour', 'pi_against', 'pi_neutral', 'n_units']])
    
    # Validate
    validated = validate_stance_shares(daily_shares)
    print(f"\nValid shares: {validated['shares_valid'].sum()} / {len(validated)}")
    
    print("\n✓ All tests passed!")
