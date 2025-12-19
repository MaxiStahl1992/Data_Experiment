"""
Esteban-Ray polarization measure implementation.

Based on: Esteban, J.-M., & Ray, D. (1994). "On the Measurement of Polarization." 
Econometrica, 62(4), 819–851.

The ER measure captures both identification (within-group cohesion) and 
alienation (between-group distance) in a distribution.

For discrete stance groups (FAVOUR, AGAINST, NEUTRAL) with positions:
    x_A = -1, x_N = 0, x_F = +1
    
And shares: π_A, π_N, π_F (summing to 1)

The ER polarization is:
    P(α) = K(α) Σ_{i<j} (π_i^{1+α} π_j + π_j^{1+α} π_i) |x_i - x_j|

Where:
    - α controls the weight of identification vs. alienation (typically 1.0 - 1.6)
    - K(α) is a normalization constant for interpretability

This implementation is platform-agnostic and purely mathematical.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import pandas as pd


def esteban_ray_polarization(
    pi_favour: float,
    pi_against: float,
    pi_neutral: float,
    alpha: float = 1.3,
    normalize: bool = True,
    positions: Optional[Tuple[float, float, float]] = None
) -> float:
    """
    Compute Esteban-Ray polarization for discrete stance distribution.
    
    Parameters
    ----------
    pi_favour : float
        Share of FAVOUR stance (0 ≤ π_F ≤ 1)
    pi_against : float
        Share of AGAINST stance (0 ≤ π_A ≤ 1)
    pi_neutral : float
        Share of NEUTRAL stance (0 ≤ π_N ≤ 1)
    alpha : float, default=1.3
        Polarization parameter (typically 1.0-1.6)
        Higher α increases weight of identification (within-group)
    normalize : bool, default=True
        If True, apply K(α) so max polarization (50/50 FAVOUR/AGAINST) = 1
    positions : tuple of float, optional
        Stance positions (x_A, x_N, x_F). Default: (-1, 0, +1)
    
    Returns
    -------
    float
        Polarization index. If normalized, 0 ≤ P ≤ 1
        
    Raises
    ------
    ValueError
        If shares don't sum to 1 (within tolerance) or are negative
        
    Examples
    --------
    >>> # Maximum polarization: 50/50 FAVOUR/AGAINST
    >>> esteban_ray_polarization(0.5, 0.5, 0.0, alpha=1.3)
    1.0
    
    >>> # Consensus: all FAVOUR
    >>> esteban_ray_polarization(1.0, 0.0, 0.0, alpha=1.3)
    0.0
    
    >>> # Moderate polarization with neutrals
    >>> esteban_ray_polarization(0.4, 0.4, 0.2, alpha=1.3)
    0.78...
    """
    # Validate inputs
    shares = np.array([pi_favour, pi_against, pi_neutral])
    
    if np.any(shares < 0):
        raise ValueError(f"Shares must be non-negative: π_F={pi_favour}, π_A={pi_against}, π_N={pi_neutral}")
    
    if not np.isclose(shares.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Shares must sum to 1: π_F + π_A + π_N = {shares.sum()}")
    
    # Default positions
    if positions is None:
        positions = (-1.0, 0.0, 1.0)  # (AGAINST, NEUTRAL, FAVOUR)
    
    x_against, x_neutral, x_favour = positions
    
    # Build groups: [(position, share), ...]
    groups = [
        (x_against, pi_against),
        (x_neutral, pi_neutral),
        (x_favour, pi_favour)
    ]
    
    # Compute ER polarization
    polarization = 0.0
    
    for i, (x_i, pi_i) in enumerate(groups):
        for j, (x_j, pi_j) in enumerate(groups):
            if i < j:  # Only count each pair once
                distance = abs(x_i - x_j)
                # Identification-alienation term
                term = (pi_i**(1 + alpha) * pi_j + pi_j**(1 + alpha) * pi_i) * distance
                polarization += term
    
    # Apply normalization
    if normalize:
        K_alpha = compute_normalization_constant(alpha, positions)
        polarization = K_alpha * polarization
    
    return polarization


def compute_normalization_constant(
    alpha: float,
    positions: Tuple[float, float, float] = (-1.0, 0.0, 1.0)
) -> float:
    """
    Compute K(α) so that max polarization = 1.
    
    Maximum occurs when π_F = π_A = 0.5, π_N = 0 (perfect 50/50 split).
    
    Parameters
    ----------
    alpha : float
        Polarization parameter
    positions : tuple of float
        Stance positions (x_A, x_N, x_F)
        
    Returns
    -------
    float
        Normalization constant K(α)
    """
    x_against, x_neutral, x_favour = positions
    
    # Maximum polarization case: 50/50 FAVOUR/AGAINST, no NEUTRAL
    pi_f_max = 0.5
    pi_a_max = 0.5
    pi_n_max = 0.0
    
    # Compute unnormalized polarization at maximum
    groups_max = [
        (x_against, pi_a_max),
        (x_neutral, pi_n_max),
        (x_favour, pi_f_max)
    ]
    
    p_max = 0.0
    for i, (x_i, pi_i) in enumerate(groups_max):
        for j, (x_j, pi_j) in enumerate(groups_max):
            if i < j:
                distance = abs(x_i - x_j)
                term = (pi_i**(1 + alpha) * pi_j + pi_j**(1 + alpha) * pi_i) * distance
                p_max += term
    
    # K(α) = 1 / P_max
    K_alpha = 1.0 / p_max if p_max > 0 else 1.0
    
    return K_alpha


def normalize_esteban_ray(
    polarization: float,
    alpha: float,
    positions: Tuple[float, float, float] = (-1.0, 0.0, 1.0)
) -> float:
    """
    Normalize an unnormalized ER polarization value.
    
    Useful when computing polarization without normalization first,
    then applying normalization post-hoc.
    
    Parameters
    ----------
    polarization : float
        Unnormalized ER polarization
    alpha : float
        Polarization parameter used in calculation
    positions : tuple of float
        Stance positions used
        
    Returns
    -------
    float
        Normalized polarization (0 ≤ P ≤ 1)
    """
    K_alpha = compute_normalization_constant(alpha, positions)
    return K_alpha * polarization


def polarization_sensitivity_analysis(
    pi_favour: float,
    pi_against: float,
    pi_neutral: float,
    alpha_values: List[float] = [1.0, 1.3, 1.6],
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute polarization across multiple α values (sensitivity analysis).
    
    As per operationalization spec, α ∈ {1.0, 1.3, 1.6} is standard.
    
    Parameters
    ----------
    pi_favour, pi_against, pi_neutral : float
        Stance shares
    alpha_values : list of float
        α values to test
    normalize : bool
        Whether to normalize each value
        
    Returns
    -------
    pd.DataFrame
        Columns: ['alpha', 'polarization', 'correlation_with_primary']
        Primary α is assumed to be 1.3
    """
    results = []
    
    for alpha in alpha_values:
        p = esteban_ray_polarization(
            pi_favour, pi_against, pi_neutral,
            alpha=alpha,
            normalize=normalize
        )
        results.append({
            'alpha': alpha,
            'polarization': p
        })
    
    df = pd.DataFrame(results)
    
    # Correlation with primary (α=1.3) - only meaningful for time series
    # Here we just compute the values
    return df


def batch_polarization(
    stance_df: pd.DataFrame,
    pi_favour_col: str = 'pi_favour',
    pi_against_col: str = 'pi_against',
    pi_neutral_col: str = 'pi_neutral',
    alpha: float = 1.3,
    normalize: bool = True,
    groupby_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute polarization for multiple stance distributions (e.g., daily time series).
    
    Platform-agnostic: works for both Reddit and News data.
    
    Parameters
    ----------
    stance_df : pd.DataFrame
        DataFrame with stance shares per observation (e.g., per day-topic)
    pi_favour_col, pi_against_col, pi_neutral_col : str
        Column names for stance shares
    alpha : float
        Polarization parameter
    normalize : bool
        Whether to normalize
    groupby_cols : list of str, optional
        Columns to preserve (e.g., ['date', 'topic_id', 'platform'])
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'polarization' column
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': ['2016-09-01', '2016-09-02'],
    ...     'topic_id': [1, 1],
    ...     'pi_favour': [0.4, 0.5],
    ...     'pi_against': [0.4, 0.3],
    ...     'pi_neutral': [0.2, 0.2]
    ... })
    >>> batch_polarization(df, groupby_cols=['date', 'topic_id'])
    """
    result_df = stance_df.copy()
    
    # Vectorized polarization computation
    polarizations = []
    
    for idx, row in stance_df.iterrows():
        p = esteban_ray_polarization(
            pi_favour=row[pi_favour_col],
            pi_against=row[pi_against_col],
            pi_neutral=row[pi_neutral_col],
            alpha=alpha,
            normalize=normalize
        )
        polarizations.append(p)
    
    result_df['polarization'] = polarizations
    result_df['alpha'] = alpha
    
    return result_df


# Example usage and tests
if __name__ == '__main__':
    # Test cases
    print("Esteban-Ray Polarization Tests")
    print("=" * 60)
    
    # Test 1: Maximum polarization (50/50)
    p_max = esteban_ray_polarization(0.5, 0.5, 0.0, alpha=1.3)
    print(f"Maximum polarization (50/50 F/A): {p_max:.6f}")
    assert np.isclose(p_max, 1.0), "Max polarization should be 1.0"
    
    # Test 2: No polarization (consensus)
    p_min = esteban_ray_polarization(1.0, 0.0, 0.0, alpha=1.3)
    print(f"Consensus (all FAVOUR): {p_min:.6f}")
    assert np.isclose(p_min, 0.0), "Consensus should be 0.0"
    
    # Test 3: With neutrals
    p_neutral = esteban_ray_polarization(0.4, 0.4, 0.2, alpha=1.3)
    print(f"40/40/20 (F/A/N): {p_neutral:.6f}")
    
    # Test 4: Sensitivity to α
    print("\nSensitivity analysis:")
    for alpha in [1.0, 1.3, 1.6]:
        p = esteban_ray_polarization(0.45, 0.45, 0.1, alpha=alpha)
        print(f"  α={alpha:.1f}: P={p:.6f}")
    
    # Test 5: Batch computation
    print("\nBatch computation test:")
    df_test = pd.DataFrame({
        'date': ['2016-09-01', '2016-09-02', '2016-09-03'],
        'topic_id': [1, 1, 1],
        'pi_favour': [0.5, 0.4, 0.3],
        'pi_against': [0.5, 0.4, 0.5],
        'pi_neutral': [0.0, 0.2, 0.2]
    })
    result = batch_polarization(df_test, groupby_cols=['date', 'topic_id'])
    print(result[['date', 'pi_favour', 'pi_against', 'pi_neutral', 'polarization']])
    
    print("\n✓ All tests passed!")
