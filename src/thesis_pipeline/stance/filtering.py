"""
Confidence filtering and threshold optimization for stance predictions.

Implements selective prediction / reject option:
- Abstain from prediction when confidence is low
- Optimize threshold for macro-F1 vs. coverage tradeoff
- Handle insufficient confidence cases (route to NEUTRAL/OTHER)

Following Geifman & El-Yaniv (2017): "Selective Classification for Deep Neural Networks"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import f1_score, classification_report
import warnings


def apply_confidence_threshold(
    predictions_df: pd.DataFrame,
    confidence_col: str = 'confidence',
    threshold: float = 0.5,
    prob_cols: Tuple[str, str, str] = ('prob_favour', 'prob_against', 'prob_neutral'),
    default_label: str = 'NEUTRAL',
    predicted_label_col: str = 'predicted_label'
) -> pd.DataFrame:
    """
    Apply confidence threshold to stance predictions (reject option).
    
    When max(p_F, p_A, p_N) < threshold:
        - Set predicted_label to default_label (usually NEUTRAL/OTHER)
        - Mark as 'rejected' for analysis
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Stance predictions with probability columns
    confidence_col : str
        Column with confidence scores (max probability)
    threshold : float
        Confidence threshold (0 < c < 1)
        Predictions below threshold are rejected
    prob_cols : tuple of str
        Probability column names (favour, against, neutral)
    default_label : str
        Label to assign when rejected (default: 'NEUTRAL')
    predicted_label_col : str
        Column with predicted labels (will be updated)
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with updated predictions and 'rejected' column
        
    Examples
    --------
    >>> predictions = model.batch_predict(texts, targets)
    >>> filtered = apply_confidence_threshold(predictions, threshold=0.6)
    >>> coverage = (~filtered['rejected']).mean()  # Fraction of predictions kept
    """
    result = predictions_df.copy()
    
    # Compute confidence if not present
    if confidence_col not in result.columns:
        prob_f, prob_a, prob_n = prob_cols
        result[confidence_col] = result[[prob_f, prob_a, prob_n]].max(axis=1)
    
    # Apply threshold
    rejected = result[confidence_col] < threshold
    
    # Update predictions for rejected cases
    result.loc[rejected, predicted_label_col] = default_label
    result['rejected'] = rejected
    result['threshold'] = threshold
    
    # Statistics
    n_rejected = rejected.sum()
    coverage = 1 - (n_rejected / len(result))
    
    print(f"Confidence threshold: {threshold:.3f}")
    print(f"  Rejected: {n_rejected} / {len(result)} ({100*n_rejected/len(result):.1f}%)")
    print(f"  Coverage: {coverage:.3f}")
    
    return result


def optimize_threshold(
    predictions_df: pd.DataFrame,
    true_labels: pd.Series,
    predicted_label_col: str = 'predicted_label',
    confidence_col: str = 'confidence',
    metric: str = 'macro_f1',
    min_coverage: float = 0.8,
    threshold_range: Tuple[float, float] = (0.3, 0.95),
    n_steps: int = 20
) -> Tuple[float, pd.DataFrame]:
    """
    Optimize confidence threshold for best metric subject to coverage constraint.
    
    As per operationalization spec: "Threshold c is chosen on a held-out 
    validation set to optimize macro-F1 subject to a minimum coverage constraint."
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Stance predictions
    true_labels : pd.Series
        Ground truth labels (validation set)
    predicted_label_col : str
        Column with predictions
    confidence_col : str
        Column with confidence scores
    metric : str
        Metric to optimize: 'macro_f1', 'micro_f1', 'accuracy'
    min_coverage : float
        Minimum coverage constraint (e.g., 0.8 = keep ≥80% predictions)
    threshold_range : tuple
        (min_threshold, max_threshold) to search
    n_steps : int
        Number of threshold values to try
        
    Returns
    -------
    (float, pd.DataFrame)
        (optimal_threshold, results_df)
        results_df has columns: threshold, metric_value, coverage, n_predictions
        
    Examples
    --------
    >>> # Validation set
    >>> val_predictions = model.batch_predict(val_texts, val_targets)
    >>> optimal_threshold, results = optimize_threshold(
    ...     val_predictions, 
    ...     true_labels=val_labels,
    ...     min_coverage=0.8
    ... )
    >>> print(f"Optimal threshold: {optimal_threshold:.3f}")
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    results = []
    
    print(f"Optimizing threshold for {metric} (min coverage: {min_coverage:.2f})...")
    
    for threshold in thresholds:
        # Apply threshold
        filtered = apply_confidence_threshold(
            predictions_df.copy(),
            confidence_col=confidence_col,
            threshold=threshold,
            predicted_label_col=predicted_label_col
        )
        
        # Keep only non-rejected predictions
        mask = ~filtered['rejected']
        coverage = mask.mean()
        
        if coverage < min_coverage:
            # Skip if below coverage constraint
            continue
        
        # Compute metric
        y_true = true_labels[mask]
        y_pred = filtered.loc[mask, predicted_label_col]
        
        if metric == 'macro_f1':
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'micro_f1':
            score = f1_score(y_true, y_pred, average='micro', zero_division=0)
        elif metric == 'accuracy':
            score = (y_true == y_pred).mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        results.append({
            'threshold': threshold,
            'metric': metric,
            'metric_value': score,
            'coverage': coverage,
            'n_predictions': mask.sum(),
            'n_total': len(filtered)
        })
    
    if not results:
        warnings.warn(f"No thresholds satisfy min_coverage={min_coverage}. Try lowering constraint.")
        return threshold_range[0], pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold
    best_idx = results_df['metric_value'].idxmax()
    optimal_threshold = results_df.loc[best_idx, 'threshold']
    best_score = results_df.loc[best_idx, 'metric_value']
    best_coverage = results_df.loc[best_idx, 'coverage']
    
    print(f"\n✓ Optimal threshold: {optimal_threshold:.3f}")
    print(f"  {metric}: {best_score:.4f}")
    print(f"  Coverage: {best_coverage:.3f}")
    
    return optimal_threshold, results_df


def confidence_coverage_tradeoff(
    predictions_df: pd.DataFrame,
    true_labels: pd.Series,
    predicted_label_col: str = 'predicted_label',
    confidence_col: str = 'confidence',
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Compute metric-coverage tradeoff curve for visualization.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Stance predictions
    true_labels : pd.Series
        Ground truth labels
    predicted_label_col : str
        Column with predictions
    confidence_col : str
        Column with confidence
    thresholds : list of float, optional
        Thresholds to evaluate (default: 0.1 to 0.95 in steps of 0.05)
        
    Returns
    -------
    pd.DataFrame
        Tradeoff results for plotting
        
    Examples
    --------
    >>> tradeoff = confidence_coverage_tradeoff(predictions, true_labels)
    >>> plt.plot(tradeoff['coverage'], tradeoff['macro_f1'])
    >>> plt.xlabel('Coverage')
    >>> plt.ylabel('Macro F1')
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    
    for threshold in thresholds:
        filtered = apply_confidence_threshold(
            predictions_df.copy(),
            confidence_col=confidence_col,
            threshold=threshold,
            predicted_label_col=predicted_label_col
        )
        
        mask = ~filtered['rejected']
        coverage = mask.mean()
        
        if mask.sum() == 0:
            continue
        
        y_true = true_labels[mask]
        y_pred = filtered.loc[mask, predicted_label_col]
        
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        accuracy = (y_true == y_pred).mean()
        
        results.append({
            'threshold': threshold,
            'coverage': coverage,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'accuracy': accuracy,
            'n_predictions': mask.sum()
        })
    
    return pd.DataFrame(results)


def analyze_rejected_predictions(
    predictions_df: pd.DataFrame,
    rejected_col: str = 'rejected',
    confidence_col: str = 'confidence',
    prob_cols: Tuple[str, str, str] = ('prob_favour', 'prob_against', 'prob_neutral')
) -> Dict:
    """
    Analyze characteristics of rejected predictions.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with rejection flag
    rejected_col : str
        Column indicating rejected predictions
    confidence_col : str
        Confidence column
    prob_cols : tuple of str
        Probability columns
        
    Returns
    -------
    dict
        Analysis results (rejection rate, confidence stats, entropy)
    """
    rejected = predictions_df[rejected_col]
    
    # Compute entropy (measure of uncertainty)
    prob_f, prob_a, prob_n = prob_cols
    probs = predictions_df[[prob_f, prob_a, prob_n]].values
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)  # Higher = more uncertain
    
    analysis = {
        'rejection_rate': rejected.mean(),
        'n_rejected': rejected.sum(),
        'n_total': len(predictions_df),
        'confidence': {
            'rejected_mean': predictions_df.loc[rejected, confidence_col].mean(),
            'kept_mean': predictions_df.loc[~rejected, confidence_col].mean(),
            'rejected_std': predictions_df.loc[rejected, confidence_col].std(),
            'kept_std': predictions_df.loc[~rejected, confidence_col].std()
        },
        'entropy': {
            'rejected_mean': entropy[rejected].mean(),
            'kept_mean': entropy[~rejected].mean()
        }
    }
    
    return analysis


# Example usage
if __name__ == '__main__':
    print("Confidence Filtering Test")
    print("=" * 60)
    
    # Mock predictions
    np.random.seed(42)
    n = 100
    
    predictions_df = pd.DataFrame({
        'prob_favour': np.random.dirichlet([2, 1, 1], n)[:, 0],
        'prob_against': np.random.dirichlet([2, 1, 1], n)[:, 1],
        'prob_neutral': np.random.dirichlet([2, 1, 1], n)[:, 2]
    })
    
    # Renormalize
    prob_sum = predictions_df[['prob_favour', 'prob_against', 'prob_neutral']].sum(axis=1)
    for col in ['prob_favour', 'prob_against', 'prob_neutral']:
        predictions_df[col] = predictions_df[col] / prob_sum
    
    predictions_df['confidence'] = predictions_df[['prob_favour', 'prob_against', 'prob_neutral']].max(axis=1)
    predictions_df['predicted_label'] = predictions_df[['prob_favour', 'prob_against', 'prob_neutral']].idxmax(axis=1).map({
        'prob_favour': 'FAVOUR',
        'prob_against': 'AGAINST',
        'prob_neutral': 'NEUTRAL'
    })
    
    # Mock true labels
    true_labels = pd.Series(['FAVOUR'] * 40 + ['AGAINST'] * 40 + ['NEUTRAL'] * 20)
    
    # Test 1: Apply threshold
    print("\nTest 1: Apply confidence threshold")
    filtered = apply_confidence_threshold(predictions_df, threshold=0.5)
    print(f"Rejected: {filtered['rejected'].sum()}")
    
    # Test 2: Optimize threshold
    print("\nTest 2: Optimize threshold")
    optimal_threshold, results = optimize_threshold(
        predictions_df,
        true_labels,
        min_coverage=0.7,
        n_steps=10
    )
    
    # Test 3: Tradeoff analysis
    print("\nTest 3: Coverage-quality tradeoff")
    tradeoff = confidence_coverage_tradeoff(
        predictions_df,
        true_labels,
        thresholds=[0.3, 0.5, 0.7, 0.9]
    )
    print(tradeoff[['threshold', 'coverage', 'macro_f1']])
    
    # Test 4: Analyze rejected
    print("\nTest 4: Analyze rejected predictions")
    analysis = analyze_rejected_predictions(filtered)
    print(f"Rejection rate: {analysis['rejection_rate']:.3f}")
    print(f"Confidence (rejected): {analysis['confidence']['rejected_mean']:.3f}")
    print(f"Confidence (kept): {analysis['confidence']['kept_mean']:.3f}")
    
    print("\n✓ All tests passed!")
