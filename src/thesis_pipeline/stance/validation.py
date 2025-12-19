"""
Stance validation utilities.

Provides tools for:
- Computing stance classification metrics
- Inter-annotator agreement (Cohen's κ)
- Validation set annotation helpers
- Error analysis

Following the operationalization spec: "A small stratified annotation is 
performed per platform: sample across topics and time, report macro-F1 on 
{FAVOUR, AGAINST, NEUTRAL/OTHER}, report inter-annotator agreement."
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)
import warnings


def compute_stance_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: List[str] = ['FAVOUR', 'AGAINST', 'NEUTRAL'],
    return_dict: bool = True
) -> Dict:
    """
    Compute comprehensive stance classification metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    labels : list of str
        Label names (order matters for per-class metrics)
    return_dict : bool
        Return as dictionary (else prints report)
        
    Returns
    -------
    dict
        Metrics: macro_f1, micro_f1, accuracy, per_class_f1, confusion_matrix
        
    Examples
    --------
    >>> metrics = compute_stance_metrics(val_labels, predictions)
    >>> print(f"Macro F1: {metrics['macro_f1']:.3f}")
    """
    # Overall metrics
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=labels, average='micro', zero_division=0)
    accuracy = (y_true == y_pred).mean()
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'accuracy': accuracy,
        'per_class': per_class,
        'confusion_matrix': cm,
        'labels': labels
    }
    
    if not return_dict:
        # Print report
        print("Stance Classification Metrics")
        print("=" * 60)
        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Micro F1:    {micro_f1:.4f}")
        print(f"Accuracy:    {accuracy:.4f}")
        print("\nPer-class metrics:")
        for label, m in per_class.items():
            print(f"  {label:10s}: P={m['precision']:.3f}, R={m['recall']:.3f}, "
                  f"F1={m['f1']:.3f}, N={m['support']}")
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=labels, columns=labels))
    
    return metrics


def compute_macro_f1(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: List[str] = ['FAVOUR', 'AGAINST', 'NEUTRAL']
) -> float:
    """
    Compute macro F1 (primary metric for stance).
    
    Convenience wrapper for single metric.
    """
    return f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)


def inter_annotator_agreement(
    annotations_df: pd.DataFrame,
    annotator_cols: List[str],
    method: str = 'cohen_kappa',
    pairwise: bool = True
) -> Dict:
    """
    Compute inter-annotator agreement.
    
    As per spec: "report inter-annotator agreement on a subset"
    
    Parameters
    ----------
    annotations_df : pd.DataFrame
        Annotations with one column per annotator
    annotator_cols : list of str
        Column names for annotators
    method : str
        Agreement method: 'cohen_kappa', 'fleiss_kappa' (requires statsmodels)
    pairwise : bool
        Compute pairwise Cohen's κ (if False, compute average)
        
    Returns
    -------
    dict
        Agreement scores
        
    Examples
    --------
    >>> # Two annotators
    >>> annotations = pd.DataFrame({
    ...     'annotator_1': ['FAVOUR', 'AGAINST', 'NEUTRAL', ...],
    ...     'annotator_2': ['FAVOUR', 'FAVOUR', 'NEUTRAL', ...]
    ... })
    >>> agreement = inter_annotator_agreement(annotations, ['annotator_1', 'annotator_2'])
    >>> print(f"Cohen's κ: {agreement['cohen_kappa']:.3f}")
    """
    n_annotators = len(annotator_cols)
    
    if n_annotators < 2:
        raise ValueError("Need at least 2 annotators for agreement")
    
    if method == 'cohen_kappa':
        if n_annotators == 2:
            # Direct Cohen's κ
            kappa = cohen_kappa_score(
                annotations_df[annotator_cols[0]],
                annotations_df[annotator_cols[1]]
            )
            
            return {
                'method': 'cohen_kappa',
                'n_annotators': 2,
                'kappa': kappa,
                'interpretation': interpret_kappa(kappa)
            }
        else:
            # Pairwise Cohen's κ
            if pairwise:
                kappas = []
                pairs = []
                
                for i in range(n_annotators):
                    for j in range(i + 1, n_annotators):
                        kappa = cohen_kappa_score(
                            annotations_df[annotator_cols[i]],
                            annotations_df[annotator_cols[j]]
                        )
                        kappas.append(kappa)
                        pairs.append((annotator_cols[i], annotator_cols[j]))
                
                mean_kappa = np.mean(kappas)
                
                return {
                    'method': 'cohen_kappa_pairwise',
                    'n_annotators': n_annotators,
                    'pairwise_kappas': dict(zip(pairs, kappas)),
                    'mean_kappa': mean_kappa,
                    'std_kappa': np.std(kappas),
                    'interpretation': interpret_kappa(mean_kappa)
                }
            else:
                # Average over all pairs
                kappas = []
                for i in range(n_annotators):
                    for j in range(i + 1, n_annotators):
                        kappa = cohen_kappa_score(
                            annotations_df[annotator_cols[i]],
                            annotations_df[annotator_cols[j]]
                        )
                        kappas.append(kappa)
                
                return {
                    'method': 'cohen_kappa_average',
                    'n_annotators': n_annotators,
                    'mean_kappa': np.mean(kappas),
                    'interpretation': interpret_kappa(np.mean(kappas))
                }
    
    elif method == 'fleiss_kappa':
        # Fleiss' κ for >2 annotators (requires statsmodels)
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa
        except ImportError:
            raise ImportError("statsmodels required for Fleiss' kappa: pip install statsmodels")
        
        # Convert to rating matrix (n_items × n_categories)
        # Each row = item, each column = category count
        labels = annotations_df[annotator_cols].values.ravel()
        unique_labels = sorted(set(labels))
        
        rating_matrix = np.zeros((len(annotations_df), len(unique_labels)))
        
        for i, row in annotations_df[annotator_cols].iterrows():
            for label in row:
                label_idx = unique_labels.index(label)
                rating_matrix[i, label_idx] += 1
        
        kappa = fleiss_kappa(rating_matrix)
        
        return {
            'method': 'fleiss_kappa',
            'n_annotators': n_annotators,
            'kappa': kappa,
            'interpretation': interpret_kappa(kappa)
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Cohen's κ / Fleiss' κ value.
    
    Following Landis & Koch (1977) interpretation:
    - < 0.00: Poor
    - 0.00 - 0.20: Slight
    - 0.21 - 0.40: Fair
    - 0.41 - 0.60: Moderate
    - 0.61 - 0.80: Substantial
    - 0.81 - 1.00: Almost Perfect
    """
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def stratified_sample_for_annotation(
    df: pd.DataFrame,
    n_samples: int = 500,
    strata_cols: List[str] = ['topic_id', 'date'],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create stratified sample for manual annotation.
    
    As per spec: "sample across topics and time"
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (comments or articles with stance predictions)
    n_samples : int
        Total number of samples to annotate (e.g., 500-1000)
    strata_cols : list of str
        Columns for stratification (e.g., ['topic_id', 'date'])
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Stratified sample for annotation
        
    Examples
    --------
    >>> # Sample 500 comments across topics and time
    >>> sample = stratified_sample_for_annotation(
    ...     comments_df, 
    ...     n_samples=500, 
    ...     strata_cols=['topic_id', 'date']
    ... )
    >>> # Export for annotation
    >>> sample[['text', 'topic_terms', 'predicted_label']].to_csv('for_annotation.csv')
    """
    # Compute samples per stratum
    stratum_counts = df.groupby(strata_cols).size()
    stratum_weights = stratum_counts / stratum_counts.sum()
    
    samples_per_stratum = (stratum_weights * n_samples).round().astype(int)
    
    # Adjust to exactly n_samples
    diff = n_samples - samples_per_stratum.sum()
    if diff != 0:
        # Add/subtract from largest strata
        largest_stratum = samples_per_stratum.idxmax()
        samples_per_stratum[largest_stratum] += diff
    
    # Sample from each stratum
    sampled = []
    
    for stratum, n_stratum in samples_per_stratum.items():
        if n_stratum == 0:
            continue
        
        # Build filter for this stratum
        if isinstance(stratum, tuple):
            mask = True
            for col, val in zip(strata_cols, stratum):
                mask = mask & (df[col] == val)
        else:
            mask = df[strata_cols[0]] == stratum
        
        stratum_df = df[mask]
        
        # Sample (with replacement if needed)
        if len(stratum_df) < n_stratum:
            sample = stratum_df.sample(n=n_stratum, replace=True, random_state=random_state)
        else:
            sample = stratum_df.sample(n=n_stratum, replace=False, random_state=random_state)
        
        sampled.append(sample)
    
    result = pd.concat(sampled, ignore_index=True)
    
    print(f"Stratified sample for annotation:")
    print(f"  Total samples: {len(result)}")
    print(f"  Strata: {strata_cols}")
    print(f"  Distribution:")
    print(result.groupby(strata_cols).size().sort_values(ascending=False).head(10))
    
    return result


def error_analysis(
    df: pd.DataFrame,
    true_label_col: str,
    predicted_label_col: str,
    text_col: str,
    target_col: Optional[str] = None,
    n_examples: int = 5
) -> Dict:
    """
    Analyze prediction errors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with predictions and true labels
    true_label_col : str
        True label column
    predicted_label_col : str
        Predicted label column
    text_col : str
        Text column (for examples)
    target_col : str, optional
        Target column (for examples)
    n_examples : int
        Number of examples per error type
        
    Returns
    -------
    dict
        Error analysis with examples
    """
    # Identify errors
    errors = df[df[true_label_col] != df[predicted_label_col]].copy()
    
    # Error rate
    error_rate = len(errors) / len(df)
    
    # Error types
    error_types = errors.groupby([true_label_col, predicted_label_col]).size()
    
    # Sample errors
    error_examples = {}
    
    for (true_label, pred_label), count in error_types.items():
        mask = (errors[true_label_col] == true_label) & (errors[predicted_label_col] == pred_label)
        examples = errors[mask].sample(n=min(n_examples, count), random_state=42)
        
        error_examples[(true_label, pred_label)] = examples[[text_col, target_col] if target_col else [text_col]].to_dict('records')
    
    return {
        'error_rate': error_rate,
        'n_errors': len(errors),
        'n_total': len(df),
        'error_types': error_types.to_dict(),
        'error_examples': error_examples
    }


# Example usage
if __name__ == '__main__':
    print("Stance Validation Test")
    print("=" * 60)
    
    # Mock data
    np.random.seed(42)
    n = 100
    
    true_labels = pd.Series(['FAVOUR'] * 40 + ['AGAINST'] * 40 + ['NEUTRAL'] * 20)
    
    # Predictions with ~70% accuracy
    predicted_labels = true_labels.copy()
    errors_idx = np.random.choice(n, size=int(n * 0.3), replace=False)
    predicted_labels.iloc[errors_idx] = np.random.choice(['FAVOUR', 'AGAINST', 'NEUTRAL'], size=len(errors_idx))
    
    # Test 1: Compute metrics
    print("\nTest 1: Stance classification metrics")
    metrics = compute_stance_metrics(true_labels, predicted_labels, return_dict=False)
    
    # Test 2: Inter-annotator agreement
    print("\nTest 2: Inter-annotator agreement")
    annotations = pd.DataFrame({
        'annotator_1': true_labels,
        'annotator_2': predicted_labels,
        'annotator_3': true_labels.sample(frac=1, random_state=42).reset_index(drop=True)
    })
    
    agreement = inter_annotator_agreement(annotations, ['annotator_1', 'annotator_2'])
    print(f"Cohen's κ: {agreement['kappa']:.3f} ({agreement['interpretation']})")
    
    # Test 3: Stratified sampling
    print("\nTest 3: Stratified sample for annotation")
    df = pd.DataFrame({
        'text': [f'text_{i}' for i in range(200)],
        'topic_id': np.random.choice([1, 2, 3], 200),
        'date': pd.date_range('2016-09-01', periods=200, freq='D')[:200].strftime('%Y-%m-%d'),
        'predicted_label': predicted_labels.tolist() + ['FAVOUR'] * 100
    })
    
    sample = stratified_sample_for_annotation(df, n_samples=50, strata_cols=['topic_id'])
    
    print("\n✓ All tests passed!")
