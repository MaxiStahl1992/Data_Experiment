"""
Stance Model Comparison Framework

Provides utilities for:
- Comparing multiple stance detection models
- Calculating evaluation metrics
- Visualizing comparison results
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


def evaluate_stance_model(
    predictions: List[str],
    ground_truth: List[str],
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for stance detection.
    
    Parameters
    ----------
    predictions : list of str
        Predicted stance labels (FAVOUR/AGAINST/NEUTRAL)
    ground_truth : list of str
        Ground truth stance labels
    model_name : str
        Name of the model being evaluated
        
    Returns
    -------
    dict
        Metrics including accuracy, macro/micro F1, per-class metrics, kappa
    """
    # Calculate overall metrics
    accuracy = accuracy_score(ground_truth, predictions)
    
    # Per-class metrics
    labels = ['FAVOUR', 'AGAINST', 'NEUTRAL']
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, zero_division=0
    )
    
    # Macro and weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='macro', zero_division=0
    )
    
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )
    
    # Cohen's Kappa (inter-rater reliability)
    kappa = cohen_kappa_score(ground_truth, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_p,
        'weighted_recall': weighted_r,
        'weighted_f1': weighted_f1,
        'cohen_kappa': kappa,
        'favour_precision': precision[0],
        'favour_recall': recall[0],
        'favour_f1': f1[0],
        'favour_support': int(support[0]),
        'against_precision': precision[1],
        'against_recall': recall[1],
        'against_f1': f1[1],
        'against_support': int(support[1]),
        'neutral_precision': precision[2],
        'neutral_recall': recall[2],
        'neutral_f1': f1[2],
        'neutral_support': int(support[2]),
        'confusion_matrix': cm.tolist()
    }


class StanceModelComparison:
    """
    Framework for comparing multiple stance detection models.
    
    Handles:
    - Running predictions from multiple models
    - Calculating metrics for each model
    - Visualizing comparison results
    - Selecting best model based on criteria
    """
    
    def __init__(self, test_set: pd.DataFrame):
        """
        Initialize comparison framework.
        
        Parameters
        ----------
        test_set : pd.DataFrame
            Test set with columns: text, topic_label, ground_truth_stance
        """
        if 'ground_truth_stance' not in test_set.columns:
            raise ValueError("test_set must have 'ground_truth_stance' column")
        
        self.test_set = test_set
        self.results = {}  # Store results per model
        self.metrics = {}  # Store metrics per model
        
    def evaluate_model(
        self,
        model,
        model_name: str,
        batch_size: int = 16,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate a stance model on the test set.
        
        Parameters
        ----------
        model : ImprovedNLIStanceModel or similar
            Stance model with predict() method
        model_name : str
            Name for this model in results
        batch_size : int
            Batch size for predictions
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        pd.DataFrame
            Test set with predictions added
        """
        print(f"Evaluating {model_name}...")
        
        # Get predictions
        texts = self.test_set['text'].tolist()
        targets = self.test_set['topic_label'].tolist()
        
        # Run batch prediction
        predictions_df = model.batch_predict(
            texts=texts,
            targets=targets,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Add predictions to test set
        results_df = self.test_set.copy()
        results_df[f'{model_name}_prediction'] = predictions_df['predicted_label']
        results_df[f'{model_name}_confidence'] = predictions_df['confidence']
        results_df[f'{model_name}_prob_favour'] = predictions_df['prob_favour']
        results_df[f'{model_name}_prob_against'] = predictions_df['prob_against']
        results_df[f'{model_name}_prob_neutral'] = predictions_df['prob_neutral']
        
        # Calculate metrics
        metrics = evaluate_stance_model(
            predictions=predictions_df['predicted_label'].tolist(),
            ground_truth=self.test_set['ground_truth_stance'].tolist(),
            model_name=model_name
        )
        
        # Store results
        self.results[model_name] = results_df
        self.metrics[model_name] = metrics
        
        # Print summary
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Macro-F1: {metrics['macro_f1']:.3f}")
        print(f"  Cohen's κ: {metrics['cohen_kappa']:.3f}")
        print(f"  FAVOUR F1: {metrics['favour_f1']:.3f}")
        print(f"  AGAINST F1: {metrics['against_f1']:.3f}")
        print(f"  NEUTRAL F1: {metrics['neutral_f1']:.3f}")
        
        return results_df
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get summary table of all model metrics.
        
        Returns
        -------
        pd.DataFrame
            Metrics comparison table
        """
        if not self.metrics:
            raise ValueError("No model results yet. Run evaluate_model() first.")
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([
            {
                'model_name': m['model_name'],
                'accuracy': m['accuracy'],
                'macro_f1': m['macro_f1'],
                'weighted_f1': m['weighted_f1'],
                'cohen_kappa': m['cohen_kappa'],
                'favour_f1': m['favour_f1'],
                'against_f1': m['against_f1'],
                'neutral_f1': m['neutral_f1'],
                'favour_support': m['favour_support'],
                'against_support': m['against_support'],
                'neutral_support': m['neutral_support']
            }
            for m in self.metrics.values()
        ])
        
        # Sort by macro_f1
        metrics_df = metrics_df.sort_values('macro_f1', ascending=False).reset_index(drop=True)
        
        return metrics_df
    
    def plot_comparison(self, figsize=(14, 10)) -> plt.Figure:
        """
        Create comprehensive comparison visualization.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
            
        Returns
        -------
        plt.Figure
            Comparison figure
        """
        if not self.metrics:
            raise ValueError("No model results yet. Run evaluate_model() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        model_names = list(self.metrics.keys())
        
        # 1. Overall metrics comparison
        metrics_df = self.get_metrics_summary()
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, metrics_df['accuracy'], width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x, metrics_df['macro_f1'], width, label='Macro-F1', alpha=0.8)
        axes[0, 0].bar(x + width, metrics_df['cohen_kappa'], width, label="Cohen's κ", alpha=0.8)
        
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Overall Metrics Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.0)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Per-class F1 scores
        favour_f1 = [self.metrics[m]['favour_f1'] for m in model_names]
        against_f1 = [self.metrics[m]['against_f1'] for m in model_names]
        neutral_f1 = [self.metrics[m]['neutral_f1'] for m in model_names]
        
        axes[0, 1].bar(x - width, favour_f1, width, label='FAVOUR', color='green', alpha=0.7)
        axes[0, 1].bar(x, against_f1, width, label='AGAINST', color='red', alpha=0.7)
        axes[0, 1].bar(x + width, neutral_f1, width, label='NEUTRAL', color='gray', alpha=0.7)
        
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Per-Class F1 Scores', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1.0)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Confusion matrix for best model
        best_model = metrics_df.iloc[0]['model_name']
        cm = np.array(self.metrics[best_model]['confusion_matrix'])
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['FAVOUR', 'AGAINST', 'NEUTRAL'],
            yticklabels=['FAVOUR', 'AGAINST', 'NEUTRAL'],
            ax=axes[1, 0]
        )
        axes[1, 0].set_title(f'Confusion Matrix - {best_model}\n(Best Model)', fontweight='bold')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # 4. Ranking summary
        axes[1, 1].axis('off')
        
        # Create ranking text
        ranking_text = "MODEL RANKING\n" + "="*40 + "\n\n"
        for i, row in metrics_df.iterrows():
            ranking_text += f"{i+1}. {row['model_name']}\n"
            ranking_text += f"   Macro-F1: {row['macro_f1']:.3f}\n"
            ranking_text += f"   Accuracy: {row['accuracy']:.3f}\n"
            ranking_text += f"   Cohen's κ: {row['cohen_kappa']:.3f}\n\n"
        
        axes[1, 1].text(
            0.1, 0.9, ranking_text,
            transform=axes[1, 1].transAxes,
            fontfamily='monospace',
            fontsize=10,
            verticalalignment='top'
        )
        
        plt.tight_layout()
        return fig
    
    def get_best_model(self, criterion: str = 'macro_f1') -> str:
        """
        Get name of best model based on criterion.
        
        Parameters
        ----------
        criterion : str
            Metric to use for selection (default: macro_f1)
            
        Returns
        -------
        str
            Name of best model
        """
        if not self.metrics:
            raise ValueError("No model results yet. Run evaluate_model() first.")
        
        best_model = max(
            self.metrics.items(),
            key=lambda x: x[1][criterion]
        )[0]
        
        return best_model
    
    def export_results(self, output_path: str):
        """
        Export all results to files.
        
        Parameters
        ----------
        output_path : str or Path
            Directory to save results
        """
        from pathlib import Path
        import json
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics summary
        metrics_df = self.get_metrics_summary()
        metrics_df.to_csv(output_path / 'model_comparison_metrics.csv', index=False)
        
        # Save detailed results for each model
        for model_name, results_df in self.results.items():
            filename = f"{model_name.lower().replace(' ', '_')}_predictions.csv"
            results_df.to_csv(output_path / filename, index=False)
        
        # Save metrics as JSON
        with open(output_path / 'model_metrics_detailed.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save comparison plot
        fig = self.plot_comparison()
        fig.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Results exported to {output_path}")
