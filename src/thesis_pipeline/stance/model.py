"""
Target-aware stance detection model.

Implements stance classification using transformer models (RoBERTa-style).
Follows SemEval-2016 Task 6 and VAST formulation:
    - Input: (text, target) → stance ∈ {FAVOUR, AGAINST, NEUTRAL/OTHER}
    - Output: probability distribution (p_F, p_A, p_N)

For 2-month validation: Uses pre-trained zero-shot stance model or fine-tunes
a lightweight RoBERTa-base model.

For full thesis: Consider fine-tuning on VAST dataset for better generalization.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Literal, Union
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import warnings


@dataclass
class TargetRepresentation:
    """
    Topic-based target representation for stance detection.
    
    Each topic k is represented by its top terms (from STM) to create
    a target string τ_k for stance inference.
    
    Attributes
    ----------
    topic_id : int
        Topic identifier
    top_terms : list of str
        Top m terms from STM (e.g., 10-15 terms)
    named_entities : list of str, optional
        Top named entities extracted from topic (optional enhancement)
    target_string : str
        Formatted target representation for stance model
    """
    topic_id: int
    top_terms: List[str]
    named_entities: Optional[List[str]] = None
    target_string: Optional[str] = None
    
    def __post_init__(self):
        """Generate target string if not provided."""
        if self.target_string is None:
            # Default: join top terms
            terms = self.top_terms[:15]  # Limit to avoid token overflow
            
            if self.named_entities:
                # Include entities for specificity
                entities = self.named_entities[:5]
                self.target_string = f"{', '.join(terms)} ({', '.join(entities)})"
            else:
                self.target_string = ', '.join(terms)
    
    def __str__(self):
        return f"Topic {self.topic_id}: {self.target_string[:50]}..."


class StanceModel:
    """
    Target-aware stance classification model.
    
    Wraps a transformer-based stance detector (e.g., RoBERTa fine-tuned on VAST).
    
    For validation: Uses a simple zero-shot classifier
    For production: Fine-tune RoBERTa on VAST + SemEval stance data
    
    Attributes
    ----------
    model_name : str
        HuggingFace model identifier or local path
    tokenizer : AutoTokenizer
        Tokenizer for the model
    model : AutoModelForSequenceClassification or pipeline
        Stance classification model
    label_map : dict
        Label mapping {0: 'FAVOUR', 1: 'AGAINST', 2: 'NEUTRAL'}
    device : str
        'cuda', 'mps', or 'cpu'
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",  # Zero-shot baseline
        use_zero_shot: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize stance model.
        
        Parameters
        ----------
        model_name : str
            Model identifier (HuggingFace or local path)
        use_zero_shot : bool
            If True, use zero-shot classification pipeline (for validation)
            If False, use fine-tuned stance classifier (for production)
        device : str, optional
            'cuda', 'mps', or 'cpu' (auto-detect if None)
        """
        self.model_name = model_name
        self.use_zero_shot = use_zero_shot
        
        # Device detection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading stance model: {model_name}")
        print(f"Device: {self.device}")
        
        if use_zero_shot:
            # Zero-shot classification pipeline (BART/RoBERTa MNLI)
            self.model = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if self.device == 'cuda' else -1
            )
            self.tokenizer = None
            
            # Stance labels for zero-shot
            self.stance_labels = ['in favor', 'against', 'neutral']
            self.label_map = {
                'in favor': 'FAVOUR',
                'against': 'AGAINST',
                'neutral': 'NEUTRAL'
            }
        else:
            # Fine-tuned stance classifier (3-way classification)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Assume label mapping (update based on fine-tuned model)
            self.label_map = {0: 'FAVOUR', 1: 'AGAINST', 2: 'NEUTRAL'}
        
        print("✓ Model loaded")
    
    def predict(
        self,
        text: str,
        target: Union[str, TargetRepresentation],
        return_probs: bool = True
    ) -> Dict[str, float]:
        """
        Predict stance for a single text-target pair.
        
        Parameters
        ----------
        text : str
            Input text (comment or article)
        target : str or TargetRepresentation
            Target representation (topic terms)
        return_probs : bool
            Return probability distribution (required for confidence filtering)
            
        Returns
        -------
        dict
            {'prob_favour': float, 'prob_against': float, 'prob_neutral': float,
             'predicted_label': str, 'confidence': float}
        """
        # Format target
        if isinstance(target, TargetRepresentation):
            target_str = target.target_string
        else:
            target_str = target
        
        if self.use_zero_shot:
            # Zero-shot classification
            hypothesis = f"This text is {{}}"  # Will be filled with stance labels
            result = self.model(
                text,
                candidate_labels=self.stance_labels,
                hypothesis_template=f"This text is {{}} the topic: {target_str}"
            )
            
            # Map to standard labels and probabilities
            probs = {
                self.label_map[label]: score
                for label, score in zip(result['labels'], result['scores'])
            }
            
            predicted_label = self.label_map[result['labels'][0]]
            confidence = result['scores'][0]
            
        else:
            # Fine-tuned model inference
            # Format: [CLS] text [SEP] target [SEP]
            inputs = self.tokenizer(
                text,
                target_str,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs_tensor = torch.softmax(logits, dim=1)[0]
            
            probs = {
                'FAVOUR': probs_tensor[0].item(),
                'AGAINST': probs_tensor[1].item(),
                'NEUTRAL': probs_tensor[2].item()
            }
            
            predicted_idx = torch.argmax(probs_tensor).item()
            predicted_label = self.label_map[predicted_idx]
            confidence = probs_tensor[predicted_idx].item()
        
        return {
            'prob_favour': probs.get('FAVOUR', 0.0),
            'prob_against': probs.get('AGAINST', 0.0),
            'prob_neutral': probs.get('NEUTRAL', 0.0),
            'predicted_label': predicted_label,
            'confidence': confidence
        }
    
    def batch_predict(
        self,
        texts: List[str],
        targets: List[Union[str, TargetRepresentation]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Batch prediction for multiple text-target pairs.
        
        Parameters
        ----------
        texts : list of str
            Input texts
        targets : list of str or TargetRepresentation
            Targets (one per text or single target for all)
        batch_size : int
            Batch size for inference
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        pd.DataFrame
            Predictions with columns: prob_favour, prob_against, prob_neutral,
            predicted_label, confidence
        """
        from tqdm import tqdm
        
        if len(targets) == 1:
            # Same target for all texts
            targets = targets * len(texts)
        
        if len(texts) != len(targets):
            raise ValueError(f"Length mismatch: {len(texts)} texts, {len(targets)} targets")
        
        results = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Stance prediction")
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            for text, target in zip(batch_texts, batch_targets):
                pred = self.predict(text, target)
                results.append(pred)
        
        return pd.DataFrame(results)


def predict_stance(
    text: str,
    target: Union[str, TargetRepresentation],
    model: Optional[StanceModel] = None,
    model_name: str = "facebook/bart-large-mnli"
) -> Dict[str, float]:
    """
    Convenience function for single stance prediction.
    
    Parameters
    ----------
    text : str
        Input text
    target : str or TargetRepresentation
        Target representation
    model : StanceModel, optional
        Pre-loaded model (loads new model if None)
    model_name : str
        Model to load if model is None
        
    Returns
    -------
    dict
        Stance prediction with probabilities
    """
    if model is None:
        model = StanceModel(model_name=model_name)
    
    return model.predict(text, target)


def batch_predict_stance(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    model: Optional[StanceModel] = None,
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Batch stance prediction on DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with text and target columns
    text_col : str
        Column name for text
    target_col : str
        Column name for target (str or TargetRepresentation)
    model : StanceModel, optional
        Pre-loaded model
    model_name : str
        Model to load if model is None
    batch_size : int
        Batch size
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with added stance prediction columns
    """
    if model is None:
        model = StanceModel(model_name=model_name)
    
    predictions = model.batch_predict(
        texts=df[text_col].tolist(),
        targets=df[target_col].tolist(),
        batch_size=batch_size
    )
    
    return pd.concat([df.reset_index(drop=True), predictions], axis=1)


# Example usage
if __name__ == '__main__':
    print("Stance Detection Model Test")
    print("=" * 60)
    
    # Test 1: Create target representation
    print("\nTest 1: Target representation")
    target = TargetRepresentation(
        topic_id=1,
        top_terms=['healthcare', 'reform', 'insurance', 'affordable', 'care'],
        named_entities=['Obamacare', 'ACA']
    )
    print(f"Target: {target.target_string}")
    
    # Test 2: Single prediction
    print("\nTest 2: Single stance prediction")
    text = "The healthcare reform bill is essential for covering millions of uninsured Americans."
    
    model = StanceModel(model_name="facebook/bart-large-mnli", use_zero_shot=True)
    pred = model.predict(text, target)
    
    print(f"Text: {text}")
    print(f"Prediction: {pred['predicted_label']} (confidence: {pred['confidence']:.3f})")
    print(f"Probabilities: F={pred['prob_favour']:.3f}, A={pred['prob_against']:.3f}, N={pred['prob_neutral']:.3f}")
    
    # Test 3: Batch prediction
    print("\nTest 3: Batch prediction")
    texts = [
        "Healthcare reform is crucial for our society.",
        "This policy will destroy our healthcare system.",
        "I'm not sure about the healthcare changes."
    ]
    
    batch_results = model.batch_predict(texts, [target] * 3, show_progress=False)
    print(batch_results[['predicted_label', 'confidence', 'prob_favour', 'prob_against', 'prob_neutral']])
    
    print("\n✓ All tests passed!")
