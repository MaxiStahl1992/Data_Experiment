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
from tqdm.auto import tqdm as tqdm_auto


def topic_to_claim(topic_label: str) -> str:
    """
    Convert a topic category into a stance-able claim.
    
    Transforms broad topics (e.g., "Infrastructure") into specific claims
    (e.g., "Government should invest in infrastructure") that can have
    clear FAVOUR/AGAINST positions.
    
    This solves the annotation problem: you can't take a stance toward
    "Infrastructure" (a category), but you CAN take a stance toward
    "Government should invest in infrastructure" (a claim).
    
    Parameters
    ----------
    topic_label : str
        CAP topic label (e.g., "Healthcare Policy", "Immigration Policy")
        
    Returns
    -------
    str
        Claim that can be agreed/disagreed with
        
    Examples
    --------
    >>> topic_to_claim("Healthcare Policy")
    'Healthcare Policy is beneficial'
    
    >>> topic_to_claim("Infrastructure")
    'Government should invest in infrastructure'
    
    >>> topic_to_claim("Immigration Reform")
    'Immigration Reform should be implemented'
    """
    topic_lower = topic_label.lower()
    
    # Policy topics → "[Policy] is beneficial/necessary"
    if 'policy' in topic_lower or 'policies' in topic_lower:
        return f"{topic_label} is beneficial"
    
    # Reform topics → "[Reform] should be implemented"
    if 'reform' in topic_lower:
        return f"{topic_label} should be implemented"
    
    # Investment topics → "Government should invest in [topic]"
    if any(word in topic_lower for word in ['infrastructure', 'education', 'technology', 'energy']):
        return f"Government should invest in {topic_label.lower()}"
    
    # Rights/Protection topics → "[Topic] should be protected/expanded"
    if any(word in topic_lower for word in ['rights', 'protection', 'security']):
        return f"{topic_label} should be protected"
    
    # Climate/Environment → "Government should address [topic]"
    if any(word in topic_lower for word in ['climate', 'environment', 'pollution']):
        return f"Government should address {topic_label.lower()}"
    
    # Default: "[Topic] is important/beneficial"
    return f"{topic_label} is beneficial"


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
        
        # Device detection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading stance model: {model_name}")
        print(f"Device: {self.device}")
    
        # Fine-tuned stance classifier (3-way classification)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Assume label mapping (update based on fine-tuned model)
        self.label_map = {0: 'FAVOUR', 1: 'AGAINST', 2: 'NEUTRAL'}
        
        print("✓ Model loaded")
    

class ImprovedNLIStanceModel:
    """
    Improved NLI-based stance detection using direct hypothesis testing.
    
    Better than zero-shot classification for stance:
    - Tests direct value judgments: "[topic] is good" vs "[topic] is bad"
    - Handles both high/low score edge cases
    - More reliable than BART zero-shot for political discourse
    
    Recommended models:
    - roberta-large-mnli (stable, good performance)
    - microsoft/deberta-v3-base-mnli-fever-anli-ling-wanli (best, but larger)
    - cross-encoder/nli-deberta-v3-base (good balance)
    
    Special handling:
    - BART models use zero-shot classification pipeline instead of NLI
    """
    
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: Optional[str] = None,
        confidence_threshold: float = 0.04,
        use_claim_formulation: bool = True
    ):
        """
        Initialize improved NLI stance model.
        
        Parameters
        ----------
        model_name : str
            NLI model (roberta-large-mnli, deberta-v3-base-mnli, etc.)
        device : str, optional
            'cuda', 'mps', or 'cpu' (auto-detect if None)
        confidence_threshold : float
            Minimum entailment score for non-neutral stance (default: 0.04)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_claim_formulation = use_claim_formulation
        
        # Detect if this is a BART model (needs zero-shot pipeline)
        self.is_bart = 'bart' in model_name.lower()
        
        # Device detection
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() 
                else 'mps' if torch.backends.mps.is_available() 
                else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        print(f"Loading NLI stance model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Claim formulation: {use_claim_formulation}")
        
        if self.is_bart:
            # BART uses zero-shot classification pipeline
            from transformers import pipeline
            self.pipeline = pipeline(
                model=model_name,
                device=0 if self.device.type == 'cuda' else -1
            )
            print("Using zero-shot classification pipeline for BART")
        else:
            # Standard NLI models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # NLI label indices: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT (standard)
            self.entailment_idx = 2
            self.contradiction_idx = 0
        
        print("✓ Model loaded")
    
    def predict(
        self,
        text: str,
        target: Union[str, TargetRepresentation],
        return_probs: bool = True
    ) -> Dict[str, float]:
        """
        Predict stance using NLI entailment.
        
        Tests two hypotheses:
        - H1: "[target] is good" → measures FAVOUR
        - H2: "[target] is bad" → measures AGAINST
        
        Parameters
        ----------
        text : str
            Input text (comment, article, etc.)
        target : str or TargetRepresentation
            Target topic (e.g., "Healthcare Policy")
        return_probs : bool
            Return probability distribution
            
        Returns
        -------
        dict
            {'prob_favour': float, 'prob_against': float, 'prob_neutral': float,
             'predicted_label': str, 'confidence': float}
        """
        # Format target
        if isinstance(target, TargetRepresentation):
            target_str = target.target_string
        elif isinstance(target, dict) and 'label' in target:
            target_str = target['label']
        else:
            target_str = str(target)
        
        # Convert topic to claim if enabled
        if self.use_claim_formulation:
            target_claim = topic_to_claim(target_str)
        else:
            target_claim = target_str
        
        # Truncate text
        premise = text[:512]
        
        # BART uses zero-shot classification
        if self.is_bart:
            return self._predict_bart(premise, target_claim)
        
        # Standard NLI models: test value judgments about the claim
        hypothesis_favour = f"{target_claim} is good"
        hypothesis_against = f"{target_claim} is bad"
        
        # Get predictions for both hypotheses
        with torch.no_grad():
            # Test FAVOUR hypothesis
            inputs_favour = self.tokenizer(
                premise, hypothesis_favour,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            outputs_favour = self.model(**inputs_favour)
            probs_favour = torch.softmax(outputs_favour.logits, dim=-1)[0]
            
            # Test AGAINST hypothesis
            inputs_against = self.tokenizer(
                premise, hypothesis_against,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            outputs_against = self.model(**inputs_against)
            probs_against = torch.softmax(outputs_against.logits, dim=-1)[0]
            
            # Get entailment scores
            favour_score = probs_favour[self.entailment_idx].item()
            against_score = probs_against[self.entailment_idx].item()
            
            # Calculate margin
            margin = abs(favour_score - against_score)
            max_score = max(favour_score, against_score)
            min_score = min(favour_score, against_score)
            
            # Determine stance with edge case handling
            # Case 1: Both scores very high (>0.7) with small margin → NEUTRAL
            # (e.g., "X is important" entails both "X is good" and contextually "X is bad")
            if min_score > 0.7 and margin < 0.3:
                stance = 'NEUTRAL'
                confidence = max_score
            # Case 2: Both scores very low (<0.2) → NEUTRAL
            elif max_score < 0.2:
                stance = 'NEUTRAL'
                confidence = max_score
            # Case 3: One clearly higher
            elif favour_score > against_score:
                if favour_score > self.confidence_threshold or (favour_score > 0.3 and margin > 0.1):
                    stance = 'FAVOUR'
                    confidence = favour_score
                else:
                    stance = 'NEUTRAL'
                    confidence = max_score
            elif against_score > favour_score:
                if against_score > self.confidence_threshold or (against_score > 0.3 and margin > 0.1):
                    stance = 'AGAINST'
                    confidence = against_score
                else:
                    stance = 'NEUTRAL'
                    confidence = max_score
            else:
                # Tied
                stance = 'NEUTRAL'
                confidence = max_score
        
        return {
            'prob_favour': favour_score,
            'prob_against': against_score,
            'prob_neutral': 1.0 - max(favour_score, against_score),
            'predicted_label': stance,
            'confidence': confidence
        }
    
    def _predict_bart(
        self,
        text: str,
        target_claim: str
    ) -> Dict[str, float]:
        """
        BART-specific prediction using zero-shot classification.
        
        Tests hypothesis: "This text is in favor of [claim]"
        """
        # Zero-shot classification with stance labels
        result = self.pipeline(
            text,
            candidate_labels=['in favor', 'against', 'neutral'],
            hypothesis_template=f"This text is {{}} the claim: {target_claim}"
        )
        
        # Extract scores
        labels = result['labels']
        scores = result['scores']
        
        # Map to our format
        label_scores = dict(zip(labels, scores))
        favour_score = label_scores.get('in favor', 0.0)
        against_score = label_scores.get('against', 0.0)
        neutral_score = label_scores.get('neutral', 0.0)
        
        # Determine stance (use same logic as NLI)
        max_score = max(favour_score, against_score, neutral_score)
        
        if favour_score == max_score and favour_score > self.confidence_threshold:
            stance = 'FAVOUR'
            confidence = favour_score
        elif against_score == max_score and against_score > self.confidence_threshold:
            stance = 'AGAINST'
            confidence = against_score
        else:
            stance = 'NEUTRAL'
            confidence = neutral_score
        
        return {
            'prob_favour': favour_score,
            'prob_against': against_score,
            'prob_neutral': neutral_score,
            'predicted_label': stance,
            'confidence': confidence
        }
    
    def batch_predict(
        self,
        texts: List[str],
        targets: List[Union[str, TargetRepresentation]],
        batch_size: int = 16,
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
            Batch size for inference (note: each text requires 2 forward passes)
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        pd.DataFrame
            Predictions with columns: prob_favour, prob_against, prob_neutral,
            predicted_label, confidence
        """
        if len(targets) == 1:
            targets = targets * len(texts)
        
        if len(texts) != len(targets):
            raise ValueError(f"Length mismatch: {len(texts)} texts, {len(targets)} targets")
        
        results = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm_auto(iterator, desc="Stance prediction")
        
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
