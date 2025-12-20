"""
Stance detection utilities.

Implements target-aware stance classification following the operationalization spec:
- RoBERTa-based stance model (VAST-style)
- Target representation from topic terms
- Confidence filtering with reject option
- Platform-aware input formatting
"""

from .model import (
    StanceModel,
    TargetRepresentation,
    ImprovedNLIStanceModel,
    topic_to_claim,
    predict_stance,
    batch_predict_stance
)

from .filtering import (
    apply_confidence_threshold,
    optimize_threshold,
    confidence_coverage_tradeoff
)

from .validation import (
    compute_stance_metrics,
    compute_macro_f1,
    inter_annotator_agreement
)

from .comparison import (
    StanceModelComparison,
    evaluate_stance_model
)

__all__ = [
    'StanceModel',
    'TargetRepresentation',
    'ImprovedNLIStanceModel',
    'topic_to_claim',
    'predict_stance',
    'batch_predict_stance',
    'apply_confidence_threshold',
    'optimize_threshold',
    'confidence_coverage_tradeoff',
    'compute_stance_metrics',
    'compute_macro_f1',
    'inter_annotator_agreement',
    'StanceModelComparison',
    'evaluate_stance_model'
]
