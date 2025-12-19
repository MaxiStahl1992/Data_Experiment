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

__all__ = [
    'StanceModel',
    'TargetRepresentation',
    'predict_stance',
    'batch_predict_stance',
    'apply_confidence_threshold',
    'optimize_threshold',
    'confidence_coverage_tradeoff',
    'compute_stance_metrics',
    'compute_macro_f1',
    'inter_annotator_agreement'
]
