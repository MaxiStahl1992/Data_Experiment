"""
Polarization measurement and aggregation utilities.

This module implements the operationalized polarization pipeline:
- Esteban-Ray polarization calculation
- Stance distribution aggregation (topic-weighted)
- Quality metrics and validation
"""

from .esteban_ray import (
    esteban_ray_polarization,
    normalize_esteban_ray,
    compute_normalization_constant,
    polarization_sensitivity_analysis
)

from .aggregation import (
    aggregate_stance_distribution,
    topic_weighted_stance_shares
)

__all__ = [
    'esteban_ray_polarization',
    'normalize_esteban_ray',
    'compute_normalization_constant',
    'polarization_sensitivity_analysis',
    'aggregate_stance_distribution',
    'topic_weighted_stance_shares'
]
