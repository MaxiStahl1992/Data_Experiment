"""
Topic modeling utilities.

Provides topic model fitting, quality assessment, and stability analysis
following the operationalization spec.
"""

from .stm_utils import (
    fit_topic_model,
    get_topic_descriptors,
    assign_topics_to_documents,
    TopicModelResults
)

from .coherence import (
    compute_topic_coherence,
    compute_coherence_scores,
    filter_by_coherence_threshold
)

from .stability import (
    fit_multiple_runs,
    align_topics_across_runs,
    compute_stability_scores,
    StabilityResults
)

__all__ = [
    'fit_topic_model',
    'get_topic_descriptors',
    'assign_topics_to_documents',
    'TopicModelResults',
    'compute_topic_coherence',
    'compute_coherence_scores',
    'filter_by_coherence_threshold',
    'fit_multiple_runs',
    'align_topics_across_runs',
    'compute_stability_scores',
    'StabilityResults'
]
