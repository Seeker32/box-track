"""Similarity computation for cross-camera tracklet matching."""

import logging

import numpy as np

from src.utils.tracklet import Tracklet

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a, b: Feature vectors (assumed already L2-normalized or not).

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_backbone_similarity(t_a: Tracklet, t_b: Tracklet) -> float:
    """Compute backbone feature similarity between two tracklets.

    Args:
        t_a, t_b: Tracklets with aggregated_feature set.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if features are missing.
    """
    if t_a.aggregated_feature is None or t_b.aggregated_feature is None:
        return 0.0
    return cosine_similarity(t_a.aggregated_feature, t_b.aggregated_feature)


def compute_spatiotemporal_score(
    t_a: Tracklet, t_b: Tracklet, config: dict
) -> float:
    """Compute spatiotemporal plausibility score between two tracklets.

    In Phase 1, this is a hard binary gate:
    - Must be from different cameras
    - t_b must start after t_a ends (t_a is the earlier tracklet)
    - Time gap must be within [min_transit, max_transit]

    Args:
        t_a: Earlier tracklet (assumed to appear first).
        t_b: Later tracklet (assumed to appear second).
        config: Matching configuration dict with keys:
            - min_transit (float): Minimum valid transit time (seconds)
            - max_transit (float): Maximum valid transit time (seconds)
            - soft_time_score (bool): If True, use Gaussian soft score
            - expected_transit (float): Expected transit time for Gaussian
            - time_sigma (float): Gaussian sigma

    Returns:
        1.0 if constraints are satisfied (or soft Gaussian score),
        0.0 if constraints are violated.
    """
    # Must be from different cameras
    if t_a.camera_id == t_b.camera_id:
        return 0.0

    # Time direction: t_a must end before t_b starts
    time_gap = t_b.start_time - t_a.end_time
    if time_gap < 0:
        return 0.0

    # Time window constraint
    min_transit = config.get("min_transit", 1.0)
    max_transit = config.get("max_transit", 60.0)

    if time_gap < min_transit or time_gap > max_transit:
        return 0.0

    # Optional soft Gaussian score
    if config.get("soft_time_score", False):
        expected = config.get("expected_transit", 15.0)
        sigma = config.get("time_sigma", 10.0)
        return float(np.exp(-((time_gap - expected) ** 2) / (2 * sigma ** 2)))

    return 1.0


def compute_tracklet_similarity(
    t_a: Tracklet, t_b: Tracklet, config: dict
) -> float:
    """Compute combined similarity between two tracklets.

    Spatiotemporal constraints act as a hard gate: if the constraints are
    violated, similarity is 0.0 regardless of feature similarity.

    Args:
        t_a: Earlier tracklet.
        t_b: Later tracklet.
        config: Matching configuration dict.

    Returns:
        Similarity score in [0, 1].
    """
    # Spatiotemporal gate
    st_score = compute_spatiotemporal_score(t_a, t_b, config)
    if st_score <= 0.0:
        return 0.0

    # Backbone feature similarity (primary signal)
    backbone_sim = compute_backbone_similarity(t_a, t_b)

    # Clamp negative cosine to 0 (features should be positively correlated)
    return max(0.0, backbone_sim)
