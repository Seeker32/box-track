"""Hungarian algorithm-based cross-camera tracklet matching."""

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.matching.similarity import compute_tracklet_similarity
from src.utils.tracklet import Tracklet

logger = logging.getLogger(__name__)


def build_similarity_matrix(
    tracklets_a: list[Tracklet],
    tracklets_b: list[Tracklet],
    config: dict,
) -> np.ndarray:
    """Build a pairwise similarity matrix between two sets of tracklets.

    Args:
        tracklets_a: Source tracklets (earlier camera, rows in matrix).
        tracklets_b: Target tracklets (later camera, columns in matrix).
        config: Matching configuration dict.

    Returns:
        Similarity matrix of shape (len(tracklets_b), len(tracklets_a)).
        Entry [i, j] is the similarity between tracklets_b[i] and tracklets_a[j].
    """
    n_a = len(tracklets_a)
    n_b = len(tracklets_b)

    if n_a == 0 or n_b == 0:
        return np.zeros((n_b, n_a))

    matrix = np.zeros((n_b, n_a))
    for i, t_b in enumerate(tracklets_b):
        for j, t_a in enumerate(tracklets_a):
            matrix[i, j] = compute_tracklet_similarity(t_a, t_b, config)

    return matrix


def cross_camera_match(
    tracklets_a: list[Tracklet],
    tracklets_b: list[Tracklet],
    config: dict,
) -> tuple[dict[int, int], list[int], list[int]]:
    """Match tracklets from camera A (earlier) to camera B (later).

    Uses the Hungarian algorithm to find the optimal assignment that maximizes
    total similarity. Only matches above the similarity threshold are kept.

    Args:
        tracklets_a: Tracklets from the earlier camera (gallery).
        tracklets_b: Tracklets from the later camera (query).
        config: Matching configuration dict with key:
            - similarity_threshold (float): Minimum similarity for a valid match.

    Returns:
        Tuple of:
        - matches: dict mapping {b_index: a_index} for matched pairs
        - unmatched_a: list of indices in tracklets_a that were not matched
        - unmatched_b: list of indices in tracklets_b that were not matched
    """
    n_a = len(tracklets_a)
    n_b = len(tracklets_b)

    if n_a == 0 or n_b == 0:
        return {}, list(range(n_a)), list(range(n_b))

    # Build similarity matrix
    sim = build_similarity_matrix(tracklets_a, tracklets_b, config)

    # Hungarian algorithm minimizes cost; maximize similarity = minimize -similarity
    b_indices, a_indices = linear_sum_assignment(-sim)

    threshold = config.get("similarity_threshold", 0.5)

    matches: dict[int, int] = {}
    matched_a: set[int] = set()
    matched_b: set[int] = set()

    for b_idx, a_idx in zip(b_indices, a_indices):
        if sim[b_idx, a_idx] >= threshold:
            matches[b_idx] = a_idx
            matched_a.add(a_idx)
            matched_b.add(b_idx)
        else:
            logger.debug(
                "Rejected match B[%d]→A[%d]: similarity %.4f < threshold %.2f",
                b_idx, a_idx, sim[b_idx, a_idx], threshold,
            )

    unmatched_a = [i for i in range(n_a) if i not in matched_a]
    unmatched_b = [i for i in range(n_b) if i not in matched_b]

    logger.info(
        "Cross-camera match: %d matched, %d unmatched (A), %d unmatched (B)",
        len(matches), len(unmatched_a), len(unmatched_b),
    )

    return matches, unmatched_a, unmatched_b
