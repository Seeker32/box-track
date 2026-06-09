import unittest

import numpy as np

from src.matching.similarity import compute_tracklet_similarity
from src.utils.tracklet import Tracklet


def make_tracklet(
    camera_id: int,
    local_id: int,
    start_time: float,
    end_time: float,
    feature: np.ndarray,
    cls_id: int = 0,
) -> Tracklet:
    tracklet = Tracklet(
        camera_id=camera_id,
        local_id=local_id,
        start_time=start_time,
        end_time=end_time,
        cls_id=cls_id,
    )
    tracklet.aggregated_feature = feature
    return tracklet


class MatchingSimilarityTests(unittest.TestCase):
    def test_cross_camera_match_does_not_require_time_order(self):
        feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        earlier_finishing_tracklet = make_tracklet(0, 1, 10.0, 20.0, feature)
        earlier_starting_tracklet = make_tracklet(1, 2, 5.0, 8.0, feature)

        similarity = compute_tracklet_similarity(
            earlier_finishing_tracklet,
            earlier_starting_tracklet,
            {"similarity_threshold": 0.5, "min_transit": 1.0, "max_transit": 60.0},
        )

        self.assertAlmostEqual(similarity, 1.0)

    def test_same_camera_tracklets_are_not_cross_camera_matches(self):
        feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tracklet_a = make_tracklet(0, 1, 0.0, 1.0, feature)
        tracklet_b = make_tracklet(0, 2, 0.0, 1.0, feature)

        similarity = compute_tracklet_similarity(tracklet_a, tracklet_b, {})

        self.assertEqual(similarity, 0.0)

    def test_class_gate_still_rejects_different_classes(self):
        feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tracklet_a = make_tracklet(0, 1, 0.0, 1.0, feature, cls_id=0)
        tracklet_b = make_tracklet(1, 2, 0.0, 1.0, feature, cls_id=1)

        similarity = compute_tracklet_similarity(
            tracklet_a,
            tracklet_b,
            {"class_gate_enabled": True},
        )

        self.assertEqual(similarity, 0.0)


if __name__ == "__main__":
    unittest.main()
