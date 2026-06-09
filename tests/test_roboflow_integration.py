import unittest

import numpy as np


class RoboflowDetectorTests(unittest.TestCase):
    def test_detect_passes_numpy_frame_to_inference_sdk(self):
        from src.detection.roboflow_detector import RoboflowDetector

        class FakeClient:
            def __init__(self):
                self.inference_input = None
                self.model_id = None

            def infer(self, inference_input, model_id=None):
                self.inference_input = inference_input
                self.model_id = model_id
                return {"predictions": []}

        client = FakeClient()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detector = RoboflowDetector(client=client, model_id="project/1")

        bboxes, raw = detector.detect(frame, frame_id=3, timestamp=0.5)

        self.assertEqual(bboxes, [])
        self.assertEqual(raw, {"predictions": []})
        self.assertIs(client.inference_input, frame)
        self.assertEqual(client.model_id, "project/1")

    def test_predictions_convert_to_filtered_bboxes(self):
        from src.detection.roboflow_detector import RoboflowDetector

        predictions = [
            {
                "class": "cardboard",
                "confidence": 0.81,
                "x": 50,
                "y": 40,
                "width": 20,
                "height": 10,
            },
            {
                "class": "cardboard",
                "confidence": 0.2,
                "x": 10,
                "y": 10,
                "width": 4,
                "height": 4,
            },
            {
                "class": "person",
                "confidence": 0.99,
                "x": 30,
                "y": 30,
                "width": 8,
                "height": 8,
            },
        ]

        bboxes = RoboflowDetector.predictions_to_bboxes(
            predictions,
            frame_id=12,
            timestamp=1.5,
            target_class="cardboard",
            conf=0.5,
            class_id=3,
        )

        self.assertEqual(len(bboxes), 1)
        bbox = bboxes[0]
        self.assertEqual((bbox.x1, bbox.y1, bbox.x2, bbox.y2), (40.0, 35.0, 60.0, 45.0))
        self.assertEqual(bbox.conf, 0.81)
        self.assertEqual(bbox.cls_id, 3)
        self.assertEqual(bbox.frame_id, 12)
        self.assertEqual(bbox.timestamp, 1.5)


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame, frame_id=0, timestamp=0.0):
        from src.utils.tracklet import BBox

        self.calls += 1
        if self.calls == 1:
            return [
                BBox(0, 0, 10, 10, 0.9, 0, frame_id=frame_id, timestamp=timestamp),
                BBox(20, 20, 30, 30, 0.8, 0, frame_id=frame_id, timestamp=timestamp),
            ], []
        return [], []


class FakeTrackedDetections:
    def __init__(self, xyxy, confidence, class_id, tracker_id):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.confidence = np.asarray(confidence, dtype=float)
        self.class_id = np.asarray(class_id, dtype=int)
        self.tracker_id = np.asarray(tracker_id, dtype=int)


class FakeByteTracker:
    def __init__(self):
        self.calls = 0

    def update(self, detections, frame=None):
        self.calls += 1
        if self.calls == 1:
            return FakeTrackedDetections(
                xyxy=[[0, 0, 10, 10], [20, 20, 30, 30]],
                confidence=[0.9, 0.8],
                class_id=[0, 0],
                tracker_id=[7, -1],
            )
        return FakeTrackedDetections(xyxy=[], confidence=[], class_id=[], tracker_id=[])


class FakeFeatureExtractor:
    feature_dim = 4

    def extract(self, frame, bboxes):
        return [np.ones(self.feature_dim, dtype=np.float32) for _ in bboxes]


class RoboflowByteTrackTrackerTests(unittest.TestCase):
    def test_skips_unmatched_tracker_ids_and_finalizes_lost_tracklets(self):
        from src.tracking.roboflow_bytetrack_tracker import RoboflowByteTrackTracker

        tracker = RoboflowByteTrackTracker(
            camera_id=2,
            detector=FakeDetector(),
            byte_tracker=FakeByteTracker(),
            feature_extractor=FakeFeatureExtractor(),
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        active = tracker.process_frame(frame, frame_id=1, timestamp=0.1)
        self.assertEqual(active, [7])
        self.assertEqual(list(tracker.active_tracks.keys()), [7])

        active = tracker.process_frame(frame, frame_id=2, timestamp=0.2)
        self.assertEqual(active, [])
        self.assertEqual(len(tracker.completed_tracklets), 1)
        completed = tracker.completed_tracklets[0]
        self.assertEqual(completed.camera_id, 2)
        self.assertEqual(completed.local_id, 7)
        self.assertEqual(completed.frames, [1])
        self.assertEqual(completed.cls_id, 0)
        self.assertIsNotNone(completed.aggregated_feature)


class TrackerFactoryTests(unittest.TestCase):
    def test_roboflow_backend_uses_yolo26n_feature_model(self):
        from src.tracking.factory import create_tracker
        from src.tracking.roboflow_bytetrack_tracker import RoboflowByteTrackTracker

        tracker = create_tracker(
            camera_id=0,
            config={
                "tracker_backend": "roboflow_bytetrack",
                "feature": {"model_path": "models/yolo26n.pt"},
            },
            detector=FakeDetector(),
            byte_tracker=FakeByteTracker(),
            feature_extractor=FakeFeatureExtractor(),
        )

        self.assertIsInstance(tracker, RoboflowByteTrackTracker)


if __name__ == "__main__":
    unittest.main()
