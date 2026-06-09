"""Roboflow detection + ByteTrack single-camera tracker."""

import logging
from typing import Any

import numpy as np
import supervision as sv

from src.detection.roboflow_detector import RoboflowDetector
from src.features.backbone_feature import YOLOBackboneFeatureExtractor
from src.utils.tracklet import BBox, Tracklet

logger = logging.getLogger(__name__)


class RoboflowByteTrackTracker:
    """Single-camera tracker using Roboflow detections and ByteTrack IDs."""

    def __init__(
        self,
        camera_id: int,
        detector: RoboflowDetector | None = None,
        byte_tracker: Any | None = None,
        feature_extractor: YOLOBackboneFeatureExtractor | None = None,
        roboflow_cfg: dict[str, Any] | None = None,
        feature_model_path: str = "models/yolo26n.pt",
        feature_hook_layer: int = -2,
        conf: float = 0.7,
        tracker_cfg: dict[str, Any] | None = None,
    ):
        self.camera_id = camera_id
        self.detector = detector or self._create_detector(roboflow_cfg or {}, conf)
        self.byte_tracker = byte_tracker or self._create_byte_tracker(tracker_cfg or {})
        self.feature_extractor = feature_extractor or YOLOBackboneFeatureExtractor(
            model_path=feature_model_path,
            hook_layer=feature_hook_layer,
            normalize=True,
        )
        self.active_tracks: dict[int, Tracklet] = {}
        self.completed_tracklets: list[Tracklet] = []

    @staticmethod
    def _create_detector(config: dict[str, Any], conf: float) -> RoboflowDetector:
        return RoboflowDetector(
            api_url=config.get("api_url", "https://serverless.roboflow.com"),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env", "API_KEY"),
            model_id=config.get("model_id", "box-detection-sz4gh-dum2a/2"),
            target_class=config.get("target_class", "cardboard"),
            conf=config.get("conf", conf),
            class_id=config.get("class_id", 0),
        )

    @staticmethod
    def _create_byte_tracker(config: dict[str, Any]):
        try:
            from trackers import ByteTrackTracker
        except ImportError as exc:
            raise ImportError(
                "Roboflow ByteTrack backend requires the 'trackers' package. "
                "Install project dependencies after adding trackers to pyproject.toml."
            ) from exc

        return ByteTrackTracker(**config)

    def process_frame(
        self, frame: np.ndarray, frame_id: int, timestamp: float
    ) -> list[int]:
        """Process a frame and update local tracklets."""
        bboxes, _ = self.detector.detect(frame, frame_id=frame_id, timestamp=timestamp)
        detections = self._bboxes_to_detections(bboxes)
        tracked = self.byte_tracker.update(detections)
        tracked_data = self._tracked_detections_to_data(tracked, frame_id, timestamp)

        tracked_bboxes = [bbox for bbox, _ in tracked_data]
        features = self.feature_extractor.extract(frame, tracked_bboxes)
        if len(features) != len(tracked_data):
            logger.warning(
                "Feature count mismatch: %d features for %d tracked detections",
                len(features),
                len(tracked_data),
            )
            while len(features) < len(tracked_data):
                features.append(
                    np.zeros(self.feature_extractor.feature_dim, dtype=np.float32)
                )

        active_ids: set[int] = set()
        for (bbox, track_id), feature in zip(tracked_data, features):
            active_ids.add(track_id)
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = Tracklet(
                    camera_id=self.camera_id,
                    local_id=track_id,
                    start_time=timestamp,
                    end_time=timestamp,
                )

            tracklet = self.active_tracks[track_id]
            tracklet.frames.append(frame_id)
            tracklet.bboxes.append(bbox)
            tracklet.features.append(feature)
            tracklet.end_time = timestamp

        self._finalize_lost_tracks(active_ids)
        return sorted(active_ids)

    @staticmethod
    def _bboxes_to_detections(bboxes: list[BBox]) -> sv.Detections:
        if not bboxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.asarray(
                [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes],
                dtype=np.float32,
            ),
            confidence=np.asarray([bbox.conf for bbox in bboxes], dtype=np.float32),
            class_id=np.asarray([bbox.cls_id for bbox in bboxes], dtype=int),
        )

    @staticmethod
    def _tracked_detections_to_data(
        detections: sv.Detections,
        frame_id: int,
        timestamp: float,
    ) -> list[tuple[BBox, int]]:
        if detections.tracker_id is None:
            return []

        confidence = detections.confidence
        if confidence is None:
            confidence = np.ones(len(detections.xyxy), dtype=np.float32)
        class_id = detections.class_id
        if class_id is None:
            class_id = np.zeros(len(detections.xyxy), dtype=int)

        tracked_data: list[tuple[BBox, int]] = []
        for xyxy, conf, cls_id, track_id in zip(
            detections.xyxy,
            confidence,
            class_id,
            detections.tracker_id,
        ):
            track_id_int = int(track_id)
            if track_id_int < 0:
                continue

            x1, y1, x2, y2 = [float(value) for value in xyxy]
            tracked_data.append(
                (
                    BBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        conf=float(conf),
                        cls_id=int(cls_id),
                        frame_id=frame_id,
                        timestamp=timestamp,
                    ),
                    track_id_int,
                )
            )

        return tracked_data

    def _finalize_lost_tracks(self, active_ids: set[int]) -> None:
        lost_ids = [tid for tid in self.active_tracks if tid not in active_ids]
        for tid in lost_ids:
            tracklet = self.active_tracks.pop(tid)
            if tracklet.features:
                tracklet.aggregate_features()
                tracklet.infer_majority_class()
                self.completed_tracklets.append(tracklet)

    def flush(self) -> None:
        for tid in list(self.active_tracks.keys()):
            tracklet = self.active_tracks.pop(tid)
            if tracklet.features:
                tracklet.aggregate_features()
                tracklet.infer_majority_class()
                self.completed_tracklets.append(tracklet)

    @property
    def num_active(self) -> int:
        return len(self.active_tracks)

    @property
    def num_completed(self) -> int:
        return len(self.completed_tracklets)
