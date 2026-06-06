"""BoT-SORT single-camera tracker with backbone feature extraction."""

import logging

import numpy as np
from ultralytics import YOLO

from src.features.backbone_feature import YOLOBackboneFeatureExtractor
from src.utils.tracklet import BBox, Tracklet

logger = logging.getLogger(__name__)


class BOTSORTTracker:
    """Single-camera tracker: detection + BoT-SORT tracking + backbone features.

    Processes a video stream frame by frame, running YOLO detection with
    BoT-SORT tracking, extracting backbone features for each detection, and
    accumulating them into Tracklet objects.

    Usage:
        tracker = BOTSORTTracker(camera_id=0)
        for frame_id, frame in enumerate(frames):
            tracker.process_frame(frame, frame_id, timestamp)
        tracker.flush()
        tracklets = tracker.completed_tracklets
    """

    def __init__(
        self,
        camera_id: int,
        model_path: str = "models/best.pt",
        tracker_cfg: str = "configs/botsort.yaml",
        conf: float = 0.25,
        hook_layer: int = -2,
    ):
        """Initialize the tracker.

        Args:
            camera_id: Identifier for this camera.
            model_path: Path to the YOLO model weights.
            tracker_cfg: Path to the BoT-SORT YAML config.
            conf: Detection confidence threshold.
            hook_layer: Neck layer index for feature extraction hook.
        """
        self.camera_id = camera_id
        self.tracker_cfg = tracker_cfg
        self.conf = conf

        # Single YOLO model shared for tracking + feature extraction
        self.model = YOLO(model_path)

        # Feature extractor shares the model (no double loading)
        self.feature_extractor = YOLOBackboneFeatureExtractor(
            model_path=model_path,
            hook_layer=hook_layer,
            normalize=True,
            model=self.model,
        )

        # Track management
        self.active_tracks: dict[int, Tracklet] = {}  # local_id → Tracklet
        self.completed_tracklets: list[Tracklet] = []

    def process_frame(
        self, frame: np.ndarray, frame_id: int, timestamp: float
    ) -> list[int]:
        """Process a single video frame.

        Args:
            frame: Input frame in BGR format (H, W, 3).
            frame_id: Sequential frame number.
            timestamp: Frame timestamp in seconds.

        Returns:
            List of active track IDs in this frame.
        """
        # Step 1: Run tracking — triggers the feature extraction hook
        self.feature_extractor._captured.clear()

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            verbose=False,
        )

        # Step 2: Parse tracked detections
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            self._finalize_lost_tracks(set())
            return []

        boxes = results[0].boxes
        tracked_data: list[tuple[BBox, int]] = []  # (bbox, track_id)

        for box_xyxy, conf_val, cls_id, track_id in zip(
            boxes.xyxy, boxes.conf, boxes.cls, boxes.id
        ):
            x1, y1, x2, y2 = box_xyxy.tolist()
            bbox = BBox(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                conf=float(conf_val.item()),
                cls_id=int(cls_id.item()),
                frame_id=frame_id,
                timestamp=timestamp,
            )
            tracked_data.append((bbox, int(track_id.item())))

        # Step 3: Extract backbone features from the captured feature map
        bboxes_only = [bb for bb, _ in tracked_data]
        features = self.feature_extractor.extract_from_captured(frame, bboxes_only)

        if len(features) != len(tracked_data):
            logger.warning(
                "Feature count mismatch: %d features for %d bboxes",
                len(features), len(tracked_data),
            )
            # Pad or truncate to match
            while len(features) < len(tracked_data):
                features.append(np.zeros(
                    self.feature_extractor.feature_dim, dtype=np.float32
                ))

        # Step 4: Update active tracklets
        active_ids: set[int] = set()
        for (bbox, track_id), feat in zip(tracked_data, features):
            active_ids.add(track_id)

            if track_id not in self.active_tracks:
                # New track
                self.active_tracks[track_id] = Tracklet(
                    camera_id=self.camera_id,
                    local_id=track_id,
                    start_time=timestamp,
                    end_time=timestamp,
                )

            tracklet = self.active_tracks[track_id]
            tracklet.frames.append(frame_id)
            tracklet.bboxes.append(bbox)
            tracklet.features.append(feat)
            tracklet.end_time = timestamp

        # Step 5: Finalize tracks that disappeared this frame
        self._finalize_lost_tracks(active_ids)

        return list(active_ids)

    def _finalize_lost_tracks(self, active_ids: set[int]) -> None:
        """Move tracks not in active_ids from active to completed."""
        lost_ids = [tid for tid in self.active_tracks if tid not in active_ids]
        for tid in lost_ids:
            tracklet = self.active_tracks.pop(tid)
            if len(tracklet.features) > 0:
                tracklet.aggregate_features()
                self.completed_tracklets.append(tracklet)

    def flush(self) -> None:
        """Finalize all remaining active tracks (call at end of video)."""
        for tid in list(self.active_tracks.keys()):
            tracklet = self.active_tracks.pop(tid)
            if len(tracklet.features) > 0:
                tracklet.aggregate_features()
                self.completed_tracklets.append(tracklet)

    @property
    def num_active(self) -> int:
        return len(self.active_tracks)

    @property
    def num_completed(self) -> int:
        return len(self.completed_tracklets)
