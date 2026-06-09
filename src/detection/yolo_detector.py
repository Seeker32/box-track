"""YOLO detector wrapper for box detection."""

import logging
from collections.abc import Generator

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.utils.tracklet import BBox

logger = logging.getLogger(__name__)


class YOLODetector:
    """Thin wrapper around YOLO for running box detection on images or videos."""

    def __init__(self, model_path: str = "models/best.pt", conf: float = 0.7):
        """Initialize the detector.

        Args:
            model_path: Path to the YOLO model weights.
            conf: Detection confidence threshold.
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> tuple[list[BBox], list[Results]]:
        """Run detection on a single image.

        Args:
            image: Input image in BGR format (H, W, 3).
            frame_id: Optional frame number for metadata.
            timestamp: Optional timestamp for metadata.

        Returns:
            Tuple of (list of BBox objects, raw ultralytics Results).
        """
        results = self.model(image, conf=self.conf, verbose=False)
        bboxes: list[BBox] = []

        for r in results:
            if r.boxes is None:
                continue
            for box, conf_val, cls_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                x1, y1, x2, y2 = box.tolist()
                bboxes.append(
                    BBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        conf=float(conf_val.item()),
                        cls_id=int(cls_id.item()),
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                )

        return bboxes, results

    def detect_video(
        self, video_path: str, stride: int = 1
    ) -> Generator[tuple[int, list[BBox], np.ndarray], None, None]:
        """Iterate over video frames, yielding detections.

        Args:
            video_path: Path to the video file.
            stride: Process every Nth frame (1 = all frames).

        Yields:
            Tuple of (frame_id, list of BBox objects, frame image).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % stride == 0:
                    timestamp = frame_id / fps if fps > 0 else float(frame_id)
                    bboxes, _ = self.detect(frame, frame_id=frame_id, timestamp=timestamp)
                    yield frame_id, bboxes, frame

                frame_id += 1
        finally:
            cap.release()

    def close(self):
        """Release resources."""
        pass
