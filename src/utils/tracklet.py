"""Tracklet and BBox data structures for box-track."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BBox:
    """A single detected bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int
    frame_id: int = 0
    timestamp: float = 0.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-6)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return (center_x, center_y, width, height)."""
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
            self.width,
            self.height,
        )


@dataclass
class Tracklet:
    """A single-camera tracklet: one tracked object's trajectory within one camera."""

    camera_id: int
    local_id: int
    global_id: int | None = None

    # Temporal data
    start_time: float = 0.0
    end_time: float = 0.0
    frames: list[int] = field(default_factory=list)

    # Detection boxes
    bboxes: list[BBox] = field(default_factory=list)

    # Per-frame backbone features (one per detection, same order as bboxes)
    features: list[np.ndarray] = field(default_factory=list)

    # Aggregated tracklet-level feature vector
    aggregated_feature: np.ndarray | None = None

    # Majority class inferred from bboxes (None until computed)
    cls_id: int | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_detections(self) -> int:
        return len(self.bboxes)

    def aggregate_features(self, method: str = "mean") -> np.ndarray | None:
        """Aggregate per-frame features into a single tracklet-level feature vector.

        Args:
            method: Aggregation method — "mean", "median", or "max_conf".
                    "max_conf" uses the feature from the highest-confidence detection.

        Returns:
            L2-normalized feature vector, or None if no features are available.
        """
        if not self.features:
            return None

        stacked = np.stack(self.features, axis=0)  # (N, D)

        if method == "mean":
            feat = stacked.mean(axis=0)
        elif method == "median":
            feat = np.median(stacked, axis=0)
        elif method == "max_conf":
            if not self.bboxes:
                return None
            best_idx = max(range(len(self.bboxes)), key=lambda i: self.bboxes[i].conf)
            feat = stacked[best_idx]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # L2 normalize
        norm = np.linalg.norm(feat)
        if norm > 1e-12:
            feat = feat / norm

        self.aggregated_feature = feat
        return feat

    def infer_majority_class(self) -> int | None:
        """Infer the majority class from bboxes via confidence-weighted voting.

        Scans all stored BBox objects and selects the cls_id with the
        highest total confidence sum. Sets self.cls_id and returns it.

        Returns:
            The majority cls_id, or None if no bboxes exist.
        """
        if not self.bboxes:
            self.cls_id = None
            return None

        conf_sums: dict[int, float] = {}
        for bb in self.bboxes:
            conf_sums[bb.cls_id] = conf_sums.get(bb.cls_id, 0.0) + bb.conf

        self.cls_id = max(conf_sums, key=conf_sums.__getitem__)
        return self.cls_id

    def to_summary(self) -> dict:
        """Return a human-readable summary dict."""
        return {
            "camera_id": self.camera_id,
            "local_id": self.local_id,
            "global_id": self.global_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "num_detections": self.num_detections,
            "has_features": self.aggregated_feature is not None,
            "cls_id": self.cls_id,
        }
