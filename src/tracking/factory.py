"""Tracker factory for Phase 2 pipeline backends."""

from typing import Any

from src.tracking.botsort_tracker import BOTSORTTracker
from src.tracking.roboflow_bytetrack_tracker import RoboflowByteTrackTracker


def create_tracker(
    camera_id: int,
    config: dict[str, Any],
    **overrides: Any,
) -> BOTSORTTracker | RoboflowByteTrackTracker:
    """Create a single-camera tracker from pipeline config."""
    backend = config.get("tracker_backend", "botsort")
    conf = config.get("conf", 0.7)

    if backend == "botsort":
        return BOTSORTTracker(
            camera_id=camera_id,
            model_path=config.get("model_path", "models/best.pt"),
            tracker_cfg=config.get("tracker_cfg", "configs/botsort.yaml"),
            conf=conf,
            hook_layer=config.get("feature", {}).get("hook_layer", -2),
        )

    if backend == "roboflow_bytetrack":
        feature_cfg = config.get("feature", {})
        return RoboflowByteTrackTracker(
            camera_id=camera_id,
            roboflow_cfg=config.get("roboflow", {}),
            feature_model_path=feature_cfg.get("model_path", "models/yolo26n.pt"),
            feature_hook_layer=feature_cfg.get("hook_layer", -2),
            conf=conf,
            tracker_cfg=config.get("bytetrack", {}),
            **overrides,
        )

    raise ValueError(f"Unknown tracker_backend: {backend}")
