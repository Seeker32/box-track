"""Cross-camera pipeline: orchestrates tracking, matching, and global ID assignment."""

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.association.global_id_manager import GlobalIDManager
from src.matching.similarity import compute_backbone_similarity
from src.tracking.botsort_tracker import BOTSORTTracker
from src.utils.tracklet import Tracklet

logger = logging.getLogger(__name__)


class CrossCameraPipeline:
    """Full cross-camera tracking pipeline for Phase 1.

    Usage:
        pipeline = CrossCameraPipeline("configs/pipeline.yaml")
        cam0_tracklets = pipeline.process_video(0, "cam0.mp4")
        cam1_tracklets = pipeline.process_video(1, "cam1.mp4")
        all_tracklets = pipeline.run_match({0: cam0_tracklets, 1: cam1_tracklets})
        metrics = pipeline.evaluate(all_tracklets)
    """

    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        """Initialize the pipeline.

        Args:
            config_path: Path to the pipeline YAML config.
        """
        self.config = self._load_config(config_path)
        self.global_id_mgr = GlobalIDManager(
            disappearance_timeout=self.config.get("global_id", {}).get(
                "disappearance_timeout", 60.0
            )
        )

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def process_video(
        self, camera_id: int, video_path: str
    ) -> list[Tracklet]:
        """Process a single camera's video, returning completed tracklets.

        Args:
            camera_id: Camera identifier.
            video_path: Path to the video file.

        Returns:
            List of completed Tracklet objects with aggregated features.
        """
        matching_cfg = self.config.get("matching", {})
        feature_cfg = self.config.get("feature", {})

        tracker = BOTSORTTracker(
            camera_id=camera_id,
            tracker_cfg=self.config.get("tracker_cfg", "configs/botsort.yaml"),
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 0

        logger.info("Processing camera %d: %s (%d frames, %.1f fps)",
                     camera_id, video_path, total_frames, fps)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_id / fps if fps > 0 else float(frame_id)
                tracker.process_frame(frame, frame_id, timestamp)
                frame_id += 1

                if frame_id % 100 == 0:
                    logger.info("Camera %d: processed %d/%d frames",
                                camera_id, frame_id, total_frames)
        finally:
            cap.release()

        # Finalize remaining active tracks
        tracker.flush()

        logger.info("Camera %d: completed — %d tracklets",
                     camera_id, len(tracker.completed_tracklets))

        return tracker.completed_tracklets

    def run_match(
        self, all_tracklets: dict[int, list[Tracklet]]
    ) -> list[Tracklet]:
        """Match tracklets across cameras and assign global IDs.

        For Phase 1: processes cameras in sorted order. Tracklets from
        camera 0 are registered first, then camera 1 is matched against
        camera 0, etc.

        Args:
            all_tracklets: dict mapping camera_id → list of tracklets.

        Returns:
            Flat list of all tracklets with global_id assigned.
        """
        matching_cfg = self.config.get("matching", {})
        cameras = sorted(all_tracklets.keys())

        if len(cameras) < 2:
            logger.warning("Need at least 2 cameras for matching, got %d", len(cameras))
            # Still register all tracklets with unique global IDs
            for cam_id in cameras:
                for t in all_tracklets[cam_id]:
                    self.global_id_mgr.register_new(t)
        else:
            # Register camera 0 tracklets
            for t in all_tracklets[cameras[0]]:
                self.global_id_mgr.register_new(t)

            # Match each subsequent camera against the previous
            for i in range(1, len(cameras)):
                cam_prev = cameras[i - 1]
                cam_curr = cameras[i]
                ta = all_tracklets[cam_prev]
                tb = all_tracklets[cam_curr]

                self.global_id_mgr.match_and_assign(ta, tb, matching_cfg)

        # Return flat list
        result: list[Tracklet] = []
        for cam_tracklets in all_tracklets.values():
            result.extend(cam_tracklets)
        return result

    def evaluate(self, tracklets: list[Tracklet]) -> dict:
        """Evaluate cross-camera matching accuracy.

        Computes Top-1 retrieval accuracy: for each query tracklet from the
        last camera, finds the best match in the gallery (first camera) by
        backbone feature similarity. A match is correct if the gallery
        tracklet has the same global ID.

        Args:
            tracklets: All tracklets with global_id assigned.

        Returns:
            dict with keys: top1_accuracy, total_queries, correct, num_cameras.
        """
        # Group by camera
        cam_tracklets: dict[int, list[Tracklet]] = {}
        for t in tracklets:
            cam_tracklets.setdefault(t.camera_id, []).append(t)

        cameras = sorted(cam_tracklets.keys())
        if len(cameras) < 2:
            return {
                "error": "Need at least 2 cameras for evaluation",
                "num_cameras": len(cameras),
            }

        # Camera 0 = gallery, last camera = query
        gallery = [t for t in cam_tracklets[cameras[0]] if t.aggregated_feature is not None]
        query = [t for t in cam_tracklets[cameras[-1]] if t.aggregated_feature is not None]

        if not gallery or not query:
            return {
                "error": "No tracklets with features in gallery or query",
                "gallery_count": len(gallery),
                "query_count": len(query),
            }

        top1_correct = 0
        for q in query:
            # Find best gallery match by feature similarity
            best_sim = -1.0
            best_match: Tracklet | None = None
            for g in gallery:
                sim = compute_backbone_similarity(q, g)
                if sim > best_sim:
                    best_sim = sim
                    best_match = g

            if best_match is not None and best_match.global_id == q.global_id:
                top1_correct += 1

        accuracy = top1_correct / len(query) if query else 0.0

        return {
            "top1_accuracy": accuracy,
            "total_queries": len(query),
            "correct": top1_correct,
            "num_cameras": len(cameras),
            "gallery_size": len(gallery),
        }
