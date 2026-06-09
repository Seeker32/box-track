"""Online (real-time) cross-camera tracking pipeline.

Phase 2: Multi-camera real-time tracking with streaming input, live matching,
visualization, and persistence.
"""

import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.association.global_id_manager import GlobalIDManager
from src.io import MultiStreamManager, StreamReader
from src.tracking.botsort_tracker import BOTSORTTracker
from src.tracking.factory import create_tracker
from src.utils.persistence import TrajectoryLogger, export_csv, export_summary_json
from src.utils.tracklet import Tracklet
from src.utils.visualization import draw_frame, VideoWriter

logger = logging.getLogger(__name__)


class OnlineCrossCameraPipeline:
    """Real-time multi-camera tracking pipeline.

    Reads frames from multiple sources (RTSP or files) in background threads,
    runs detection + tracking per camera, matches completed tracklets across
    cameras online, and optionally outputs annotated video and trajectory logs.

    Usage:
        pipeline = OnlineCrossCameraPipeline("configs/pipeline.yaml")
        pipeline.setup_streams([
            "rtsp://192.168.1.10/stream",
            "rtsp://192.168.1.11/stream",
        ])
        pipeline.run()  # blocks until Ctrl+C or all streams end
    """

    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        """Initialize the pipeline.

        Args:
            config_path: Path to the pipeline YAML config.
        """
        self.config = self._load_config(config_path)
        cfg_online = self.config.get("online", {})
        cfg_stream = self.config.get("stream", {})
        cfg_persistence = self.config.get("persistence", {})

        # Core components
        self.global_id_mgr = GlobalIDManager(
            disappearance_timeout=self.config.get("global_id", {}).get(
                "disappearance_timeout", 60.0
            )
        )
        self._trackers: dict[int, BOTSORTTracker] = {}
        self._stream_mgr: MultiStreamManager | None = None

        # Configuration
        self.matching_cfg = self.config.get("matching", {})
        self.feature_cfg = self.config.get("feature", {})
        self.viz_enabled = cfg_online.get("visualization", True)
        self.persistence_enabled = cfg_online.get("persistence", True)
        self.max_fps = cfg_online.get("max_fps", 0)  # 0 = unlimited
        self.match_interval = cfg_online.get("match_interval", 0.5)  # seconds
        self.display_enabled = cfg_online.get("display", False)

        # Output directories
        viz_dir = cfg_online.get("output_video_dir", "output/videos")
        persist_dir = cfg_persistence.get("output_dir", "output/trajectories")
        self.output_video_dir = Path(viz_dir)
        self.output_persist_dir = Path(persist_dir)

        # Stream settings
        self.reconnect_enabled = cfg_stream.get("reconnect", True)
        self.reconnect_base_delay = cfg_stream.get("reconnect_base_delay", 1.0)
        self.reconnect_max_delay = cfg_stream.get("reconnect_max_delay", 30.0)

        # Runtime state
        self._running = False
        self._frame_id: int = 0
        self._last_match_time: float = 0.0
        self._logger: TrajectoryLogger | None = None
        self._video_writers: dict[int, VideoWriter] = {}
        self._all_completed: list[Tracklet] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def setup_streams(self, sources: list[str]) -> None:
        """Create StreamReader instances for each source.

        Args:
            sources: List of video file paths or RTSP URLs (one per camera).
        """
        readers: list[StreamReader] = []
        for cam_id, source in enumerate(sources):
            readers.append(StreamReader(
                camera_id=cam_id,
                source=source,
                reconnect=self.reconnect_enabled,
                reconnect_base_delay=self.reconnect_base_delay,
                reconnect_max_delay=self.reconnect_max_delay,
                on_reconnect=lambda cid: logger.warning("Camera %d: reconnecting...", cid),
            ))
        self._stream_mgr = MultiStreamManager(readers)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run the online pipeline. Blocks until stopped or all streams end.

        Returns:
            Summary dict with total tracklets, global targets, etc.
        """
        if self._stream_mgr is None:
            raise RuntimeError("Call setup_streams() before run()")

        self._running = True
        self._setup_signal_handlers()
        self._init_trackers()
        self._init_outputs()

        logger.info("=== Online Pipeline Started ===")
        logger.info("Cameras: %d, Viz: %s, Persist: %s",
                     len(self._stream_mgr.camera_ids),
                     self.viz_enabled, self.persistence_enabled)

        self._stream_mgr.start_all()
        start_time = time.time()
        frame_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        last_process_time = 0.0

        try:
            with ThreadPoolExecutor(max_workers=len(self._trackers)) as executor:
                while self._running:
                    loop_start = time.time()

                    # Rate limiting
                    if frame_interval > 0 and (loop_start - last_process_time) < frame_interval:
                        time.sleep(0.001)
                        continue

                    # Get latest frames from all cameras
                    frames = self._stream_mgr.get_frames()
                    if not frames:
                        # No frames available — check if all streams are dead
                        if not self._stream_mgr.all_connected:
                            all_dead = all(
                                not r.is_connected
                                for r in [self._stream_mgr.get_reader(cid)
                                          for cid in self._stream_mgr.camera_ids]
                                if r is not None
                            )
                            if all_dead:
                                logger.warning("All streams disconnected, stopping")
                                break
                        time.sleep(0.01)
                        continue

                    # Process each camera's frame in parallel
                    futures: dict[int, Future] = {}
                    for cam_id, (frame, timestamp) in frames.items():
                        future = executor.submit(
                            self._process_camera_frame, cam_id, frame, timestamp
                        )
                        futures[cam_id] = future

                    # Collect results
                    for cam_id, future in futures.items():
                        try:
                            annotated = future.result(timeout=10.0)
                        except Exception:
                            logger.exception("Camera %d: frame processing failed", cam_id)
                            continue

                        if annotated is not None and cam_id in self._video_writers:
                            self._video_writers[cam_id].write(annotated)

                        if self.display_enabled and annotated is not None:
                            cv2.imshow(f"Camera {cam_id}", annotated)

                    # Periodic cross-camera matching
                    now = time.time()
                    if now - self._last_match_time >= self.match_interval:
                        self._run_online_matching(now)
                        self._last_match_time = now

                    # Check for display keypress
                    if self.display_enabled:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            logger.info("'q' pressed, stopping")
                            self._running = False
                            break

                    self._frame_id += 1
                    last_process_time = loop_start

                    # Periodic status
                    if self._frame_id % 300 == 0:
                        elapsed = now - start_time
                        logger.info(
                            "Frame %d | %.1f fps | %d active global targets | %d completed tracklets",
                            self._frame_id,
                            self._frame_id / elapsed if elapsed > 0 else 0,
                            self.global_id_mgr.num_active,
                            len(self._all_completed),
                        )
        finally:
            self._stream_mgr.stop_all()
            self._finalize_outputs()

        elapsed = time.time() - start_time
        summary = {
            "total_frames": self._frame_id,
            "elapsed_seconds": round(elapsed, 1),
            "avg_fps": round(self._frame_id / elapsed, 1) if elapsed > 0 else 0,
            "total_tracklets": len(self._all_completed),
            "total_global_targets": self.global_id_mgr.num_active,
        }
        logger.info("=== Pipeline Finished ===")
        logger.info("Summary: %s", summary)
        return summary

    def stop(self) -> None:
        """Signal the pipeline to stop gracefully."""
        self._running = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        """Register graceful shutdown on SIGINT/SIGTERM."""

        def handler(signum, frame):
            logger.info("Received signal %d, stopping", signum)
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass  # not in main thread

    def _init_trackers(self) -> None:
        """Create a BOTSORTTracker for each camera."""
        if self._stream_mgr is None:
            return
        for cam_id in self._stream_mgr.camera_ids:
            self._trackers[cam_id] = create_tracker(
                camera_id=cam_id,
                config=self.config,
            )
            logger.info("Tracker initialized for camera %d", cam_id)

    def _init_outputs(self) -> None:
        """Initialize video writers and trajectory logger."""
        if self.viz_enabled and self._stream_mgr is not None:
            self.output_video_dir.mkdir(parents=True, exist_ok=True)
            # We'll create writers lazily on first frame (so we know the resolution)

        if self.persistence_enabled:
            self.output_persist_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = self.output_persist_dir / f"trajectories_{ts}.jsonl"
            self._logger = TrajectoryLogger(str(log_path))

    def _ensure_video_writer(self, cam_id: int, frame: np.ndarray) -> None:
        """Lazily create a VideoWriter for the given camera."""
        if cam_id in self._video_writers:
            return
        h, w = frame.shape[:2]
        stream_name = f"cam{cam_id}"
        out_path = self.output_video_dir / f"{stream_name}_annotated.mp4"
        fps = 30.0
        if self._stream_mgr is not None:
            reader = self._stream_mgr.get_reader(cam_id)
            if reader is not None:
                fps = reader.fps
        self._video_writers[cam_id] = VideoWriter(str(out_path), fps=fps, size=(w, h))

    def _process_camera_frame(
        self, cam_id: int, frame: np.ndarray, timestamp: float
    ) -> np.ndarray | None:
        """Process one frame for one camera: detect, track, extract features.

        Also detects newly completed tracklets and runs online matching.

        Args:
            cam_id: Camera identifier.
            frame: BGR image.
            timestamp: Frame timestamp in seconds.

        Returns:
            Annotated frame (if visualization enabled), or None.
        """
        tracker = self._trackers.get(cam_id)
        if tracker is None:
            return None

        # Before processing, count current completed tracklets
        prev_completed = tracker.num_completed

        # Run tracking on this frame
        tracker.process_frame(frame, self._frame_id, timestamp)

        # Check for newly completed tracklets
        if tracker.num_completed > prev_completed:
            # New tracklets have been finalized — match them online
            new_count = tracker.num_completed - prev_completed
            new_tracklets = tracker.completed_tracklets[-new_count:]
            for tracklet in new_tracklets:
                gid = self.global_id_mgr.register_or_match(tracklet, self.matching_cfg)
                tracklet.global_id = gid
                self._all_completed.append(tracklet)

                if self._logger is not None:
                    self._logger.log(tracklet)

        # Visualization
        annotated = None
        if self.viz_enabled:
            self._ensure_video_writer(cam_id, frame)
            annotated = frame.copy()
            active = [tracker.active_tracks[tid]
                      for tid in tracker.active_tracks
                      if tid in tracker.active_tracks]
            draw_frame(annotated, active, cam_id)

        return annotated

    def _run_online_matching(self, current_time: float) -> None:
        """Periodic housekeeping: cleanup expired targets."""
        expired = self.global_id_mgr.cleanup_expired(current_time)
        if expired:
            logger.debug("Cleaned up %d expired target(s)", len(expired))

    def _finalize_outputs(self) -> None:
        """Flush trackers, close video writers, export final results."""
        # Flush any remaining active tracks in all trackers
        for tracker in self._trackers.values():
            tracker.flush()
            for tracklet in tracker.completed_tracklets:
                if tracklet not in self._all_completed:
                    gid = self.global_id_mgr.register_or_match(tracklet, self.matching_cfg)
                    tracklet.global_id = gid
                    self._all_completed.append(tracklet)
                    if self._logger is not None:
                        self._logger.log(tracklet)

        # Close video writers
        for writer in self._video_writers.values():
            writer.close()
        self._video_writers.clear()

        # Close trajectory logger
        if self._logger is not None:
            self._logger.close()

        # Export final summaries
        if self.persistence_enabled and self._all_completed:
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_persist_dir / f"detections_{ts}.csv"
            json_path = self.output_persist_dir / f"summary_{ts}.json"
            export_csv(self._all_completed, str(csv_path))
            export_summary_json(self._all_completed, str(json_path))

        if self.display_enabled:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Batch file mode (no streaming)
    # ------------------------------------------------------------------

    def process_videos(self, video_paths: list[str]) -> dict:
        """Process pre-recorded video files (non-streaming batch mode).

        Unlike run(), this processes videos sequentially — useful for
        evaluation and debugging.

        Args:
            video_paths: List of video file paths (one per camera).

        Returns:
            Summary dict.
        """
        all_tracklets: dict[int, list[Tracklet]] = {}
        matching_cfg = self.config.get("matching", {})

        for cam_id, video_path in enumerate(video_paths):
            tracker = create_tracker(camera_id=cam_id, config=self.config)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_id = 0

            logger.info("Processing camera %d: %s (%d frames, %.1f fps)",
                         cam_id, video_path, total_frames, fps)

            # Setup video writer for annotated output
            writer = None
            if self.viz_enabled:
                self.output_video_dir.mkdir(parents=True, exist_ok=True)
                cam_name = Path(video_path).stem
                out_path = self.output_video_dir / f"{cam_name}_annotated.mp4"
                writer = VideoWriter(str(out_path), fps=fps, size=(w, h))

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    timestamp = frame_id / fps if fps > 0 else float(frame_id)
                    tracker.process_frame(frame, frame_id, timestamp)
                    frame_id += 1

                    # Draw and write annotated frame
                    if writer is not None:
                        annotated = frame.copy()
                        active = [tracker.active_tracks[tid]
                                  for tid in tracker.active_tracks
                                  if tid in tracker.active_tracks]
                        draw_frame(annotated, active, cam_id)
                        writer.write(annotated)

                    if frame_id % 100 == 0:
                        logger.info("Camera %d: %d/%d frames", cam_id, frame_id, total_frames)
            finally:
                cap.release()
                if writer is not None:
                    writer.close()

            tracker.flush()
            all_tracklets[cam_id] = list(tracker.completed_tracklets)
            logger.info("Camera %d: %d tracklets", cam_id, len(all_tracklets[cam_id]))

        # Cross-camera matching (sequential batch)
        cameras = sorted(all_tracklets.keys())
        for t in all_tracklets[cameras[0]]:
            self.global_id_mgr.register_new(t)

        for i in range(1, len(cameras)):
            self.global_id_mgr.match_and_assign(
                all_tracklets[cameras[i - 1]], all_tracklets[cameras[i]], matching_cfg,
            )

        # Collect all tracklets
        self._all_completed = []
        for cam_tracklets in all_tracklets.values():
            self._all_completed.extend(cam_tracklets)

        # Persist
        if self.persistence_enabled:
            self.output_persist_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = self.output_persist_dir / f"trajectories_{ts}.jsonl"
            logger_inst = TrajectoryLogger(str(log_path))
            for t in self._all_completed:
                logger_inst.log(t)
            logger_inst.close()

            csv_path = self.output_persist_dir / f"detections_{ts}.csv"
            json_path = self.output_persist_dir / f"summary_{ts}.json"
            export_csv(self._all_completed, str(csv_path))
            export_summary_json(self._all_completed, str(json_path))
            logger.info("Exported: %s, %s, %s", log_path, csv_path, json_path)

        return {
            "total_tracklets": len(self._all_completed),
            "total_global_targets": self.global_id_mgr.num_active,
            "cameras": len(video_paths),
        }
