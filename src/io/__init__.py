"""Multi-camera stream reader with RTSP support and auto-reconnect."""

import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum frames to buffer per camera (bounds memory for fast input sources)
DEFAULT_MAX_QUEUE_SIZE = 30


class StreamReader:
    """Reads frames from a video source (file or RTSP URL) in a background thread.

    Frames are placed into a thread-safe queue. Supports auto-reconnect for
    network streams (RTSP/RTMP/HTTP) with exponential backoff.

    Usage:
        reader = StreamReader(0, "rtsp://192.168.1.10/stream", max_queue=30)
        reader.start()
        frame = reader.read()  # non-blocking; returns None if no frame available
        reader.stop()
    """

    def __init__(
        self,
        camera_id: int,
        source: str,
        max_queue: int = DEFAULT_MAX_QUEUE_SIZE,
        reconnect: bool = True,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        on_reconnect: Callable[[int], None] | None = None,
    ):
        """Initialize the stream reader.

        Args:
            camera_id: Logical camera identifier.
            source: Video file path or RTSP/RTMP/HTTP stream URL.
            max_queue: Maximum number of buffered frames (drop oldest on overflow).
            reconnect: If True, attempt to reconnect on stream failure.
            reconnect_base_delay: Initial reconnect delay in seconds (doubles each retry).
            reconnect_max_delay: Maximum reconnect delay in seconds.
            on_reconnect: Optional callback invoked when a reconnect attempt starts.
        """
        self.camera_id = camera_id
        self.source = source
        self.max_queue = max_queue
        self.reconnect_enabled = reconnect
        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.on_reconnect = on_reconnect

        self._cap: cv2.VideoCapture | None = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False
        self._fps: float = 0.0
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background read thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"stream-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Camera %d: stream reader started (source=%s)", self.camera_id, self.source)

    def stop(self) -> None:
        """Stop the background thread and release the capture device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._release_capture()
        # Drain the queue so no references linger
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Camera %d: stream reader stopped", self.camera_id)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Read the next available frame (non-blocking).

        Returns:
            Tuple of (ok, frame, timestamp).
            ok=False means the reader is stopped or the queue is empty.
            frame is None if no data is available.
            timestamp is the approximate capture time in seconds.
        """
        if not self._running:
            return False, None, 0.0
        try:
            frame = self._queue.get_nowait()
            if frame is None:
                return False, None, 0.0
            return True, frame, time.time()
        except queue.Empty:
            return False, None, 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_capture(self) -> bool:
        """Open the video capture. Returns True on success."""
        # Detect source type
        source_lower = self.source.lower()
        if (
            source_lower.startswith("rtsp://")
            or source_lower.startswith("rtmp://")
            or source_lower.startswith("http://")
            or source_lower.startswith("https://")
        ):
            # Network stream: use FFmpeg backend with TCP transport for reliability
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            # Prefer TCP over UDP for RTSP (more reliable)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        elif Path(self.source).exists():
            cap = cv2.VideoCapture(self.source)
        else:
            # Try as-is
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            cap.release()
            return False

        self._cap = cap
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0  # default assumption

        self._connected = True
        logger.info("Camera %d: capture opened (fps=%.1f)", self.camera_id, self._fps)
        return True

    def _release_capture(self) -> None:
        """Release the current capture device."""
        self._connected = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_loop(self) -> None:
        """Background loop: read frames and put them in the queue."""
        reconnect_delay = self.reconnect_base_delay

        while self._running:
            # Open capture if needed
            if self._cap is None or not self._connected:
                if not self._open_capture():
                    if not self.reconnect_enabled:
                        logger.error("Camera %d: cannot open source, stopping", self.camera_id)
                        self._running = False
                        break
                    logger.warning(
                        "Camera %d: cannot open source, retrying in %.1fs",
                        self.camera_id, reconnect_delay,
                    )
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, self.reconnect_max_delay)
                    continue
                reconnect_delay = self.reconnect_base_delay  # reset on success

            # Read frame
            try:
                ret, frame = self._cap.read()
            except Exception:
                logger.exception("Camera %d: exception during read", self.camera_id)
                ret = False

            if not ret or frame is None:
                logger.warning("Camera %d: read failed, reconnecting", self.camera_id)
                self._release_capture()
                if self.on_reconnect:
                    try:
                        self.on_reconnect(self.camera_id)
                    except Exception:
                        pass
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self.reconnect_max_delay)
                continue

            self._frame_count += 1

            # Non-blocking put: if queue is full, discard oldest frame
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass  # give up


class MultiStreamManager:
    """Manages multiple StreamReader instances, one per camera.

    Usage:
        mgr = MultiStreamManager([
            StreamReader(0, "rtsp://cam0/stream"),
            StreamReader(1, "cam1.mp4"),
        ])
        mgr.start_all()
        while True:
            frames = mgr.get_frames()
            # frames is dict[camera_id, tuple[frame, timestamp]]
            ...
        mgr.stop_all()
    """

    def __init__(self, readers: list[StreamReader]):
        """Initialize with a list of StreamReader instances."""
        self._readers: dict[int, StreamReader] = {}
        for reader in readers:
            self._readers[reader.camera_id] = reader

    def start_all(self) -> None:
        """Start all stream readers."""
        for reader in self._readers.values():
            reader.start()

    def stop_all(self) -> None:
        """Stop all stream readers."""
        for reader in self._readers.values():
            reader.stop()

    def get_frames(self) -> dict[int, tuple[np.ndarray, float]]:
        """Get the latest frame from each camera (non-blocking).

        Returns:
            dict mapping camera_id → (frame, timestamp).
            Cameras with no new frame are omitted.
        """
        result: dict[int, tuple[np.ndarray, float]] = {}
        for cam_id, reader in self._readers.items():
            ok, frame, ts = reader.read()
            if ok and frame is not None:
                result[cam_id] = (frame, ts)
        return result

    def get_reader(self, camera_id: int) -> StreamReader | None:
        """Get a specific reader by camera ID."""
        return self._readers.get(camera_id)

    @property
    def camera_ids(self) -> list[int]:
        return sorted(self._readers.keys())

    @property
    def all_connected(self) -> bool:
        return all(r.is_connected for r in self._readers.values())

    @property
    def status(self) -> dict[int, dict]:
        """Return connection status for all cameras."""
        return {
            cam_id: {
                "connected": r.is_connected,
                "fps": r.fps,
                "frame_count": r.frame_count,
                "queue_size": r.queue_size,
            }
            for cam_id, r in self._readers.items()
        }
