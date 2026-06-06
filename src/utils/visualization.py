"""Visualization utilities for cross-camera tracking.

Draws bounding boxes with consistent colors keyed on global_id so the same
box appears in the same color across all camera views.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils.tracklet import Tracklet

logger = logging.getLogger(__name__)

# 24 distinct BGR colors — indexed by global_id % len(palette)
COLOR_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
    (0, 255, 0),     # green
    (255, 255, 0),   # cyan
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (128, 0, 128),   # purple
    (0, 128, 128),   # olive
    (128, 128, 0),   # teal
    (0, 0, 128),     # dark red
    (128, 0, 0),     # dark blue
    (0, 128, 0),     # dark green
    (255, 128, 0),   # orange
    (0, 128, 255),   # light orange
    (128, 255, 0),   # lime
    (255, 0, 128),   # pink
    (128, 0, 255),   # violet
    (0, 255, 128),   # spring green
    (255, 128, 128), # salmon
    (128, 255, 128), # pale green
    (128, 128, 255), # lavender
    (255, 255, 128), # light cyan
    (128, 255, 255), # light yellow
    (255, 128, 255), # light magenta
]


def get_color(global_id: int | None, local_id: int = 0) -> tuple[int, int, int]:
    """Get a consistent BGR color for the given ID.

    Uses global_id if assigned; falls back to local_id otherwise.

    Args:
        global_id: Cross-camera global ID (may be None).
        local_id: Per-camera track ID (fallback).

    Returns:
        BGR color tuple.
    """
    idx = global_id if global_id is not None else local_id
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]


def draw_tracklet(
    frame: np.ndarray,
    tracklet: Tracklet,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw the latest bbox and label for a single tracklet.

    Args:
        frame: BGR image (modified in place).
        tracklet: Tracklet with at least one bbox.
        thickness: Bounding box line thickness.
        font_scale: Label text scale.

    Returns:
        The frame (modified in place, also returned for convenience).
    """
    if not tracklet.bboxes:
        return frame

    bbox = tracklet.bboxes[-1]  # latest detection
    color = get_color(tracklet.global_id, tracklet.local_id)
    x1, y1 = int(bbox.x1), int(bbox.y1)
    x2, y2 = int(bbox.x2), int(bbox.y2)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label
    if tracklet.global_id is not None:
        label = f"G{tracklet.global_id}"
    else:
        label = f"L{tracklet.local_id}"

    _draw_label(frame, label, x1, y1, color, font_scale, thickness)

    return frame


def draw_frame(
    frame: np.ndarray,
    active_tracklets: list[Tracklet],
    camera_id: int,
    thickness: int = 2,
    font_scale: float = 0.6,
    show_camera_label: bool = True,
) -> np.ndarray:
    """Draw all active tracklets on a camera frame.

    Args:
        frame: BGR image (modified in place).
        active_tracklets: All currently active tracklets for this camera.
        camera_id: Camera identifier (for the corner label).
        thickness: Bounding box line thickness.
        font_scale: Label text scale.
        show_camera_label: If True, show "Camera N" in the top-left corner.

    Returns:
        The annotated frame.
    """
    for tracklet in active_tracklets:
        draw_tracklet(frame, tracklet, thickness, font_scale)

    if show_camera_label:
        cam_label = f"Camera {camera_id}"
        cv2.putText(
            frame, cam_label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

    return frame


def _draw_label(
    frame: np.ndarray,
    label: str,
    x: int, y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw a text label with a filled background above the bbox."""
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
    )
    label_y = max(y - th - baseline - 4, 0)

    # Filled background
    cv2.rectangle(
        frame,
        (x, label_y),
        (x + tw + 4, label_y + th + baseline + 2),
        color, cv2.FILLED,
    )
    # Text in white or black depending on color brightness
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    cv2.putText(
        frame, label, (x + 2, label_y + th),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1,
    )


class VideoWriter:
    """Helper to write annotated frames to a video file.

    Usage:
        writer = VideoWriter("output/cam0.mp4", fps=30, size=(1920, 1080))
        writer.write(frame)
        writer.close()
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        size: tuple[int, int] = (1920, 1080),
        codec: str = "mp4v",
    ):
        """Initialize the video writer.

        Args:
            output_path: Path for the output video file.
            fps: Frames per second.
            size: (width, height) of the output video.
            codec: FourCC codec string.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        self._path = output_path
        logger.info("VideoWriter: %s (%.1f fps, %dx%d)", output_path, fps, size[0], size[1])

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video."""
        self._writer.write(frame)

    def close(self) -> None:
        """Release the video writer."""
        self._writer.release()
        logger.info("VideoWriter closed: %s", self._path)

    @property
    def path(self) -> str:
        return self._path
