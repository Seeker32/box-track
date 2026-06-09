#!/usr/bin/env python3
"""Render a presentation video for an ID-0 red cross-camera demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - fallback is exercised only without Pillow.
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_INPUT_DIR = Path("output/videos")
DEFAULT_PATTERN = "*_csv_id0_red.mp4"
DEFAULT_OUTPUT = Path("output/videos/cross_camera_id0_red_demo2.mp4")
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 24.0

BG = (18, 22, 28)
PANEL = (31, 37, 47)
PANEL_DARK = (24, 29, 37)
TEXT = (238, 242, 247)
TEXT_MUTED = (163, 174, 190)
ACCENT = (94, 197, 34)
ACCENT_SOFT = (153, 211, 52)
RED = (68, 68, 239)
LINE = (80, 94, 113)
WHITE = (255, 255, 255)


@dataclass(frozen=True)
class VideoSegment:
    path: Path
    label: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class TimelineItem:
    label: str
    start_seconds: float
    end_seconds: float


def discover_demo_videos(input_dir: Path, pattern: str) -> list[VideoSegment]:
    """Find demo videos and attach camera labels in playback order."""
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise ValueError(f"Expected at least 1 video matching {pattern!r} in {input_dir}, found 0")

    return build_video_segments(paths)


def build_video_segments(paths: list[Path]) -> list[VideoSegment]:
    """Attach camera labels to explicit video paths in the provided order."""
    if not paths:
        raise ValueError("Expected at least 1 video")
    segments: list[VideoSegment] = []
    for index, path in enumerate(paths, start=1):
        fps, frame_count, width, height = read_video_metadata(path)
        duration = frame_count / fps if fps > 0 else 0.0
        segments.append(
            VideoSegment(
                path=path,
                label=f"摄像头{index}",
                duration_seconds=duration,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
            )
        )
    return segments


def read_video_metadata(path: Path) -> tuple[float, int, int, int]:
    """Return fps, frame count, width, and height for one video."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid metadata: {path}")
        return fps, frame_count, width, height
    finally:
        cap.release()


def build_timeline(segments: list[VideoSegment]) -> list[TimelineItem]:
    """Build cumulative relative-time ranges for each camera segment."""
    timeline: list[TimelineItem] = []
    cursor = 0.0
    for segment in segments:
        end = cursor + segment.duration_seconds
        timeline.append(TimelineItem(segment.label, cursor, end))
        cursor = end
    return timeline


def output_frame_count(segment: VideoSegment, *, target_fps: float) -> int:
    """Return the number of output frames needed to preserve segment duration."""
    return max(1, int(round(segment.duration_seconds * target_fps)))


def compose_frame(
    source_frame: np.ndarray,
    *,
    timeline: list[TimelineItem],
    active_index: int,
    elapsed_seconds: float,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> np.ndarray:
    """Compose one 16:9 frame with video on the left and path state on the right."""
    canvas = np.full((height, width, 3), BG, dtype=np.uint8)
    video_right = int(width * 0.69)
    sidebar_left = video_right

    cv2.rectangle(canvas, (0, 0), (video_right - 1, height - 1), PANEL_DARK, cv2.FILLED)
    draw_video_frame(canvas, source_frame, 44, 78, video_right - 88, height - 156)

    active = timeline[active_index]
    draw_text(canvas, "跨镜头跟踪路径", (sidebar_left + 56, 84), 42, TEXT, bold=True)
    draw_text(canvas, f"当前：{active.label}", (sidebar_left + 56, 205), 34, WHITE, bold=True)
    draw_text(canvas, f"相对时间 {elapsed_seconds:.1f}s", (sidebar_left + 56, 252), 28, ACCENT_SOFT)

    draw_path_panel(canvas, timeline, active_index, elapsed_seconds, sidebar_left, width, height)
    draw_timeline_panel(canvas, timeline, active_index, sidebar_left, width, height)
    return canvas


def draw_video_frame(canvas: np.ndarray, frame: np.ndarray, x: int, y: int, max_w: int, max_h: int) -> None:
    """Fit a source video frame into a stable area without cropping."""
    src_h, src_w = frame.shape[:2]
    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    left = x + (max_w - new_w) // 2
    top = y + (max_h - new_h) // 2

    cv2.rectangle(canvas, (x - 2, y - 2), (x + max_w + 2, y + max_h + 2), (55, 65, 81), 2)
    canvas[top : top + new_h, left : left + new_w] = resized


def draw_path_panel(
    canvas: np.ndarray,
    timeline: list[TimelineItem],
    active_index: int,
    elapsed_seconds: float,
    sidebar_left: int,
    width: int,
    height: int,
) -> None:
    panel_x = sidebar_left + 48
    panel_y = 310
    panel_w = width - sidebar_left - 96
    panel_h = 390
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), PANEL, cv2.FILLED)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (61, 73, 91), 2)

    node_x = panel_x + panel_w // 2
    top_y = panel_y + 72
    bottom_y = panel_y + panel_h - 80
    if len(timeline) <= 1:
        node_ys = [panel_y + panel_h // 2]
    else:
        node_ys = [int(y) for y in np.linspace(top_y, bottom_y, len(timeline))]
    for index in range(len(timeline) - 1):
        color = ACCENT if index < active_index else LINE
        cv2.line(canvas, (node_x, node_ys[index] + 28), (node_x, node_ys[index + 1] - 28), color, 8)

    label_size = 24 if len(timeline) > 3 else 28
    time_size = 20 if len(timeline) > 3 else 23
    for index, item in enumerate(timeline):
        is_done = index < active_index
        is_active = index == active_index
        fill = RED if is_active else (ACCENT if is_done else (71, 85, 105))
        ring = WHITE if is_active else fill
        cv2.circle(canvas, (node_x, node_ys[index]), 34, ring, 4)
        cv2.circle(canvas, (node_x, node_ys[index]), 25, fill, cv2.FILLED)
        draw_text(canvas, item.label, (panel_x + 38, node_ys[index] - 14), label_size, TEXT, bold=is_active)
        draw_text(
            canvas,
            f"{item.start_seconds:.1f}s - {item.end_seconds:.1f}s",
            (node_x + 54, node_ys[index] - 12),
            time_size,
            TEXT_MUTED,
        )

    total = timeline[-1].end_seconds if timeline else 1.0
    progress = min(max(elapsed_seconds / total, 0.0), 1.0)
    bar_x = panel_x + 38
    bar_y = panel_y + panel_h - 38
    bar_w = panel_w - 76
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (64, 75, 91), cv2.FILLED)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 10), ACCENT, cv2.FILLED)


def draw_timeline_panel(
    canvas: np.ndarray,
    timeline: list[TimelineItem],
    active_index: int,
    sidebar_left: int,
    width: int,
    height: int,
) -> None:
    panel_x = sidebar_left + 48
    panel_y = height - 310
    panel_w = width - sidebar_left - 96
    panel_h = 230
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), PANEL, cv2.FILLED)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (61, 73, 91), 2)
    draw_text(canvas, "移动时间线", (panel_x + 34, panel_y + 48), 30, TEXT, bold=True)

    row_y = panel_y + 90
    row_step = min(42, max(24, (panel_h - 110) // max(len(timeline), 1)))
    label_size = 20 if row_step < 34 else 24
    time_size = 19 if row_step < 34 else 23
    for index, item in enumerate(timeline):
        color = RED if index == active_index else (ACCENT if index < active_index else TEXT_MUTED)
        center = (panel_x + 45, row_y + 12)
        if index == active_index:
            cv2.circle(canvas, center, 10, color, cv2.FILLED)
        elif index < active_index:
            cv2.circle(canvas, center, 10, color, cv2.FILLED)
            cv2.line(canvas, (center[0] - 5, center[1]), (center[0] - 1, center[1] + 5), WHITE, 2)
            cv2.line(canvas, (center[0] - 1, center[1] + 5), (center[0] + 7, center[1] - 6), WHITE, 2)
        else:
            cv2.circle(canvas, center, 10, color, 2)
        draw_text(canvas, item.label, (panel_x + 78, row_y), label_size, TEXT)
        draw_text(canvas, f"{item.start_seconds:.1f}s - {item.end_seconds:.1f}s", (panel_x + 245, row_y), time_size, TEXT_MUTED)
        row_y += row_step


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    *,
    bold: bool = False,
) -> None:
    """Draw UTF-8 text on a BGR image."""
    if Image is None or ImageDraw is None or ImageFont is None:
        fallback = text.encode("ascii", "ignore").decode("ascii") or "Camera"
        cv2.putText(image, fallback, position, cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color, 2 if bold else 1)
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = load_font(size, bold=bold)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def load_font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_demo(
    segments: list[VideoSegment],
    output_path: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = DEFAULT_FPS,
) -> None:
    """Render all segments into one presentation MP4."""
    timeline = build_timeline(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    elapsed = 0.0
    try:
        for index, segment in enumerate(segments):
            cap = cv2.VideoCapture(str(segment.path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {segment.path}")
            try:
                frames_to_write = output_frame_count(segment, target_fps=fps)
                for frame_index in range(frames_to_write):
                    source_time = min(frame_index / fps, max(segment.duration_seconds - 0.001, 0.0))
                    cap.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(segment.frame_count - 1, 0))
                        ret, frame = cap.read()
                    if not ret:
                        raise ValueError(f"Cannot read frame from video: {segment.path}")
                    composed = compose_frame(
                        frame,
                        timeline=timeline,
                        active_index=index,
                        elapsed_seconds=elapsed,
                        width=width,
                        height=height,
                    )
                    writer.write(composed)
                    elapsed += 1.0 / fps
            finally:
                cap.release()
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a cross-camera ID0 red demo video.")
    parser.add_argument("--videos", nargs="+", type=Path, help="Explicit input video paths, rendered in the given order.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments = build_video_segments(args.videos) if args.videos else discover_demo_videos(args.input_dir, args.pattern)
    render_demo(segments, args.output, width=args.width, height=args.height, fps=args.fps)
    print(args.output)


if __name__ == "__main__":
    main()
