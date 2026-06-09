#!/usr/bin/env python3
"""Draw CSV detections on the three VID input videos with a fixed ID/color."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CSV = Path("output/trajectories/detections_20260609_143105.csv")
DEFAULT_INPUT_DIR = Path("input")
DEFAULT_OUTPUT_DIR = Path("output/videos")
DEFAULT_SUFFIX = "_csv_id0_red"
DEFAULT_LABEL = "ID 0"
RED_BGR = (0, 0, 255)

logger = logging.getLogger("annotate-videos-from-csv")


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int


DetectionsByCamera = dict[int, dict[int, list[Detection]]]


def default_video_mapping(input_dir: Path) -> dict[int, Path]:
    """Map sorted VID*.mp4 files to camera IDs."""
    videos = sorted(input_dir.glob("VID*.mp4"))
    if len(videos) != 3:
        raise ValueError(f"Expected exactly 3 VID*.mp4 files in {input_dir}, found {len(videos)}")
    return {camera_id: video_path for camera_id, video_path in enumerate(videos)}


def load_detections(csv_path: Path) -> DetectionsByCamera:
    """Load detection boxes grouped by camera_id and frame_id."""
    detections: DetectionsByCamera = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"camera_id", "frame_id", "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            camera_id = int(row["camera_id"])
            frame_id = int(row["frame_id"])
            detection = Detection(
                x1=round_float_to_int(row["x1"]),
                y1=round_float_to_int(row["y1"]),
                x2=round_float_to_int(row["x2"]),
                y2=round_float_to_int(row["y2"]),
            )
            detections.setdefault(camera_id, {}).setdefault(frame_id, []).append(detection)
    return detections


def round_float_to_int(value: str) -> int:
    """Convert CSV float strings like '447.0' to pixel coordinates."""
    return int(round(float(value)))


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    label: str = DEFAULT_LABEL,
    color: tuple[int, int, int] = RED_BGR,
) -> np.ndarray:
    """Draw all detections on a frame in the same red ID-0 style."""
    height, width = frame.shape[:2]
    for detection in detections:
        x1 = min(max(detection.x1, 0), width - 1)
        y1 = min(max(detection.y1, 0), height - 1)
        x2 = min(max(detection.x2, 0), width - 1)
        y2 = min(max(detection.y2, 0), height - 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1, color)
    return frame


def draw_label(
    frame: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled label box above the detection when space allows."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    top = max(y - text_height - baseline - 6, 0)
    bottom = top + text_height + baseline + 4
    right = min(x + text_width + 6, frame.shape[1] - 1)

    cv2.rectangle(frame, (x, top), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, label, (x + 3, top + text_height + 1), font, font_scale, (255, 255, 255), 1)


def output_path_for(video_path: Path, output_dir: Path, suffix: str) -> Path:
    """Build the annotated output path for a source video."""
    return output_dir / f"{video_path.stem}{suffix}.mp4"


def annotate_video(
    video_path: Path,
    output_path: Path,
    frame_detections: dict[int, list[Detection]],
    label: str = DEFAULT_LABEL,
) -> None:
    """Write one annotated video from one source video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            draw_detections(frame, frame_detections.get(frame_id, []), label=label)
            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    logger.info(
        "Wrote %s (%d/%d frames, %d frames with boxes)",
        output_path,
        frame_id,
        total_frames,
        len(frame_detections),
    )


def annotate_all(
    csv_path: Path = DEFAULT_CSV,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    suffix: str = DEFAULT_SUFFIX,
    label: str = DEFAULT_LABEL,
) -> list[Path]:
    """Annotate all default VID videos and return output paths."""
    detections = load_detections(csv_path)
    videos = default_video_mapping(input_dir)
    outputs: list[Path] = []
    for camera_id, video_path in videos.items():
        out_path = output_path_for(video_path, output_dir, suffix)
        annotate_video(video_path, out_path, detections.get(camera_id, {}), label=label)
        outputs.append(out_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw detections from a trajectory CSV on input/VID*.mp4 videos.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"CSV path (default: {DEFAULT_CSV})")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing VID*.mp4 files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for annotated videos (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX, help=f"Output filename suffix (default: {DEFAULT_SUFFIX})")
    parser.add_argument("--label", default=DEFAULT_LABEL, help=f"Box label text (default: {DEFAULT_LABEL})")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    outputs = annotate_all(
        csv_path=args.csv,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        suffix=args.suffix,
        label=args.label,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
