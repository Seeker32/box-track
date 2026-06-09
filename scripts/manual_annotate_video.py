#!/usr/bin/env python3
"""Interactively mark a few video frames with fixed red ID-0 boxes."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LABEL = "ID 0"
BOX_ID = 0
RED_BGR = (0, 0, 255)
MIN_BOX_SIZE = 2
PAUSED_REFRESH_DELAY_MS = 20

logger = logging.getLogger("manual-annotate-video")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class OutputPaths:
    video: Path
    json: Path
    csv: Path


def default_output_paths(video_path: Path) -> OutputPaths:
    stem = video_path.stem
    return OutputPaths(
        video=Path("output/videos") / f"{stem}_manual_id0_red.mp4",
        json=Path("output/annotations") / f"{stem}_manual_id0.json",
        csv=Path("output/annotations") / f"{stem}_manual_id0.csv",
    )


def normalize_box(
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
    height: int,
) -> Box | None:
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, 0), width - 1)
    y2 = min(max(y2, 0), height - 1)
    if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
        return None
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def draw_boxes(
    frame: np.ndarray,
    boxes: list[Box],
    label: str = LABEL,
    color: tuple[int, int, int] = RED_BGR,
) -> np.ndarray:
    for box in boxes:
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        draw_label(frame, label, box.x1, box.y1, color)
    return frame


def draw_label(
    frame: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    top = max(y - text_height - baseline - 6, 0)
    bottom = top + text_height + baseline + 4
    right = min(x + text_width + 6, frame.shape[1] - 1)
    cv2.rectangle(frame, (x, top), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, label, (x + 3, top + text_height + 1), font, font_scale, (255, 255, 255), 1)


def wait_delay_ms(playing: bool, fps: float) -> int:
    if not playing:
        return PAUSED_REFRESH_DELAY_MS
    return max(int(1000 / fps), 1)


def annotations_payload(
    annotations: dict[int, list[Box]],
    source_video: Path,
    fps: float,
    width: int,
    height: int,
) -> dict[str, object]:
    return {
        "source_video": str(source_video),
        "fps": fps,
        "width": width,
        "height": height,
        "annotations": [
            {
                "frame_id": frame_id,
                "boxes": [
                    {
                        "id": BOX_ID,
                        "label": LABEL,
                        "x1": box.x1,
                        "y1": box.y1,
                        "x2": box.x2,
                        "y2": box.y2,
                    }
                    for box in boxes
                ],
            }
            for frame_id, boxes in sorted(annotations.items())
        ],
    }


def write_annotations(
    annotations: dict[int, list[Box]],
    source_video: Path,
    json_path: Path,
    csv_path: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = annotations_payload(annotations, source_video, fps, width, height)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "id", "label", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        for frame_id, boxes in sorted(annotations.items()):
            for box in boxes:
                writer.writerow({
                    "frame_id": frame_id,
                    "id": BOX_ID,
                    "label": LABEL,
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                })


def render_annotated_video(
    video_path: Path,
    output_path: Path,
    annotations: dict[int, list[Box]],
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
            draw_boxes(frame, annotations.get(frame_id, []))
            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()


class ManualAnnotator:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_id = 0
        self.frame: np.ndarray | None = None
        self.playing = False
        self.annotations: dict[int, list[Box]] = {}
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.window_name = "Manual annotation - space play/pause, a/d frame, r undo, c clear, q save, Esc quit"

    def close(self) -> None:
        self.cap.release()
        cv2.destroyWindow(self.window_name)

    def read_current_frame(self) -> bool:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        ret, frame = self.cap.read()
        if not ret:
            return False
        self.frame = frame
        return True

    def seek(self, frame_id: int) -> None:
        if self.total_frames > 0:
            frame_id = min(max(frame_id, 0), self.total_frames - 1)
        else:
            frame_id = max(frame_id, 0)
        self.frame_id = frame_id
        self.read_current_frame()

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_current = (x, y)
            self.playing = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            box = normalize_box(self.drag_start, (x, y), self.width, self.height)
            if box is not None:
                self.annotations.setdefault(self.frame_id, []).append(box)
            self.drag_start = None
            self.drag_current = None

    def display_frame(self) -> None:
        if self.frame is None:
            return
        display = self.frame.copy()
        draw_boxes(display, self.annotations.get(self.frame_id, []))
        if self.drag_start is not None and self.drag_current is not None:
            preview = normalize_box(self.drag_start, self.drag_current, self.width, self.height)
            if preview is not None:
                cv2.rectangle(display, (preview.x1, preview.y1), (preview.x2, preview.y2), RED_BGR, 1)
        status = f"Frame {self.frame_id}"
        if self.total_frames > 0:
            status = f"{status}/{self.total_frames - 1}"
        status = f"{status} | boxes: {len(self.annotations.get(self.frame_id, []))}"
        cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(self.window_name, display)

    def run(self) -> bool:
        if not self.read_current_frame():
            raise ValueError(f"Cannot read first frame: {self.video_path}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while True:
            self.display_frame()
            key = cv2.waitKey(wait_delay_ms(self.playing, self.fps)) & 0xFF

            if key == 27:
                return False
            if key == ord("q"):
                return True
            if key == ord(" "):
                self.playing = not self.playing
            elif key == ord("d"):
                self.playing = False
                self.seek(self.frame_id + 1)
            elif key == ord("a"):
                self.playing = False
                self.seek(self.frame_id - 1)
            elif key == ord("r"):
                boxes = self.annotations.get(self.frame_id, [])
                if boxes:
                    boxes.pop()
                if not boxes:
                    self.annotations.pop(self.frame_id, None)
            elif key == ord("c"):
                self.annotations.pop(self.frame_id, None)

            if self.playing:
                next_frame = self.frame_id + 1
                if self.total_frames > 0 and next_frame >= self.total_frames:
                    self.playing = False
                else:
                    self.seek(next_frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively draw red ID-0 boxes on selected video frames.")
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-video", type=Path, help="Annotated MP4 output path.")
    parser.add_argument("--output-json", type=Path, help="JSON annotation output path.")
    parser.add_argument("--output-csv", type=Path, help="CSV annotation output path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    defaults = default_output_paths(args.video)
    output_video = args.output_video or defaults.video
    output_json = args.output_json or defaults.json
    output_csv = args.output_csv or defaults.csv

    annotator = ManualAnnotator(args.video)
    try:
        should_save = annotator.run()
        if not should_save:
            logger.info("Exited without saving.")
            return
        write_annotations(
            annotations=annotator.annotations,
            source_video=args.video,
            json_path=output_json,
            csv_path=output_csv,
            fps=annotator.fps,
            width=annotator.width,
            height=annotator.height,
        )
    finally:
        annotator.close()

    render_annotated_video(args.video, output_video, annotator.annotations)
    print(output_video)
    print(output_json)
    print(output_csv)


if __name__ == "__main__":
    main()
