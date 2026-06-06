"""Persistence utilities for trajectory logging and export.

Records completed tracklets to JSON Lines files and supports CSV/JSON export
for offline analysis.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.utils.tracklet import Tracklet

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """Appends completed tracklets to a JSON Lines file.

    Each line is a JSON object with tracklet metadata, bbox summaries,
    and global_id assignment. Feature vectors are excluded to keep file
    sizes reasonable.

    Usage:
        logger = TrajectoryLogger("output/trajectories.jsonl")
        logger.log(tracklet)
        logger.close()
    """

    def __init__(self, output_path: str):
        """Initialize the logger.

        Args:
            output_path: Path to the JSON Lines output file.
        """
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(str(self._path), "a", encoding="utf-8")
        self._count = 0
        logger.info("TrajectoryLogger: %s", self._path)

    def log(self, tracklet: Tracklet) -> None:
        """Write a single tracklet as one JSON line.

        Args:
            tracklet: A completed tracklet with aggregated_feature set.
        """
        bbox_summaries = []
        for bb in tracklet.bboxes:
            bbox_summaries.append({
                "x1": round(bb.x1, 1),
                "y1": round(bb.y1, 1),
                "x2": round(bb.x2, 1),
                "y2": round(bb.y2, 1),
                "conf": round(bb.conf, 4),
                "frame_id": bb.frame_id,
                "timestamp": round(bb.timestamp, 3),
            })

        record = {
            "camera_id": tracklet.camera_id,
            "local_id": tracklet.local_id,
            "global_id": tracklet.global_id,
            "start_time": round(tracklet.start_time, 3),
            "end_time": round(tracklet.end_time, 3),
            "duration": round(tracklet.duration, 3),
            "num_frames": len(tracklet.frames),
            "num_detections": tracklet.num_detections,
            "has_features": tracklet.aggregated_feature is not None,
            "feature_dim": (
                int(tracklet.aggregated_feature.shape[0])
                if tracklet.aggregated_feature is not None else 0
            ),
            "bboxes": bbox_summaries,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }

        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self._count += 1

    def close(self) -> None:
        """Flush and close the log file."""
        self._file.flush()
        self._file.close()
        logger.info("TrajectoryLogger: %d tracklets written to %s", self._count, self._path)

    @property
    def count(self) -> int:
        return self._count


def export_csv(tracklets: list[Tracklet], output_path: str) -> None:
    """Export all tracklet detections as a flat CSV file.

    One row per detection (bbox), suitable for Excel/Pandas analysis.

    Args:
        tracklets: All completed tracklets.
        output_path: Path to the output CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_id", "camera_id", "local_id",
            "frame_id", "timestamp",
            "x1", "y1", "x2", "y2", "width", "height",
            "conf", "cls_id",
        ])

        for t in tracklets:
            for bb in t.bboxes:
                writer.writerow([
                    t.global_id if t.global_id is not None else "",
                    t.camera_id,
                    t.local_id,
                    bb.frame_id,
                    round(bb.timestamp, 3),
                    round(bb.x1, 1), round(bb.y1, 1),
                    round(bb.x2, 1), round(bb.y2, 1),
                    round(bb.width, 1), round(bb.height, 1),
                    round(bb.conf, 4),
                    bb.cls_id,
                ])

    logger.info("CSV export: %d tracklets → %s", len(tracklets), path)


def export_summary_json(tracklets: list[Tracklet], output_path: str) -> None:
    """Export tracklet summaries grouped by global_id as a formatted JSON file.

    Args:
        tracklets: All completed tracklets.
        output_path: Path to the output JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[dict]] = {}
    for t in tracklets:
        gid = t.global_id if t.global_id is not None else -1
        grouped.setdefault(gid, []).append(t.to_summary())

    # Sort within each group by start_time
    for summaries in grouped.values():
        summaries.sort(key=lambda s: s["start_time"])

    output = {
        "total_tracklets": len(tracklets),
        "total_global_targets": len(grouped),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "targets": {
            str(gid): summaries
            for gid, summaries in sorted(grouped.items())
        },
    }

    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("JSON export: %d targets → %s", len(grouped), path)
