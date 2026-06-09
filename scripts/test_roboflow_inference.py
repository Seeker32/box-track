"""Test script for Roboflow serverless inference using box-detection-sz4gh/5 model.

Runs inference on a local image or video, filters detections to the "cardboard" class,
draws bounding boxes, and saves the annotated result to the output directory.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

load_dotenv()
# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.tracklet import BBox
from src.detection.roboflow_detector import RoboflowDetector

logger = logging.getLogger(__name__)

# ---------- Roboflow client config ----------
API_URL = "https://serverless.roboflow.com"
API_KEY = os.getenv("API_KEY")
MODEL_ID = "box-detection-sz4gh-dum2a/2"
TARGET_CLASS = "cardboard"

# BGR color for cardboard boxes (green)
BOX_COLOR = (0, 255, 0)

# ---------- Helpers ----------


def create_client() -> InferenceHTTPClient:
    """Create and return a configured Roboflow inference client."""
    return InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)


def run_inference_on_image(
    client: InferenceHTTPClient, image_path: str
) -> list[dict]:
    """Run inference and return predictions filtered to the target class.

    Args:
        client: Configured InferenceHTTPClient.
        image_path: Path to the input image.

    Returns:
        List of prediction dicts for the target class only.
    """
    result = client.infer(image_path, model_id=MODEL_ID)
    predictions = result.get("predictions", [])

    cardboard_preds = [
        p for p in predictions
        if p.get("class", "").lower() == TARGET_CLASS
    ]

    return cardboard_preds


def run_inference_on_frame(
    client: InferenceHTTPClient, frame: np.ndarray
) -> list[dict]:
    """Run inference on an in-memory frame (encoded as JPEG bytes).

    Args:
        client: Configured InferenceHTTPClient.
        frame: BGR image array (numpy).

    Returns:
        List of prediction dicts for the target class only.
    """
    _, buffer = cv2.imencode(".jpg", frame)
    result = client.infer(buffer.tobytes(), model_id=MODEL_ID)
    predictions = result.get("predictions", [])

    cardboard_preds = [
        p for p in predictions
        if p.get("class", "").lower() == TARGET_CLASS
    ]

    return cardboard_preds


def predictions_to_bboxes(
    predictions: list[dict], frame_id: int = 0
) -> list[BBox]:
    """Convert Roboflow predictions to project BBox objects.

    Args:
        predictions: List of prediction dicts from Roboflow.
        frame_id: Frame ID to assign.

    Returns:
        List of BBox objects.
    """
    return RoboflowDetector.predictions_to_bboxes(
        predictions,
        frame_id=frame_id,
        target_class=TARGET_CLASS,
        conf=0.0,
        class_id=0,
    )


def draw_bboxes(
    image: np.ndarray,
    bboxes: list[BBox],
    color: tuple[int, int, int] = BOX_COLOR,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes with labels on the image (in-place).

    Args:
        image: BGR image (modified in place).
        bboxes: List of BBox objects to draw.
        color: BGR color tuple for boxes.
        thickness: Line thickness.

    Returns:
        The annotated image.
    """
    for bbox in bboxes:
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Label with confidence
        label = f"cardboard {bbox.conf:.2f}"
        _draw_label(image, label, x1, y1, color)

    return image


def _draw_label(
    image: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw a text label with filled background above the bbox.

    Adapted from src/utils/visualization.py conventions.
    """
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_y = max(y - th - baseline - 4, 0)

    # Filled background
    cv2.rectangle(
        image,
        (x, label_y),
        (x + tw + 4, label_y + th + baseline + 2),
        color,
        cv2.FILLED,
    )
    # White text on green background
    cv2.putText(
        image,
        label,
        (x + 2, label_y + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
    )


# ---------- Processing pipelines ----------


def process_image(
    client: InferenceHTTPClient,
    image_path: Path,
    project_root: Path,
    conf_threshold: float,
    output_path: Path | None,
    no_save: bool,
) -> None:
    """Process a single image: inference, draw, save."""
    logger.info("Loading image: %s", image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Failed to read image: %s", image_path)
        sys.exit(1)
    logger.info("Image size: %dx%d", image.shape[1], image.shape[0])

    logger.info("Running inference with model: %s", MODEL_ID)
    predictions = run_inference_on_image(client, str(image_path))
    logger.info("Raw cardboard predictions: %d", len(predictions))

    predictions = [p for p in predictions if p.get("confidence", 0) >= conf_threshold]
    logger.info(
        "After confidence filter (>=%.2f): %d predictions",
        conf_threshold,
        len(predictions),
    )

    if not predictions:
        logger.warning(
            "No cardboard boxes detected above confidence %.2f. Try lowering --conf.",
            conf_threshold,
        )
        return

    _print_detections(predictions)

    bboxes = predictions_to_bboxes(predictions)
    annotated = image.copy()
    draw_bboxes(annotated, bboxes)

    if not no_save:
        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            stem = image_path.stem
            output_path = output_dir / f"{stem}_roboflow.jpg"
        elif not output_path.is_absolute():
            output_path = project_root / output_path
        cv2.imwrite(str(output_path), annotated)
        logger.info("Annotated image saved to: %s", output_path)


def process_video(
    client: InferenceHTTPClient,
    video_path: Path,
    project_root: Path,
    conf_threshold: float,
    output_path: Path | None,
    frame_skip: int,
    display: bool,
) -> None:
    """Process a video file: read frames, run inference, produce annotated video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("Video: %dx%d, %.1f fps, %d total frames", width, height, fps, total_frames)
    logger.info("Processing every %d frame(s)", frame_skip)

    # Setup output video writer
    writer = None
    if output_path is not None:
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / frame_skip
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))
        logger.info("Output video will be saved to: %s (%.1f fps)", output_path, out_fps)

    frame_idx = 0
    processed_count = 0
    total_inference_time = 0.0

    logger.info("Starting video processing (press 'q' to quit early)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Run inference
        t_start = time.perf_counter()
        try:
            predictions = run_inference_on_frame(client, frame)
        except Exception as e:
            logger.error("Inference failed on frame %d: %s", frame_idx, e)
            frame_idx += 1
            continue
        elapsed = time.perf_counter() - t_start
        total_inference_time += elapsed
        processed_count += 1

        # Filter by confidence
        predictions = [p for p in predictions if p.get("confidence", 0) >= conf_threshold]

        # Draw bboxes
        if predictions:
            bboxes = predictions_to_bboxes(predictions, frame_id=frame_idx)
            draw_bboxes(frame, bboxes)

        # Overlay frame info
        avg_time = total_inference_time / processed_count if processed_count > 0 else 0.0
        info_lines = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Detections: {len(predictions)}",
            f"Inference: {elapsed*1000:.0f}ms (avg: {avg_time*1000:.0f}ms)",
        ]
        for i, text in enumerate(info_lines):
            cv2.putText(
                frame, text, (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
            )

        # Write output frame
        if writer is not None:
            writer.write(frame)

        # Display
        if display:
            cv2.imshow("Roboflow Inference - cardboard detection", frame)

        frame_idx += 1

        # Progress every 30 frames
        if processed_count % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            logger.info(
                "Progress: %.1f%% (%d/%d), avg inference: %.0fms",
                progress, frame_idx, total_frames, avg_time * 1000,
            )

        # Quit on 'q'
        if display and cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("User requested early exit.")
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Summary
    if processed_count == 0:
        logger.warning(
            "No frames were processed. Check that the video file is readable "
            "and the API is reachable."
        )
        return

    logger.info("=" * 50)
    logger.info("Video processing complete!")
    logger.info("  Processed frames: %d", processed_count)
    logger.info("  Total inference time: %.1fs", total_inference_time)
    logger.info("  Avg inference time: %.0fms", (total_inference_time / processed_count) * 1000)
    if output_path is not None:
        logger.info("  Output saved to: %s", output_path)
    logger.info("=" * 50)


def _print_detections(predictions: list[dict]) -> None:
    """Print a formatted detection summary."""
    print(f"\n{'=' * 60}")
    print(f"Detected {len(predictions)} cardboard box(es):")
    print(f"{'=' * 60}")
    for i, p in enumerate(predictions, 1):
        print(
            f"  [{i}] conf={p['confidence']:.3f}  "
            f"x={p['x']:.1f} y={p['y']:.1f}  "
            f"w={p['width']:.1f} h={p['height']:.1f}"
        )
    print(f"{'=' * 60}\n")


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Roboflow inference for cardboard box detection"
    )

    # Input source (mutually exclusive: image or video)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image",
        default=None,
        help="Path to input image",
    )
    input_group.add_argument(
        "--video",
        default=None,
        help="Path to input video file",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path for annotated output (image: .jpg, video: .mp4). Default: output/<name>_roboflow.<ext>",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for filtering (default: 0.3)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the annotated output",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        metavar="N",
        help="Process every N-th frame for video (default: 1, process all frames)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show real-time preview window during video processing",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    client = create_client()

    # Determine input source
    if args.video:
        video_path = project_root / args.video if not Path(args.video).is_absolute() else Path(args.video)
        if not video_path.exists():
            logger.error("Video not found: %s", video_path)
            sys.exit(1)

        output_path = None
        if not args.no_save:
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = project_root / "output" / f"{video_path.stem}_roboflow.mp4"

        process_video(
            client=client,
            video_path=video_path,
            project_root=project_root,
            conf_threshold=args.conf,
            output_path=output_path,
            frame_skip=args.frame_skip,
            display=args.display,
        )
    else:
        # Default to image mode; use default image if neither --image nor --video given
        image_path = project_root / (args.image or "input/Image_20260608100512_11_42.jpg")
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            sys.exit(1)

        output_path = Path(args.output) if args.output else None
        process_image(
            client=client,
            image_path=image_path,
            project_root=project_root,
            conf_threshold=args.conf,
            output_path=output_path,
            no_save=args.no_save,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
