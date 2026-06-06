#!/usr/bin/env python3
"""Box-Track: Multi-camera cardboard box tracking system.

Phase 1: YOLO backbone feature validation for cross-camera matching.
Phase 2: Complete multi-camera real-time tracking system.

Usage:
    # Phase 1: Feature validation
    python main.py --phase1 --mode separability
    python main.py --phase1 --mode synthetic
    python main.py --phase1 --mode video --videos cam0.mp4 cam1.mp4

    # Phase 2: Full system
    python main.py --phase2 --mode video --videos cam0.mp4 cam1.mp4
    python main.py --phase2 --mode rtsp --streams rtsp://... rtsp://...
    python main.py --phase2 --mode video --videos cam0.mp4 cam1.mp4 --viz --display
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_detector import YOLODetector
from src.features.backbone_feature import YOLOBackboneFeatureExtractor
from src.matching.hungarian_matcher import cross_camera_match
from src.matching.similarity import cosine_similarity
from src.pipeline.cross_camera_pipeline import CrossCameraPipeline
from src.pipeline.online_pipeline import OnlineCrossCameraPipeline
from src.utils.tracklet import BBox, Tracklet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("box-track")


# =========================================================================
# Phase 1: validation modes
# =========================================================================


def run_separability_test(image_path: str) -> dict:
    """Mode A: Test feature separability on a single image.

    Extracts backbone features for all detected boxes and measures
    pairwise cosine similarity. Boxes should be distinguishable.

    Args:
        image_path: Path to a test image.

    Returns:
        dict with separability metrics.
    """
    logger.info("=== Mode A: Feature Separability Test ===")
    logger.info("Image: %s", image_path)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    logger.info("Image size: %dx%d", image.shape[1], image.shape[0])

    # Initialize detector and feature extractor
    detector = YOLODetector("models/best.pt")
    extractor = YOLOBackboneFeatureExtractor("models/best.pt")

    # Detect boxes
    bboxes, _ = detector.detect(image)
    logger.info("Detected %d box(es)", len(bboxes))

    if len(bboxes) == 0:
        logger.warning("No boxes detected — cannot evaluate separability")
        return {"error": "no_detections"}

    # Print detection details
    for i, bb in enumerate(bboxes):
        logger.info(
            "  Box %d: x1=%.0f y1=%.0f x2=%.0f y2=%.0f (%.0fx%.0f) conf=%.2f",
            i, bb.x1, bb.y1, bb.x2, bb.y2, bb.width, bb.height, bb.conf,
        )

    # Extract features
    features = extractor.extract(image, bboxes)
    logger.info("Extracted %d feature vectors (dim=%d)", len(features),
                 features[0].shape[0] if features else 0)

    # Verify feature properties
    for i, feat in enumerate(features):
        norm = np.linalg.norm(feat)
        logger.info("  Feature %d: dim=%d, norm=%.4f, mean=%.4f, std=%.4f",
                     i, feat.shape[0], norm, float(feat.mean()), float(feat.std()))

    # Pairwise similarity matrix
    if len(features) >= 2:
        logger.info("\nPairwise cosine similarity matrix:")
        sim_matrix = np.zeros((len(features), len(features)))
        for i in range(len(features)):
            for j in range(len(features)):
                sim_matrix[i, j] = cosine_similarity(features[i], features[j])
                if i < j:
                    logger.info(f"  Feature {i} vs Feature {j}: {sim_matrix[i, j]:.4f}")

        off_diag = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                off_diag.append(sim_matrix[i, j])

        mean_sim = float(np.mean(off_diag)) if off_diag else 0.0
        min_sim = float(np.min(off_diag)) if off_diag else 0.0
        max_sim = float(np.max(off_diag)) if off_diag else 0.0

        logger.info("\nOff-diagonal similarity stats:")
        logger.info("  Mean: %.4f", mean_sim)
        logger.info("  Min:  %.4f", min_sim)
        logger.info("  Max:  %.4f", max_sim)

        # Verdict
        if max_sim < 0.8:
            logger.info("\n✅ PASS: All box pairs are well-separated (max sim < 0.8)")
        elif mean_sim < 0.7:
            logger.info("\n⚠️  WARN: Some box pairs are similar but overall OK")
        else:
            logger.info("\n❌ FAIL: Box features are too similar — may need multi-scale features")

        return {
            "num_boxes": len(bboxes),
            "feature_dim": features[0].shape[0],
            "mean_pairwise_sim": mean_sim,
            "min_pairwise_sim": min_sim,
            "max_pairwise_sim": max_sim,
            "passed": max_sim < 0.8,
        }
    else:
        logger.info("\nOnly 1 box detected — no pairwise comparison possible")
        return {
            "num_boxes": len(bboxes),
            "feature_dim": features[0].shape[0] if features else 0,
            "passed": None,
        }


def run_synthetic_test(image_path: str) -> dict:
    """Mode B: Synthetic cross-camera matching test.

    Simulates two "cameras" from a single image by adding Gaussian noise
    to features and testing if Hungarian matching can correctly pair
    each box with its noisy copy.

    Args:
        image_path: Path to a test image.

    Returns:
        dict with matching accuracy metrics.
    """
    logger.info("=== Mode B: Synthetic Cross-Camera Matching Test ===")
    logger.info("Image: %s", image_path)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detector = YOLODetector("models/best.pt")
    extractor = YOLOBackboneFeatureExtractor("models/best.pt")

    bboxes, _ = detector.detect(image)
    logger.info("Detected %d box(es)", len(bboxes))

    if len(bboxes) < 2:
        logger.warning("Need at least 2 detections for synthetic matching test")
        return {"error": "not_enough_detections", "num_boxes": len(bboxes)}

    features = extractor.extract(image, bboxes)

    # Split into two "cameras": first half = cam 0, second half = cam 1
    n = len(bboxes)
    split = n // 2

    cam_a_bboxes = bboxes[:split]
    cam_a_features = features[:split]
    cam_b_bboxes = bboxes[split:]
    cam_b_features = features[split:]

    logger.info("Camera A: %d boxes, Camera B: %d boxes", len(cam_a_bboxes), len(cam_b_bboxes))

    # Apply small Gaussian noise to simulate cross-camera variation
    noise_std = 0.05
    rng = np.random.RandomState(42)

    cam_b_features_noisy = []
    for feat in cam_b_features:
        noisy = feat + rng.normal(0, noise_std, size=feat.shape).astype(np.float32)
        noisy = noisy / (np.linalg.norm(noisy) + 1e-12)
        cam_b_features_noisy.append(noisy)

    # Build synthetic tracklets
    def make_tracklets(cam_id, bboxes, feats):
        tracklets = []
        for i, (bb, feat) in enumerate(zip(bboxes, feats)):
            t = Tracklet(camera_id=cam_id, local_id=i)
            t.bboxes.append(bb)
            t.features.append(feat)
            t.aggregated_feature = feat  # single-frame = feature itself
            tracklets.append(t)
        return tracklets

    cam_a_tracklets = make_tracklets(0, cam_a_bboxes, cam_a_features)
    cam_b_tracklets = make_tracklets(1, cam_b_bboxes, cam_b_features_noisy)

    # Create noisy copies of ALL features for intra/inter box comparison
    cam_b_all_features_noisy = []
    for feat in features:
        noisy = feat + rng.normal(0, noise_std, size=feat.shape).astype(np.float32)
        noisy = noisy / (np.linalg.norm(noisy) + 1e-12)
        cam_b_all_features_noisy.append(noisy)

    # Intra-box: feature[i] vs noisy_feature[i] (same box, different "camera")
    intra_sims = []
    for f_orig, f_noisy in zip(features, cam_b_all_features_noisy):
        intra_sims.append(cosine_similarity(f_orig, f_noisy))

    # Inter-box: feature[i] vs noisy_feature[j] for i != j
    inter_sims = []
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:
                inter_sims.append(cosine_similarity(features[i], cam_b_all_features_noisy[j]))

    mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    separation = mean_intra - mean_inter

    logger.info("  Mean intra-box similarity:  %.4f", mean_intra)
    logger.info("  Mean inter-box similarity:  %.4f", mean_inter)
    logger.info("  Separation margin:          %.4f", separation)

    if separation > 0.1:
        logger.info("\n✅ PASS: Intra-box similarity is significantly higher than inter-box")
        logger.info("   Backbone features can distinguish boxes across simulated cameras")
    elif separation > 0.0:
        logger.info("\n⚠️  WARN: Weak separation — features may struggle with harder cases")
    else:
        logger.info("\n❌ FAIL: Features cannot distinguish boxes")

    # Also run Hungarian matching to validate the pipeline
    cam_a_t = make_tracklets(0, bboxes, features)
    cam_b_t = make_tracklets(1, bboxes, cam_b_all_features_noisy)

    config = {
        "similarity_threshold": 0.5,
        "min_transit": 0.0,
        "max_transit": 999.0,
    }

    matches, unmatched_a, unmatched_b = cross_camera_match(cam_a_t, cam_b_t, config)

    # With identical boxes in both cameras, each should match to itself
    correct_matches = sum(1 for b_idx, a_idx in matches.items() if b_idx == a_idx)
    accuracy = correct_matches / len(features) if features else 0.0

    logger.info("\nHungarian matching (should match each box to its noisy copy):")
    logger.info("  Matches: %d/%d", len(matches), len(features))
    logger.info("  Correct (b_idx == a_idx): %d", correct_matches)
    logger.info("  Accuracy: %.2f%%", accuracy * 100)

    return {
        "num_boxes": len(bboxes),
        "mean_intra_sim": mean_intra,
        "mean_inter_sim": mean_inter,
        "separation_margin": separation,
        "matching_accuracy": accuracy,
        "passed": separation > 0.05 and accuracy > 0.5,
    }


def run_video_evaluation(video_paths: list[str]) -> dict:
    """Mode C: Full video-based cross-camera evaluation.

    Args:
        video_paths: List of video file paths (one per camera).

    Returns:
        dict with evaluation metrics.
    """
    logger.info("=== Mode C: Full Video Evaluation ===")
    logger.info("Videos: %s", video_paths)

    if len(video_paths) < 2:
        raise ValueError("Need at least 2 videos for cross-camera evaluation")

    pipeline = CrossCameraPipeline("configs/pipeline.yaml")

    # Process each camera's video
    all_tracklets: dict[int, list[Tracklet]] = {}
    for cam_id, video_path in enumerate(video_paths):
        tracklets = pipeline.process_video(cam_id, video_path)
        all_tracklets[cam_id] = tracklets

    # Run cross-camera matching
    all_tracklets = pipeline.run_match(all_tracklets)

    # Evaluate
    metrics = pipeline.evaluate(all_tracklets)

    logger.info("\n=== Evaluation Results ===")
    if "error" in metrics:
        logger.error("Evaluation error: %s", metrics["error"])
    else:
        logger.info("Top-1 Accuracy: %.2f%%", metrics["top1_accuracy"] * 100)
        logger.info("Correct: %d / %d", metrics["correct"], metrics["total_queries"])
        logger.info("Cameras: %d", metrics["num_cameras"])

        if metrics["top1_accuracy"] > 0.8:
            logger.info("\n✅ Phase 1 success criteria met (Top-1 > 80%%)")
        else:
            logger.info("\n⚠️  Top-1 accuracy below 80%% threshold")

    return metrics


# =========================================================================
# Phase 2: full system
# =========================================================================


def run_phase2_rtsp(streams: list[str], config_path: str, args: argparse.Namespace) -> None:
    """Run Phase 2 real-time tracking with RTSP streams.

    Args:
        streams: List of RTSP URLs (one per camera).
        config_path: Path to pipeline YAML config.
        args: Parsed command-line arguments.
    """
    logger.info("=== Phase 2: Real-Time RTSP Mode ===")
    logger.info("Streams: %s", streams)

    pipeline = OnlineCrossCameraPipeline(config_path)

    # Override config with CLI args
    if args.no_viz:
        pipeline.config.setdefault("online", {})["visualization"] = False
    if args.display:
        pipeline.config.setdefault("online", {})["display"] = True
    if args.no_persist:
        pipeline.config.setdefault("online", {})["persistence"] = False
    if args.output_dir:
        pipeline.config.setdefault("online", {})["output_video_dir"] = str(
            Path(args.output_dir) / "videos"
        )
        pipeline.config.setdefault("persistence", {})["output_dir"] = str(
            Path(args.output_dir) / "trajectories"
        )

    # Re-initialize to apply overrides
    pipeline.__init__(config_path)

    pipeline.setup_streams(streams)
    summary = pipeline.run()

    logger.info("Phase 2 RTSP pipeline finished: %s", summary)


def run_phase2_video(video_paths: list[str], config_path: str, args: argparse.Namespace) -> None:
    """Run Phase 2 batch processing with video files.

    Args:
        video_paths: List of video file paths (one per camera).
        config_path: Path to pipeline YAML config.
        args: Parsed command-line arguments.
    """
    logger.info("=== Phase 2: Video File Mode ===")
    logger.info("Videos: %s", video_paths)

    if len(video_paths) < 2:
        raise ValueError("Need at least 2 videos for cross-camera tracking")

    pipeline = OnlineCrossCameraPipeline(config_path)

    # Override config with CLI args
    if args.no_viz:
        pipeline.config.setdefault("online", {})["visualization"] = False
    if args.display:
        pipeline.config.setdefault("online", {})["display"] = True
    if args.no_persist:
        pipeline.config.setdefault("online", {})["persistence"] = False
    if args.output_dir:
        pipeline.config.setdefault("online", {})["output_video_dir"] = str(
            Path(args.output_dir) / "videos"
        )
        pipeline.config.setdefault("persistence", {})["output_dir"] = str(
            Path(args.output_dir) / "trajectories"
        )

    # Re-initialize to apply overrides
    pipeline.__init__(config_path)

    summary = pipeline.process_videos(video_paths)

    logger.info("Phase 2 video pipeline finished: %s", summary)


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Box-Track: Multi-camera cardboard box tracking system"
    )

    # Phase selection
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--phase1", action="store_true",
        help="Run Phase 1: feature validation",
    )
    phase_group.add_argument(
        "--phase2", action="store_true",
        help="Run Phase 2: full multi-camera tracking system",
    )

    # General arguments
    parser.add_argument(
        "--mode", choices=["separability", "synthetic", "video", "rtsp"],
        default="separability",
        help="Operation mode",
    )
    parser.add_argument(
        "--image", default="input/IMG_20260605_225534.jpg",
        help="Path to test image (for separability/synthetic modes)",
    )
    parser.add_argument(
        "--videos", nargs="+", help="Paths to test videos",
    )
    parser.add_argument(
        "--streams", nargs="+", help="RTSP/RTMP/HTTP stream URLs (one per camera)",
    )
    parser.add_argument(
        "--model", default="models/best.pt", help="Path to YOLO model",
    )
    parser.add_argument(
        "--config", default="configs/pipeline.yaml",
        help="Path to pipeline YAML config",
    )

    # Phase 2 flags
    parser.add_argument(
        "--viz", action="store_true", default=True,
        help="Enable visualization output (default: on)",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Disable visualization output",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show annotated frames in a window (requires GUI)",
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Disable trajectory persistence",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Base output directory (default: 'output/')",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------
    if args.phase2:
        if args.mode == "rtsp":
            if not args.streams:
                logger.error("--streams required for RTSP mode. Example:")
                logger.error("  python main.py --phase2 --mode rtsp "
                             "--streams rtsp://192.168.1.10/stream rtsp://192.168.1.11/stream")
                sys.exit(1)
            run_phase2_rtsp(args.streams, args.config, args)
            return

        elif args.mode == "video":
            if not args.videos:
                logger.error("--videos required for video mode. Example:")
                logger.error("  python main.py --phase2 --mode video --videos cam0.mp4 cam1.mp4")
                sys.exit(1)
            run_phase2_video(args.videos, args.config, args)
            return

        else:
            logger.error(
                "Phase 2 requires --mode rtsp or --mode video. Got: %s", args.mode
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    if args.phase1:
        if args.mode == "separability":
            result = run_separability_test(args.image)
        elif args.mode == "synthetic":
            result = run_synthetic_test(args.image)
        elif args.mode == "video":
            if not args.videos:
                logger.error("--videos required for video mode")
                sys.exit(1)
            result = run_video_evaluation(args.videos)
        else:
            logger.error("Unknown mode: %s", args.mode)
            sys.exit(1)

        # Summarize
        passed = result.get("passed")
        if passed is True:
            logger.info("\n✅ Phase 1 verification PASSED")
        elif passed is False:
            logger.warning("\n⚠️  Phase 1 verification: issues found")
        else:
            logger.info("\nℹ️  Phase 1 verification: completed (see results above)")
        return

    # ------------------------------------------------------------------
    # Legacy mode (no phase flag)
    # ------------------------------------------------------------------
    logger.info("Running legacy inference mode. Use --phase1 or --phase2 for specific modes.")
    image_path = args.image
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Cannot read image: %s", image_path)
        sys.exit(1)

    detector = YOLODetector(args.model)
    extractor = YOLOBackboneFeatureExtractor(args.model)

    bboxes, _ = detector.detect(image)
    features = extractor.extract(image, bboxes)

    logger.info("Detected %d boxes", len(bboxes))
    for i, (bb, feat) in enumerate(zip(bboxes, features)):
        logger.info("  Box %d: %.0fx%.0f @ (%.0f, %.0f), conf=%.2f, feat_dim=%d",
                     i, bb.width, bb.height, bb.x1, bb.y1, bb.conf, feat.shape[0])

    # Save annotated image
    from ultralytics import YOLO
    model = YOLO(args.model)
    results = model(image_path)
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(image_path).stem}_annotated.jpg"
    cv2.imwrite(str(out_path), results[0].plot())
    logger.info("Saved annotated image to %s", out_path)


if __name__ == "__main__":
    main()
