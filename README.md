# Box-Track

**Multi-camera cardboard box tracking system** — tracking visually similar cardboard boxes across cameras with non-overlapping fields of view.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> **[中文版](README-CN.md)** — 访问本项目的简体中文文档。

---

## Overview

Box-Track solves the problem of **keeping the same ID on a cardboard box as it moves between cameras** in a surveillance/logistics setting. This is challenging because:

- Cardboard boxes **all look alike** (brown corrugated surface with minimal texture variation)
- Cameras have **non-overlapping views** — no spatiotemporal continuity cues
- Lighting, angle, and background differ across cameras

The key insight: **YOLO's backbone (neck) features** — extracted from the intermediate feature maps of the detection model — serve as powerful appearance descriptors that capture subtle texture, printing, and tape differences that traditional ReID features miss. And since we already run YOLO for detection, the features come **for free** (one model, two uses).

### Highlights

- **YOLO backbone feature matching** — extracts per-bbox feature vectors via a forward hook on the detection model's neck layer; no separate ReID model needed
- **BoT-SORT single-camera tracking** — per-camera tracklet generation with Ultralytics + BoT-SORT
- **Hungarian algorithm** — optimal assignment for cross-camera tracklet matching
- **Spatiotemporal gating** — time-window constraints filter implausible matches
- **Online + batch modes** — real-time RTSP streaming or offline video file processing
- **Consistent color visualization** — same global ID gets the same color across all camera views
- **Trajectory persistence** — JSON Lines, CSV, and summary JSON export

---

## System Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Camera 0    │   │  Camera 1    │   │  Camera N    │
│  (RTSP/File) │   │  (RTSP/File) │   │  (RTSP/File) │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  YOLO Detect │   │  YOLO Detect │   │  YOLO Detect │
│  + BoT-SORT  │   │  + BoT-SORT  │   │  + BoT-SORT  │
│  (per cam)   │   │  (per cam)   │   │  (per cam)   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       │  Tracklets       │  Tracklets       │  Tracklets
       ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────┐
│           Backbone Feature Extraction               │
│  (YOLO neck layer → ROI pooling → per-bbox vector)  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│           Cross-Camera Matching Engine              │
│  · Cosine similarity (backbone features)            │
│  · Spatiotemporal constraints (time window)         │
│  · Hungarian algorithm (optimal assignment)         │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Global ID Manager                      │
│  · New target registration                          │
│  · Cross-camera ID propagation                      │
│  · Expired target cleanup                           │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
box-track/
├── main.py                          # CLI entry point
├── pyproject.toml                   # Project metadata & dependencies
├── configs/
│   ├── pipeline.yaml                # Pipeline configuration
│   └── botsort.yaml                 # BoT-SORT tracker config
├── src/
│   ├── detection/
│   │   └── yolo_detector.py         # YOLO detection wrapper
│   ├── tracking/
│   │   └── botsort_tracker.py       # BoT-SORT single-camera tracking
│   ├── features/
│   │   └── backbone_feature.py      # YOLO neck hook + ROI pooling
│   ├── matching/
│   │   ├── similarity.py            # Cosine similarity & spatiotemporal scoring
│   │   └── hungarian_matcher.py     # Hungarian optimal assignment
│   ├── association/
│   │   └── global_id_manager.py     # Cross-camera ID registry
│   ├── pipeline/
│   │   ├── cross_camera_pipeline.py # Batch evaluation pipeline
│   │   └── online_pipeline.py       # Real-time streaming pipeline
│   ├── io/
│   │   └── __init__.py              # RTSP/file stream reader (multi-threaded)
│   └── utils/
│       ├── tracklet.py              # BBox & Tracklet data structures
│       ├── visualization.py         # Drawing, color palette, VideoWriter
│       └── persistence.py           # JSON Lines / CSV / JSON export
├── models/
│   ├── best.pt                      # Fine-tuned YOLO detection model
│   └── yolo26n.pt                   # YOLO26 nano pretrained weights
├── docs/
│   └── design.md                    # Full design document (Chinese)
├── input/                           # Test images (gitignored)
└── output/                          # Annotated images & tracking exports (gitignored)
```

---

## Quick Start

### Installation

```bash
# Requires Python 3.12+
pip install ultralytics pyyaml scipy opencv-python numpy
```

Or with uv:

```bash
uv sync
```

### Download / Prepare a Model

Place a fine-tuned YOLO model at `models/best.pt`. A YOLO26n pretrained model is also included for reference.

### Run Inference on a Single Image

```bash
python main.py --image input/test.jpg
```

---

## Usage Modes

### Phase 1: Feature Validation

Test whether YOLO backbone features can distinguish different cardboard boxes.

**Separability test** — measures pairwise cosine similarity between all detected boxes in one image:

```bash
python main.py --phase1 --mode separability --image input/IMG_20260605_225534.jpg
```

**Synthetic cross-camera test** — simulates two cameras by adding Gaussian noise to features, then tests if Hungarian matching correctly pairs each box:

```bash
python main.py --phase1 --mode synthetic --image input/IMG_20260605_225534.jpg
```

**Full video evaluation** — processes multi-camera video files and computes Top-1 matching accuracy:

```bash
python main.py --phase1 --mode video --videos cam0.mp4 cam1.mp4
```

### Phase 2: Full Tracking System

**Video file mode** (batch):

```bash
python main.py --phase2 --mode video --videos cam0.mp4 cam1.mp4
```

**RTSP real-time mode** (live streaming):

```bash
python main.py --phase2 --mode rtsp --streams rtsp://192.168.1.10/stream rtsp://192.168.1.11/stream
```

**Additional flags:**

| Flag | Description |
|---|---|
| `--viz` / `--no-viz` | Enable/disable annotated video output |
| `--display` | Show live preview window (requires GUI) |
| `--no-persist` | Disable trajectory logging |
| `--output-dir PATH` | Set base output directory |
| `--config PATH` | Custom pipeline YAML config |
| `--model PATH` | Custom YOLO model path |

---

## Configuration

### Pipeline config (`configs/pipeline.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `models/best.pt` | YOLO model weights |
| `conf` | `0.25` | Detection confidence threshold |
| `feature.hook_layer` | `-2` | Neck layer index for feature hook |
| `feature.normalize` | `true` | L2-normalize feature vectors |
| `feature.aggregation` | `mean` | Tracklet feature aggregation: `mean`, `median`, or `max_conf` |
| `matching.similarity_threshold` | `0.5` | Minimum similarity for a valid match |
| `matching.min_transit` | `1.0s` | Minimum transit time between cameras |
| `matching.max_transit` | `60.0s` | Maximum transit time between cameras |
| `global_id.disappearance_timeout` | `60.0s` | Seconds before a target is considered gone |
| `online.visualization` | `true` | Output annotated video files |
| `online.persistence` | `true` | Log trajectories to disk |
| `online.match_interval` | `0.5s` | Cross-camera matching interval (online mode) |

### BoT-SORT config (`configs/botsort.yaml`)

Standard Ultralytics BoT-SORT parameters. The feature extraction hook in `backbone_feature.py` replaces the built-in ReID module (`with_reid: False`).

---

## How It Works

### 1. Single-Camera Tracking

Each camera runs YOLO detection + BoT-SORT tracking independently. The `BOTSORTTracker` processes frames sequentially, maintaining a set of active `Tracklet` objects.

### 2. Backbone Feature Extraction

A forward hook is registered on the detection model's neck layer (the last C2f block before the Detect head, layer `-2`). After each forward pass, the feature map is captured. For each detected bbox, the corresponding region in the feature map is ROI-pooled via global average pooling, producing a compact feature vector.

Since the hook is on the **same model** used for detection, there is no separate inference step — the features come from the one forward pass that already runs detection.

### 3. Tracklet Feature Aggregation

Each tracklet accumulates per-frame features over its lifetime. When the tracklet ends (the target leaves the camera's view), per-frame features are aggregated into a single vector using:
- **mean** — average of all frame features
- **median** — median across frames (more robust to outliers)
- **max_conf** — feature from the highest-confidence detection

### 4. Cross-Camera Matching

When a tracklet completes in a later camera, it's matched against active global targets using:
1. **Spatiotemporal gate** — filters out pairs with implausible time gaps or same-camera tracks
2. **Backbone feature similarity** — cosine similarity between aggregated tracklet features
3. **Hungarian algorithm** — finds the optimal one-to-one assignment maximizing total similarity

### 5. Global ID Management

- On first appearance in any camera → new global ID
- On match to an existing target → inherits that global ID
- On disappearance_timeout → target removed from active list
- IDs are assigned once and never revised (simple, reliable)

---

## Outputs

When running Phase 2 with persistence enabled, the following files are generated in `output/trajectories/`:

| Format | File | Description |
|---|---|---|
| JSON Lines | `trajectories_<timestamp>.jsonl` | One JSON object per completed tracklet |
| CSV | `detections_<timestamp>.csv` | Flat table of all detections, one row per bbox |
| JSON Summary | `summary_<timestamp>.json` | Tracklets grouped by global_id |

Annotated videos are saved to `output/videos/` when visualization is enabled.

---

## Design Rationale

### Why YOLO backbone features instead of a separate ReID model?

- **Zero extra inference cost** — the feature map is already computed during detection
- **Semantic features** — the neck layer encodes high-level patterns (texture, printing, tape, structural edges) that generalize across cameras
- **One model, two uses** — simplifies deployment and reduces memory footprint
- For this domain, backbone features outperform off-the-shelf ReID models that are trained on person/vehicle datasets

### Why Hungarian over greedy matching?

Hungarian optimal assignment avoids cascading errors where a greedy early match forces a wrong assignment later. This matters when multiple boxes have similar appearance scores.

### Why "assign once, never revise"?

Cross-camera matching is inherently ambiguous, so online refinement (revisiting past matches) can introduce instability. A one-pass approach trades optimality for predictability and simplicity.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Design Document

A detailed design document (in Chinese) covering architecture decisions, feature extraction strategy, and implementation roadmap is available at [`docs/design.md`](docs/design.md).
