"""YOLO backbone feature extraction via forward hook + ROI pooling."""

import logging

import numpy as np
import torch
from ultralytics import YOLO

from src.utils.tracklet import BBox

logger = logging.getLogger(__name__)


class YOLOBackboneFeatureExtractor:
    """Extract per-bbox feature vectors from YOLO backbone neck layer.

    Registers a forward hook on a specified neck layer (default: last C2f
    before Detect head, index -2 in model.model.model). On each forward pass,
    the hook captures the feature map. The extract() method then crops each
    bbox region from the feature map and applies global average pooling to
    produce a feature vector.

    Usage:
        extractor = YOLOBackboneFeatureExtractor("models/best.pt")
        image = cv2.imread("frame.jpg")
        features = extractor.extract(image, bboxes)  # list of np.ndarray
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        hook_layer: int = -2,
        normalize: bool = True,
        model: YOLO | None = None,
    ):
        """Initialize the extractor.

        Args:
            model_path: Path to the YOLO model weights (ignored if model is provided).
            hook_layer: Index of the layer to hook in model.model.model.
                        Default -2 = last C2f in PAN neck (layer 21 in YOLO26).
            normalize: If True, L2-normalize each extracted feature vector.
            model: Optional pre-loaded YOLO model. If provided, the hook is
                   registered on this model (no second model is loaded).
        """
        self.model = model if model is not None else YOLO(model_path)
        self._owns_model = model is None
        self.normalize = normalize
        self._captured: dict[str, torch.Tensor] = {}

        # Register forward hook on the specified neck layer
        target_layer = self.model.model.model[hook_layer]
        layer_idx = len(self.model.model.model) + hook_layer if hook_layer < 0 else hook_layer
        logger.info(
            "Registered forward hook on layer %d (%s)", layer_idx, type(target_layer).__name__
        )
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """Capture the feature map from the hooked layer."""
        self._captured["neck"] = output.detach()

    def extract(
        self, image: np.ndarray, bboxes: list[BBox]
    ) -> list[np.ndarray]:
        """Extract backbone feature vectors for each bbox in the image.

        Runs a forward pass through the model (triggering the hook), then
        crops ROI regions from the captured feature map for each bbox.

        Args:
            image: Input image in BGR format (H, W, 3).
            bboxes: List of BBox objects in image pixel coordinates.

        Returns:
            List of feature vectors (numpy arrays), one per bbox.
            Returns empty list if the hook did not capture features.
        """
        if not bboxes:
            return []

        # Clear previous capture and run inference
        self._captured.clear()
        self.model(image, verbose=False)

        return self.extract_from_captured(image, bboxes)

    def extract_from_captured(
        self, image: np.ndarray, bboxes: list[BBox]
    ) -> list[np.ndarray]:
        """Extract features from the already-captured feature map (no inference).

        Use this after an external forward pass (e.g., model.track()) has
        triggered the hook. Avoids a redundant second inference.

        Args:
            image: Input image in BGR format (H, W, 3).
            bboxes: List of BBox objects in image pixel coordinates.

        Returns:
            List of feature vectors, or empty list if no feature map is captured.
        """
        if not bboxes:
            return []

        if "neck" not in self._captured:
            logger.warning("No captured feature map available")
            return []

        feat_map = self._captured["neck"]
        if feat_map.dim() == 3:
            feat_map = feat_map.unsqueeze(0)
        feat_map = feat_map[0]  # (C, H_feat, W_feat)

        c, h_feat, w_feat = feat_map.shape
        img_h, img_w = image.shape[:2]

        scale_h = h_feat / img_h
        scale_w = w_feat / img_w

        features: list[np.ndarray] = []
        for bbox in bboxes:
            fx1 = int(bbox.x1 * scale_w)
            fy1 = int(bbox.y1 * scale_h)
            fx2 = int(bbox.x2 * scale_w)
            fy2 = int(bbox.y2 * scale_h)

            # Clamp to valid range
            fx1 = max(0, min(fx1, w_feat - 1))
            fy1 = max(0, min(fy1, h_feat - 1))
            fx2 = max(fx1 + 1, min(fx2, w_feat))
            fy2 = max(fy1 + 1, min(fy2, h_feat))

            roi = feat_map[:, fy1:fy2, fx1:fx2]
            feat = roi.mean(dim=(1, 2))

            feat_np = feat.cpu().numpy().astype(np.float32)

            if self.normalize:
                norm = np.linalg.norm(feat_np)
                if norm > 1e-12:
                    feat_np = feat_np / norm

            features.append(feat_np)

        return features

    @property
    def feature_dim(self) -> int:
        """Return the feature vector dimension."""
        if "neck" in self._captured:
            return self._captured["neck"].shape[1]
        # Run a dummy forward pass to determine dim
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.extract(dummy, [BBox(0, 0, 32, 32, 1.0, 0)])
        if "neck" in self._captured:
            return self._captured["neck"].shape[1]
        return 0

    def __del__(self):
        """Clean up the hook handle."""
        try:
            self._hook_handle.remove()
        except Exception:
            pass
