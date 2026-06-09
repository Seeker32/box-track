"""Roboflow inference detector wrapper."""

import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from src.utils.tracklet import BBox

load_dotenv()


class RoboflowDetector:
    """Run Roboflow serverless inference and return project BBox objects."""

    def __init__(
        self,
        api_url: str = "https://serverless.roboflow.com",
        api_key: str | None = None,
        api_key_env: str = "API_KEY",
        model_id: str = "box-detection-sz4gh-dum2a/2",
        target_class: str = "cardboard",
        conf: float = 0.7,
        class_id: int = 0,
        client: InferenceHTTPClient | None = None,
    ):
        self.api_url = api_url
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.model_id = model_id
        self.target_class = target_class
        self.conf = conf
        self.class_id = class_id
        self.client = client or InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key,
        )

    def detect(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> tuple[list[BBox], dict[str, Any]]:
        """Run detection on a BGR frame."""
        result = self.client.infer(image, model_id=self.model_id)
        bboxes = self.predictions_to_bboxes(
            result.get("predictions", []),
            frame_id=frame_id,
            timestamp=timestamp,
            target_class=self.target_class,
            conf=self.conf,
            class_id=self.class_id,
        )
        return bboxes, result

    @staticmethod
    def predictions_to_bboxes(
        predictions: list[dict[str, Any]],
        frame_id: int = 0,
        timestamp: float = 0.0,
        target_class: str = "cardboard",
        conf: float = 0.7,
        class_id: int = 0,
    ) -> list[BBox]:
        """Convert Roboflow center-format predictions into BBox objects."""
        bboxes: list[BBox] = []
        target = target_class.lower()

        for pred in predictions:
            if pred.get("class", "").lower() != target:
                continue
            confidence = float(pred.get("confidence", 0.0))
            if confidence < conf:
                continue

            x_center = float(pred["x"])
            y_center = float(pred["y"])
            width = float(pred["width"])
            height = float(pred["height"])

            bboxes.append(
                BBox(
                    x1=x_center - width / 2,
                    y1=y_center - height / 2,
                    x2=x_center + width / 2,
                    y2=y_center + height / 2,
                    conf=confidence,
                    cls_id=class_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )

        return bboxes
