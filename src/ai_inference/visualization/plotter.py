"""Render existing results on BGR images without constructing/calling a model."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from ai_inference.contracts.data import FeaturePrediction, Prediction, Result
from ai_inference.logging import log_component_event

PALETTE = ((0, 200, 255), (80, 220, 80), (255, 160, 60), (200, 90, 255), (255, 220, 60))
# Ultralytics detection palette (RGB hex), stored as BGR for OpenCV.
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/plotting.py
CLASS_COLORS = tuple(
    (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))
    for value in (
        "042AFF",
        "0BDBEB",
        "F3F3F3",
        "00DFB7",
        "111F68",
        "FF6FDD",
        "FF444F",
        "CCED00",
        "00F344",
        "BD00FF",
        "00B4FF",
        "DD00BA",
        "00FFFF",
        "26C000",
        "01FFB3",
        "7D24FF",
        "7B0068",
        "FF1B6C",
        "FC6D2F",
        "A2FF0B",
    )
)


class Plotter:
    """Draw on an HWC uint8 BGR copy; caller thresholds probability masks.

    Features are described by dimension/norm, never interpreted as identity.
    Optional detection context supplies geometry for a feature result.
    """

    def __init__(self, *, mask_alpha: float = 0.3):
        if not np.isfinite(mask_alpha) or not 0 <= mask_alpha <= 1:
            raise ValueError("mask_alpha must be finite in [0,1]")
        self.mask_alpha = mask_alpha

    @staticmethod
    def label(class_id: int, names=None) -> str:
        if names is None:
            return str(class_id)
        if isinstance(names, Mapping):
            value = names.get(class_id)
        elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            value = names[class_id] if 0 <= class_id < len(names) else None
        else:
            raise TypeError("names must be a sequence or ID-to-name mapping")
        return str(class_id) if value is None else str(value)

    @staticmethod
    def color(class_id: int) -> tuple[int, int, int]:
        """Ultralytics-style stable class indexing over its 20 detection colors."""
        return CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]

    def _draw(self, canvas, result, thickness, scale, names):
        n = len(result.boxes)
        h, w = canvas.shape[:2]
        if (
            result.boxes.shape != (n, 4)
            or result.scores.shape != (n,)
            or result.class_ids.shape != (n,)
            or not np.isfinite(result.boxes).all()
            or not np.isfinite(result.scores).all()
        ):
            raise ValueError("invalid detection fields")
        if result.masks is not None:
            if result.masks.shape != (n, h, w) or not np.isfinite(result.masks).all():
                raise ValueError("masks must be finite original-resolution [K,H,W]")
            for mask, ident in zip(result.masks, result.class_ids):
                selected = mask != 0
                color = np.asarray(self.color(int(ident)))
                canvas[selected] = np.rint(
                    canvas[selected] * (1 - self.mask_alpha) + color * self.mask_alpha
                ).astype(np.uint8)
        if result.keypoints is not None:
            p = result.keypoints
            if p.ndim != 3 or p.shape[0] != n or p.shape[2] != 2 or not np.isfinite(p).all():
                raise ValueError("keypoints must be finite [K,P,2]")
        for i, (box, score, ident) in enumerate(zip(result.boxes, result.scores, result.class_ids)):
            color = self.color(int(ident))
            x1, y1 = np.floor(np.clip(box[:2], 0, [w - 1, h - 1])).astype(int)
            x2, y2 = np.maximum([x1, y1], np.ceil(np.clip(box[2:], 0, [w, h])).astype(int) - 1)
            cv2.rectangle(canvas, (x1, y1), (int(x2), int(y2)), color, thickness)
            label = f"{self.label(int(ident), names)} {float(score):.4f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            ty = max(th + base + 3, y1)
            cv2.rectangle(
                canvas, (x1, ty - th - base - 3), (min(w - 1, x1 + tw + 8), ty + 3), color, -1
            )
            text_color = (
                (255, 255, 255) if np.dot(color, (0.114, 0.587, 0.299)) < 150 else (10, 10, 10)
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 3, ty - base),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
            if result.keypoints is not None:
                for j, point in enumerate(result.keypoints[i]):
                    if 0 <= point[0] < w and 0 <= point[1] < h:
                        cv2.circle(
                            canvas,
                            tuple(np.rint(point).astype(int)),
                            max(2, thickness + 1),
                            PALETTE[j % len(PALETTE)],
                            -1,
                            cv2.LINE_AA,
                        )

    def render(
        self,
        image: np.ndarray,
        result: Result | None = None,
        *,
        title: str = "",
        note: str = "",
        context: Prediction | None = None,
        names=None,
        show_summary: bool = False,
    ) -> np.ndarray:
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("image must be nonempty HWC uint8 BGR")
        for item in (result, context):
            if item is not None and item.original_shape != image.shape[:2]:
                raise ValueError("result original_shape must match image")
        canvas = image.copy()
        thickness = max(1, round(min(image.shape[:2]) / 480))
        scale = max(0.45, min(image.shape[:2]) / 1100)
        if context is not None:
            self._draw(canvas, context, thickness, scale, names)
        if isinstance(result, Prediction):
            self._draw(canvas, result, thickness, scale, names)
            summary = f"detections: {len(result.boxes)}"
        elif isinstance(result, FeaturePrediction):
            if (
                result.embedding.ndim != 1
                or not result.embedding.size
                or not np.isfinite(result.embedding).all()
            ):
                raise ValueError("feature must be a finite nonempty vector")
            summary = f"feature dimensions: {result.embedding.size} | L2 norm: {np.linalg.norm(result.embedding):.6f}"
        elif result is None:
            summary = ""
        else:
            raise TypeError("unsupported plotting result")
        headline = " | ".join(s for s in (title, summary if show_summary else "") if s)
        lines = [s for s in (headline, note) if s]
        step = max(20, round(30 * scale))
        if lines:
            height = min(canvas.shape[0], 10 + step * len(lines))
            canvas[:height] = np.rint(canvas[:height] * 0.2).astype(np.uint8)
            for i, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (8, min(canvas.shape[0] - 1, step * (i + 1))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
        return canvas

    def save(
        self, path: str | Path, image: np.ndarray, result: Result | None = None, **kwargs
    ) -> Path:
        destination = Path(path)
        try:
            canvas = self.render(image, result, **kwargs)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), canvas):
                raise OSError(f"cannot write annotation: {destination}")
        except Exception as exc:
            log_component_event(
                "plotter",
                "save",
                f"Failed | error={type(exc).__name__}",
                level="WARNING",
                stage="visualize",
                path=str(destination),
                error_type=type(exc).__name__,
                status="failed",
            )
            raise
        log_component_event(
            "plotter",
            "save",
            f"Saved | path={destination}",
            stage="visualize",
            path=str(destination),
            status="completed",
        )
        return destination
