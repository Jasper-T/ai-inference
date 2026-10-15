"""Box operations on [..., 4] arrays; callers choose dtype and coordinate units."""

import numpy as np


def cxcywh_to_xyxy(boxes: np.ndarray, *, clamp_wh: bool = False) -> np.ndarray:
    wh = np.maximum(boxes[..., 2:], 0) if clamp_wh else boxes[..., 2:]
    return np.concatenate((boxes[..., :2] - wh / 2, boxes[..., :2] + wh / 2), axis=-1)


def restore_boxes(
    boxes: np.ndarray, scale: float | tuple[float, float] | None, padding: tuple[float, float]
) -> np.ndarray:
    """Accept scalar gain or (scale_x, scale_y), preserving model conventions."""
    if scale is None:
        raise ValueError("box restoration requires a scale or gain")
    divisor = [*scale, *scale] if isinstance(scale, tuple) else scale
    return (boxes - [*padding, *padding]) / divisor


def clip_boxes(boxes: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    h, w = original_shape
    return np.clip(boxes, 0, [w, h, w, h])


def nms(boxes, scores, ids, threshold, limit, agnostic):
    order = np.argsort(-scores, kind="stable")
    kept = []
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    while order.size and len(kept) < limit:
        i = order[0]
        kept.append(i)
        rest = order[1:]
        intersection = np.prod(
            np.maximum(
                np.minimum(boxes[i, 2:], boxes[rest, 2:])
                - np.maximum(boxes[i, :2], boxes[rest, :2]),
                0,
            ),
            axis=1,
        )
        union = area[i] + area[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        suppress = iou > threshold
        if not agnostic:
            suppress &= ids[rest] == ids[i]
        order = rest[~suppress]
    return np.asarray(kept, dtype=np.int64)
