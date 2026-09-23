from __future__ import annotations

import cv2
import numpy as np

from .models import OverlayCandidate, Rect


def _gradient_map(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 97))
    if scale <= 1e-6:
        return np.zeros_like(magnitude)
    return np.clip(magnitude / scale, 0.0, 1.0)


def build_persistence_map(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) < 4:
        raise ValueError("at least 4 sampled frames are required")

    gradients = np.stack([_gradient_map(frame) for frame in frames], axis=0)
    edge_votes: list[np.ndarray] = []
    for gradient in gradients:
        threshold = max(0.16, float(np.percentile(gradient, 78)))
        edge_votes.append(gradient >= threshold)

    persistence = np.mean(np.stack(edge_votes, axis=0), axis=0).astype(np.float32)
    median_strength = np.median(gradients, axis=0).astype(np.float32)
    return persistence, median_strength


def detect_static_overlay_candidates(
    frames: list[np.ndarray],
    *,
    min_persistence: float = 0.58,
    max_candidates: int = 12,
) -> list[OverlayCandidate]:
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all sampled frames must have the same size")

    persistence, strength = build_persistence_map(frames)

    raw = ((persistence >= min_persistence) & (strength >= 0.12)).astype(np.uint8) * 255

    horizontal = max(5, int(round(width * 0.006)))
    if horizontal % 2 == 0:
        horizontal += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    raw = cv2.dilate(raw, np.ones((3, 3), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    frame_area = height * width
    min_area = max(16, int(frame_area * 0.00001))
    max_area = int(frame_area * 0.15)

    candidates: list[OverlayCandidate] = []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        bbox_area = w * h

        if area < min_area or bbox_area > max_area:
            continue
        if w < 4 or h < 3:
            continue
        if w > width * 0.80 or h > height * 0.40:
            continue

        component = labels == label
        component_persistence = float(np.mean(persistence[component]))
        component_strength = float(np.mean(strength[component]))

        # Position is only a weak prior. This avoids hard-coding "watermarks live in corners".
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        edge_distance = min(cx, 1.0 - cx, cy, 1.0 - cy)
        edge_prior = float(np.clip((0.35 - edge_distance) / 0.35, 0.0, 1.0))

        fill_ratio = float(area / max(1, bbox_area))
        score = (
            0.55 * component_persistence
            + 0.25 * component_strength
            + 0.10 * edge_prior
            + 0.10 * min(fill_ratio * 4.0, 1.0)
        )

        base_rect = Rect(x, y, w, h)
        padding = max(4, int(round(min(w, h) * 0.12)))
        rect = base_rect.expand(padding, width, height)

        local_mask = component[rect.y : rect.y2, rect.x : rect.x2].astype(np.uint8)
        local_mask = cv2.morphologyEx(
            local_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        local_mask = cv2.dilate(local_mask, np.ones((3, 3), np.uint8), iterations=1)

        candidates.append(
            OverlayCandidate(
                rect=rect,
                mask=local_mask,
                detector_score=float(np.clip(score, 0.0, 1.0)),
                persistence=component_persistence,
            )
        )

    candidates.sort(key=lambda item: item.detector_score, reverse=True)
    return candidates[:max_candidates]


def candidate_from_full_frame_mask(mask: np.ndarray, frame_shape: tuple[int, int]) -> OverlayCandidate:
    height, width = frame_shape
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = mask > 127
    ys, xs = np.where(binary)
    if xs.size == 0:
        raise ValueError("manual mask contains no white pixels")

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    base = Rect(x1, y1, x2 - x1, y2 - y1)
    rect = base.expand(max(4, int(round(min(base.width, base.height) * 0.12))), width, height)
    local = binary[rect.y : rect.y2, rect.x : rect.x2].astype(np.uint8)

    return OverlayCandidate(
        rect=rect,
        mask=local,
        detector_score=1.0,
        persistence=1.0,
    )
