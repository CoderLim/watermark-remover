from __future__ import annotations

import math

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


def _repeat_edge_map(
    frame: np.ndarray,
    *,
    max_width: int = 480,
) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / max(1, width))
    small_width = max(32, int(round(width * scale)))
    small_height = max(32, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    small = cv2.resize(frame, (small_width, small_height), interpolation=interpolation)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Fixed thresholds are intentional here. Auto-Canny tends to over-detect textured
    # backgrounds and weakens the periodic watermark signal.
    edges = cv2.Canny(gray, 80, 160) > 0
    return edges, scale


def _autocorrelation_peaks(
    edges: np.ndarray,
    *,
    max_peaks: int = 16,
) -> list[tuple[float, int, int]]:
    signal = edges.astype(np.float32)
    signal -= float(signal.mean())

    energy = float(np.sum(signal * signal))
    if energy <= 1e-6:
        return []

    spectrum = np.fft.fft2(signal)
    correlation = np.fft.fftshift(
        np.fft.ifft2(spectrum * np.conj(spectrum)).real
    )

    height, width = edges.shape
    center_y, center_x = height // 2, width // 2
    center = float(correlation[center_y, center_x])
    if abs(center) <= 1e-9:
        return []

    correlation = correlation / center

    yy, xx = np.indices(correlation.shape)
    dx = xx - center_x
    dy = yy - center_y

    min_x = max(6, int(round(width * 0.12)))
    min_y = max(6, int(round(height * 0.12)))
    valid = (
        ((np.abs(dx) >= min_x) | (np.abs(dy) >= min_y))
        & (np.abs(dx) <= int(round(width * 0.60)))
        & (np.abs(dy) <= int(round(height * 0.60)))
    )

    work = np.where(valid, correlation, -1.0)
    nms_radius = max(4, int(round(min(width, height) * 0.025)))

    peaks: list[tuple[float, int, int]] = []
    for _ in range(max_peaks * 3):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        score = float(work[y, x])
        if score < 0.025:
            break

        vector_x = int(x - center_x)
        vector_y = int(y - center_y)

        # Collapse +/- autocorrelation symmetry to one canonical vector.
        if vector_x < 0 or (vector_x == 0 and vector_y < 0):
            vector_x = -vector_x
            vector_y = -vector_y

        duplicate = any(
            (vector_x - existing_x) ** 2 + (vector_y - existing_y) ** 2
            <= nms_radius**2
            for _, existing_x, existing_y in peaks
        )
        if not duplicate:
            peaks.append((score, vector_x, vector_y))

        work[
            max(0, y - nms_radius) : min(height, y + nms_radius + 1),
            max(0, x - nms_radius) : min(width, x + nms_radius + 1),
        ] = -1.0

        if len(peaks) >= max_peaks:
            break

    peaks.sort(key=lambda item: item[0], reverse=True)
    return peaks


def _choose_lattice_vectors(
    peaks: list[tuple[float, int, int]],
) -> list[tuple[float, int, int]]:
    if not peaks:
        return []

    top_score = peaks[0][0]
    threshold = max(0.025, top_score * 0.70)
    strong = [peak for peak in peaks if peak[0] >= threshold]

    # Autocorrelation often gives stronger harmonics than the primitive period.
    # Among strong peaks, prefer the shortest vector, then a short non-collinear one.
    strong.sort(key=lambda item: (math.hypot(item[1], item[2]), -item[0]))
    first = strong[0]
    selected = [first]

    first_norm = math.hypot(first[1], first[2])
    for candidate in strong[1:]:
        candidate_norm = math.hypot(candidate[1], candidate[2])
        cross = abs(
            first[1] * candidate[2] - first[2] * candidate[1]
        ) / max(first_norm * candidate_norm, 1e-9)

        if cross > 0.55:
            selected.append(candidate)
            break

    return selected


def _repeat_support(
    edges: np.ndarray,
    lattice_vectors: list[tuple[float, int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    height, width = edges.shape
    first = lattice_vectors[0]
    first_vector = (first[1], first[2])

    shifts: list[tuple[int, int]] = [
        first_vector,
        (-first_vector[0], -first_vector[1]),
    ]

    if len(lattice_vectors) > 1:
        second = lattice_vectors[1]
        second_vector = (second[1], second[2])
        a = first_vector
        b = second_vector
        shifts.extend(
            [
                b,
                (-b[0], -b[1]),
                (a[0] + b[0], a[1] + b[1]),
                (-a[0] - b[0], -a[1] - b[1]),
                (a[0] - b[0], a[1] - b[1]),
                (-a[0] + b[0], -a[1] + b[1]),
            ]
        )

    unique_shifts: list[tuple[int, int]] = []
    for shift in shifts:
        if shift == (0, 0):
            continue
        if abs(shift[0]) >= width or abs(shift[1]) >= height:
            continue
        if shift not in unique_shifts:
            unique_shifts.append(shift)

    support = np.zeros(edges.shape, dtype=np.uint8)
    for shift_x, shift_y in unique_shifts:
        shifted = np.zeros_like(edges)

        y0 = max(0, -shift_y)
        y1 = min(height, height - shift_y)
        x0 = max(0, -shift_x)
        x1 = min(width, width - shift_x)
        if y1 <= y0 or x1 <= x0:
            continue

        shifted[y0:y1, x0:x1] = edges[
            y0 + shift_y : y1 + shift_y,
            x0 + shift_x : x1 + shift_x,
        ]
        support += (edges & shifted).astype(np.uint8)

    threshold = 3 if len(unique_shifts) >= 6 else 1
    return support, unique_shifts, threshold


def detect_repeated_overlay_candidates(
    frames: list[np.ndarray],
    *,
    max_candidates: int = 16,
) -> list[OverlayCandidate]:
    """
    Detect tiled/repeated overlays without knowing the provider or logo template.

    The detector looks for a spatial lattice in the frame's edge autocorrelation.
    Unique HUD elements such as REC, timers, and battery icons do not form a lattice,
    so they are naturally excluded from this path.
    """
    if not frames:
        return []

    frame_height, frame_width = frames[0].shape[:2]
    edges, scale = _repeat_edge_map(frames[0])
    peaks = _autocorrelation_peaks(edges)
    lattice_vectors = _choose_lattice_vectors(peaks)

    if not lattice_vectors or lattice_vectors[0][0] < 0.03:
        return []

    support, shifts, support_threshold = _repeat_support(edges, lattice_vectors)
    repeated = (support >= support_threshold).astype(np.uint8) * 255

    small_height, small_width = edges.shape
    kernel_width = max(5, int(round(small_width * 0.027)))
    kernel_height = max(3, int(round(small_height * 0.026)))
    if kernel_width % 2 == 0:
        kernel_width += 1
    if kernel_height % 2 == 0:
        kernel_height += 1

    repeated = cv2.morphologyEx(
        repeated,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_width, kernel_height),
        ),
        iterations=1,
    )
    repeated = cv2.dilate(repeated, np.ones((3, 3), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        repeated,
        connectivity=8,
    )

    components: list[tuple[int, int, int, int, int, int]] = []
    min_area = max(20, int(round(repeated.size * 0.0004)))

    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]

        if area < min_area:
            continue
        if width < max(8, int(round(small_width * 0.025))):
            continue
        if height < max(6, int(round(small_height * 0.025))):
            continue
        if width * height > repeated.size * 0.08:
            continue

        components.append((label, area, x, y, width, height))

    if len(components) < 3:
        return []

    # A tiled watermark produces several similarly sized components. Keep the dominant
    # size cluster and discard incidental periodic background fragments.
    largest_area = max(component[1] for component in components)
    components = [
        component
        for component in components
        if component[1] >= largest_area * 0.50
    ]
    if len(components) < 3:
        return []

    median_width = float(np.median([component[4] for component in components]))
    median_height = float(np.median([component[5] for component in components]))
    components = [
        component
        for component in components
        if (
            median_width * 0.55 <= component[4] <= median_width * 1.80
            and median_height * 0.55 <= component[5] <= median_height * 1.80
        )
    ]
    if len(components) < 3:
        return []

    components.sort(key=lambda item: item[1], reverse=True)
    components = components[:max_candidates]

    mean_repeat_score = float(np.mean([vector[0] for vector in lattice_vectors]))
    detector_score = float(
        np.clip(0.58 + mean_repeat_score * 4.0, 0.0, 1.0)
    )

    candidates: list[OverlayCandidate] = []
    inverse_scale = 1.0 / max(scale, 1e-9)

    for label, _, x, y, width, height in components:
        full_x = int(round(x * inverse_scale))
        full_y = int(round(y * inverse_scale))
        full_width = max(1, int(round(width * inverse_scale)))
        full_height = max(1, int(round(height * inverse_scale)))

        base_rect = Rect(
            full_x,
            full_y,
            min(full_width, frame_width - full_x),
            min(full_height, frame_height - full_y),
        )
        padding = max(4, int(round(min(base_rect.width, base_rect.height) * 0.10)))
        rect = base_rect.expand(padding, frame_width, frame_height)

        small_component = (labels == label).astype(np.uint8)
        full_component = cv2.resize(
            small_component,
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST,
        )
        local_mask = full_component[
            rect.y : rect.y2,
            rect.x : rect.x2,
        ].astype(np.uint8)
        local_mask = cv2.dilate(
            local_mask,
            np.ones((5, 5), np.uint8),
            iterations=1,
        )

        candidates.append(
            OverlayCandidate(
                rect=rect,
                mask=local_mask,
                detector_score=detector_score,
                persistence=0.0,
                source="spatial-repeat",
                group_id="repeat-0",
                repeat_score=mean_repeat_score,
            )
        )

    return candidates


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
                source="temporal",
            )
        )

    candidates.sort(key=lambda item: item.detector_score, reverse=True)
    return candidates[:max_candidates]


def candidate_from_full_frame_mask(
    mask: np.ndarray,
    frame_shape: tuple[int, int],
) -> OverlayCandidate:
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
    rect = base.expand(
        max(4, int(round(min(base.width, base.height) * 0.12))),
        width,
        height,
    )
    local = binary[rect.y : rect.y2, rect.x : rect.x2].astype(np.uint8)

    return OverlayCandidate(
        rect=rect,
        mask=local,
        detector_score=1.0,
        persistence=1.0,
        source="manual",
    )
