from __future__ import annotations

import base64
import math
import zlib
from dataclasses import dataclass

import cv2
import numpy as np

from .detection import (
    _autocorrelation_peaks,
    _choose_lattice_vectors,
    _repeat_edge_map,
)
from .models import OverlayModel, Rect
from .profile_data import HEYGEN_TILED_ALPHA_ZLIB_BASE64


@dataclass(frozen=True)
class KnownTiledProfile:
    id: str
    alpha: np.ndarray
    reference_step_x: float
    reference_step_y: float
    min_aggregate_score: float = 0.35
    min_median_tile_score: float = 0.28
    min_tile_score: float = 0.15
    min_strong_tiles: int = 3
    min_tiles: int = 3


@dataclass
class KnownProfileMatch:
    profile_id: str
    alpha: np.ndarray
    positions: list[Rect]
    aggregate_score: float
    tile_scores: list[float]
    confidence: float
    step_x: float
    step_y: float
    scale: float

    def to_models(self) -> list[OverlayModel]:
        mask = (self.alpha > 0).astype(np.uint8)
        rgb = np.ones((*self.alpha.shape, 3), dtype=np.float32)
        fit_error = np.ones(self.alpha.shape, dtype=np.float32)
        fit_error[mask.astype(bool)] = 0.0

        models: list[OverlayModel] = []
        for rect, tile_score in zip(self.positions, self.tile_scores, strict=True):
            diagnostics = {
                "source": "known-profile",
                "profile_id": self.profile_id,
                "aggregate_score": self.aggregate_score,
                "tile_score": tile_score,
                "step_x": self.step_x,
                "step_y": self.step_y,
                "profile_scale": self.scale,
                "safety_fallback": "keep-original",
            }
            models.append(
                OverlayModel(
                    rect=rect,
                    mask=mask.copy(),
                    alpha=self.alpha.copy(),
                    rgb=rgb.copy(),
                    fit_error=fit_error.copy(),
                    confidence=self.confidence,
                    method="reverse-alpha-profile",
                    diagnostics=diagnostics,
                )
            )
        return models


def _decode_alpha(
    encoded: str,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    payload = zlib.decompress(base64.b64decode(encoded))
    expected = width * height
    if len(payload) != expected:
        raise RuntimeError(
            f"embedded alpha profile has {len(payload)} bytes, expected {expected}"
        )
    return (
        np.frombuffer(payload, dtype=np.uint8)
        .reshape(height, width)
        .astype(np.float32)
        / 255.0
    )


_HEYGEN_ALPHA = _decode_alpha(
    HEYGEN_TILED_ALPHA_ZLIB_BASE64,
    width=180,
    height=170,
)


KNOWN_TILED_PROFILES: tuple[KnownTiledProfile, ...] = (
    KnownTiledProfile(
        id="heygen-tiled-v1",
        alpha=_HEYGEN_ALPHA,
        reference_step_x=420.0,
        reference_step_y=420.0,
    ),
)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _axis_aligned_steps(
    frame: np.ndarray,
) -> tuple[float, float] | None:
    edges, analysis_scale = _repeat_edge_map(frame)
    peaks = _autocorrelation_peaks(edges)
    lattice = _choose_lattice_vectors(peaks)

    if len(lattice) < 2:
        return None

    full = [
        (
            score,
            float(dx) / analysis_scale,
            float(dy) / analysis_scale,
        )
        for score, dx, dy in lattice
    ]

    horizontal = max(
        full,
        key=lambda item: abs(item[1]) - abs(item[2]),
    )
    vertical = max(
        full,
        key=lambda item: abs(item[2]) - abs(item[1]),
    )

    hx, hy = horizontal[1], horizontal[2]
    vx, vy = vertical[1], vertical[2]

    if abs(hx) < 1 or abs(vy) < 1:
        return None

    # The calibrated tiled profile assumes an axis-aligned screen-space grid.
    # If the repetition lattice is skewed/rotated, leave it to the generic path.
    if abs(hy) > abs(hx) * 0.12:
        return None
    if abs(vx) > abs(vy) * 0.12:
        return None

    return abs(hx), abs(vy)


def _aggregate_grid_score(
    correlation: np.ndarray,
    *,
    step_x: int,
    step_y: int,
    min_tiles: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = correlation.shape
    accumulator = np.zeros_like(correlation, dtype=np.float32)
    counts = np.zeros_like(correlation, dtype=np.float32)

    radius_x = min(8, int(math.ceil(width / max(step_x, 1))) + 1)
    radius_y = min(8, int(math.ceil(height / max(step_y, 1))) + 1)

    for grid_y in range(-radius_y, radius_y + 1):
        for grid_x in range(-radius_x, radius_x + 1):
            dx = grid_x * step_x
            dy = grid_y * step_y

            y0 = max(0, -dy)
            y1 = min(height, height - dy)
            x0 = max(0, -dx)
            x1 = min(width, width - dx)

            if y1 <= y0 or x1 <= x0:
                continue

            accumulator[y0:y1, x0:x1] += correlation[
                y0 + dy : y1 + dy,
                x0 + dx : x1 + dx,
            ]
            counts[y0:y1, x0:x1] += 1

    score = np.full_like(correlation, -1.0, dtype=np.float32)
    valid = counts >= min_tiles
    score[valid] = accumulator[valid] / counts[valid]
    return score, counts


def _resize_alpha(
    alpha: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    width = max(8, int(round(alpha.shape[1] * scale)))
    height = max(8, int(round(alpha.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        alpha,
        (width, height),
        interpolation=interpolation,
    ).astype(np.float32)
    # Quantized profile noise below one byte should not activate removal.
    resized[resized < (1.5 / 255.0)] = 0.0
    return resized


def _match_profile(
    frame: np.ndarray,
    profile: KnownTiledProfile,
) -> KnownProfileMatch | None:
    steps = _axis_aligned_steps(frame)
    if steps is None:
        return None

    step_x, step_y = steps
    scale = float(
        np.median(
            [
                step_x / profile.reference_step_x,
                step_y / profile.reference_step_y,
            ]
        )
    )

    if not 0.45 <= scale <= 2.5:
        return None

    alpha_full = _resize_alpha(profile.alpha, scale=scale)

    frame_height, frame_width = frame.shape[:2]
    match_scale = min(1.0, 1920.0 / max(1, frame_width))

    match_width = max(32, int(round(frame_width * match_scale)))
    match_height = max(32, int(round(frame_height * match_scale)))
    frame_match = cv2.resize(
        frame,
        (match_width, match_height),
        interpolation=cv2.INTER_AREA if match_scale < 1.0 else cv2.INTER_LINEAR,
    )

    template_width = max(8, int(round(alpha_full.shape[1] * match_scale)))
    template_height = max(8, int(round(alpha_full.shape[0] * match_scale)))
    alpha_match = cv2.resize(
        alpha_full,
        (template_width, template_height),
        interpolation=cv2.INTER_AREA if match_scale < 1.0 else cv2.INTER_LINEAR,
    )

    if template_width >= match_width or template_height >= match_height:
        return None

    frame_gray = (
        cv2.cvtColor(frame_match, cv2.COLOR_BGR2GRAY).astype(np.float32)
        / 255.0
    )
    frame_feature = _gradient_magnitude(frame_gray)
    template_feature = _gradient_magnitude(alpha_match)

    if float(np.max(template_feature)) <= 1e-6:
        return None

    correlation = cv2.matchTemplate(
        frame_feature,
        template_feature,
        cv2.TM_CCOEFF_NORMED,
    )

    step_x_match = max(1, int(round(step_x * match_scale)))
    step_y_match = max(1, int(round(step_y * match_scale)))

    aggregate, counts = _aggregate_grid_score(
        correlation,
        step_x=step_x_match,
        step_y=step_y_match,
        min_tiles=profile.min_tiles,
    )

    _, aggregate_score, _, best_location = cv2.minMaxLoc(aggregate)
    best_x, best_y = best_location

    if aggregate_score < profile.min_aggregate_score:
        return None
    if counts[best_y, best_x] < profile.min_tiles:
        return None

    corr_height, corr_width = correlation.shape
    radius_x = min(
        8,
        int(math.ceil(corr_width / max(step_x_match, 1))) + 1,
    )
    radius_y = min(
        8,
        int(math.ceil(corr_height / max(step_y_match, 1))) + 1,
    )

    match_positions: list[tuple[int, int, float]] = []
    for grid_y in range(-radius_y, radius_y + 1):
        for grid_x in range(-radius_x, radius_x + 1):
            x = best_x + grid_x * step_x_match
            y = best_y + grid_y * step_y_match

            if not (0 <= x < corr_width and 0 <= y < corr_height):
                continue

            tile_score = float(correlation[y, x])
            if tile_score < profile.min_tile_score:
                continue
            match_positions.append((x, y, tile_score))

    if len(match_positions) < profile.min_tiles:
        return None

    tile_scores = [item[2] for item in match_positions]
    median_tile_score = float(np.median(tile_scores))
    strong_tiles = sum(score >= 0.25 for score in tile_scores)

    if median_tile_score < profile.min_median_tile_score:
        return None
    if strong_tiles < profile.min_strong_tiles:
        return None

    positions: list[Rect] = []
    retained_scores: list[float] = []
    for x, y, tile_score in match_positions:
        full_x = int(round(x / match_scale))
        full_y = int(round(y / match_scale))
        rect = Rect(
            x=full_x,
            y=full_y,
            width=alpha_full.shape[1],
            height=alpha_full.shape[0],
        )

        if rect.x2 > frame_width or rect.y2 > frame_height:
            continue

        positions.append(rect)
        retained_scores.append(tile_score)

    if len(positions) < profile.min_tiles:
        return None

    confidence = float(
        np.clip(
            0.70
            + 0.20 * aggregate_score
            + 0.10 * median_tile_score,
            0.0,
            0.99,
        )
    )

    return KnownProfileMatch(
        profile_id=profile.id,
        alpha=alpha_full,
        positions=positions,
        aggregate_score=float(aggregate_score),
        tile_scores=retained_scores,
        confidence=confidence,
        step_x=step_x,
        step_y=step_y,
        scale=scale,
    )


def detect_known_tiled_profile(
    frame: np.ndarray,
) -> KnownProfileMatch | None:
    matches: list[KnownProfileMatch] = []

    for profile in KNOWN_TILED_PROFILES:
        match = _match_profile(frame, profile)
        if match is not None:
            matches.append(match)

    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            item.confidence,
            item.aggregate_score,
            float(np.median(item.tile_scores)),
        ),
        reverse=True,
    )
    return matches[0]


def get_known_profile_alpha(profile_id: str) -> np.ndarray:
    for profile in KNOWN_TILED_PROFILES:
        if profile.id == profile_id:
            return profile.alpha.copy()
    raise KeyError(profile_id)
