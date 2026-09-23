from __future__ import annotations

import cv2
import numpy as np

from .models import OverlayCandidate, OverlayModel


def _prepare_estimation_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    return binary


def _estimate_background(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # This is deliberately local and conservative. It is a bootstrap estimate used to fit
    # the alpha model, not the final fallback removal.
    mask8 = (mask > 0).astype(np.uint8) * 255
    return cv2.inpaint(crop, mask8, 3.0, cv2.INPAINT_TELEA)


def estimate_overlay_model(
    frames: list[np.ndarray],
    candidate: OverlayCandidate,
    *,
    max_fit_error: float = 0.045,
) -> OverlayModel:
    rect = candidate.rect
    mask = _prepare_estimation_mask(candidate.mask)

    observed: list[np.ndarray] = []
    backgrounds: list[np.ndarray] = []

    for frame in frames:
        crop = frame[rect.y : rect.y2, rect.x : rect.x2]
        observed.append(crop.astype(np.float32) / 255.0)
        backgrounds.append(_estimate_background(crop, mask).astype(np.float32) / 255.0)

    c = np.stack(observed, axis=0)
    b = np.stack(backgrounds, axis=0)

    mean_b = np.mean(b, axis=0)
    mean_c = np.mean(c, axis=0)
    centered_b = b - mean_b[None, ...]
    centered_c = c - mean_c[None, ...]

    # One beta per pixel, shared by RGB channels:
    # C_tc = beta * B_tc + gamma_c, alpha = 1 - beta.
    numerator = np.sum(centered_b * centered_c, axis=(0, 3))
    denominator = np.sum(centered_b * centered_b, axis=(0, 3))
    beta = numerator / np.maximum(denominator, 1e-7)
    alpha_raw = 1.0 - beta

    gamma = mean_c - beta[..., None] * mean_b
    alpha_safe = np.where(np.abs(alpha_raw) > 1e-4, alpha_raw, 1.0)
    overlay_rgb = gamma / alpha_safe[..., None]

    predicted = beta[None, ..., None] * b + gamma[None, ...]
    residual = c - predicted
    fit_error = np.sqrt(np.mean(residual * residual, axis=(0, 3))).astype(np.float32)

    enough_background_variance = denominator > 8e-4
    valid_alpha = (alpha_raw >= 0.02) & (alpha_raw <= 0.90)
    valid_rgb = np.all((overlay_rgb >= -0.08) & (overlay_rgb <= 1.08), axis=2)
    good_fit = fit_error <= max_fit_error

    candidate_pixels = mask.astype(bool)
    valid = (
        candidate_pixels
        & enough_background_variance
        & valid_alpha
        & valid_rgb
        & good_fit
    )

    alpha = np.zeros_like(alpha_raw, dtype=np.float32)
    rgb = np.clip(overlay_rgb, 0.0, 1.0).astype(np.float32)

    alpha[valid] = np.clip(alpha_raw[valid], 0.0, 0.90).astype(np.float32)

    # Unresolved candidate pixels are intentionally marked as effectively opaque so
    # the removal stage routes them to inpainting instead of unstable deblending.
    unresolved = candidate_pixels & ~valid
    alpha[unresolved] = 1.0
    fit_error = fit_error.astype(np.float32)
    fit_error[~candidate_pixels] = 1.0

    pixel_count = int(candidate_pixels.sum())
    valid_count = int(valid.sum())
    valid_ratio = valid_count / max(1, pixel_count)

    if valid_count:
        median_error = float(np.median(fit_error[valid]))
        median_alpha = float(np.median(alpha[valid]))
    else:
        median_error = 1.0
        median_alpha = 1.0

    fit_score = float(np.clip(1.0 - median_error / max_fit_error, 0.0, 1.0))
    confidence = float(
        np.clip(
            0.45 * valid_ratio
            + 0.35 * fit_score
            + 0.20 * candidate.detector_score,
            0.0,
            1.0,
        )
    )

    transparent_ratio = valid_ratio
    if confidence >= 0.68 and transparent_ratio >= 0.70:
        method = "reverse-alpha"
    elif transparent_ratio >= 0.18:
        method = "hybrid"
    else:
        method = "inpaint"

    diagnostics = {
        "detector_score": candidate.detector_score,
        "persistence": candidate.persistence,
        "transparent_fit_ratio": transparent_ratio,
        "median_transparent_alpha": median_alpha,
        "median_transparent_fit_error": median_error,
        "max_fit_error": max_fit_error,
    }

    return OverlayModel(
        rect=rect,
        mask=mask.astype(np.uint8),
        alpha=alpha,
        rgb=rgb,
        fit_error=fit_error,
        confidence=confidence,
        method=method,
        diagnostics=diagnostics,
    )
