from __future__ import annotations

import math

import cv2
import numpy as np

from .models import OverlayCandidate, OverlayModel


def _prepare_estimation_mask(mask: np.ndarray) -> np.ndarray:
    # Keep the estimation support close to the detected visual strokes. The previous
    # extra dilation made thin logo masks grow into large destructive regions.
    binary = (mask > 0).astype(np.uint8)
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )


def _estimate_background(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Bootstrap estimate only. It is never written to the output frame.
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
        backgrounds.append(
            _estimate_background(crop, mask).astype(np.float32) / 255.0
        )

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
    fit_error = np.sqrt(
        np.mean(residual * residual, axis=(0, 3))
    ).astype(np.float32)

    enough_background_variance = denominator > 8e-4
    valid_alpha = (alpha_raw >= 0.02) & (alpha_raw <= 0.88)
    valid_rgb = np.all(
        (overlay_rgb >= -0.08) & (overlay_rgb <= 1.08),
        axis=2,
    )
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
    alpha[valid] = np.clip(alpha_raw[valid], 0.0, 0.88).astype(np.float32)

    # Unknown pixels are now represented as alpha=0, not alpha=1. They are not
    # "opaque"; they are simply unsupported and must stay untouched by default.
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
        median_alpha = 0.0

    fit_score = float(
        np.clip(1.0 - median_error / max_fit_error, 0.0, 1.0)
    )
    confidence = float(
        np.clip(
            0.45 * valid_ratio
            + 0.35 * fit_score
            + 0.20 * candidate.detector_score,
            0.0,
            1.0,
        )
    )

    if confidence >= 0.68 and valid_ratio >= 0.70:
        method = "reverse-alpha"
    elif valid_ratio >= 0.18:
        method = "reverse-alpha-partial"
    else:
        method = "keep-original"

    diagnostics = {
        "source": candidate.source,
        "group_id": candidate.group_id,
        "detector_score": candidate.detector_score,
        "persistence": candidate.persistence,
        "repeat_score": candidate.repeat_score,
        "transparent_fit_ratio": valid_ratio,
        "median_transparent_alpha": median_alpha,
        "median_transparent_fit_error": median_error,
        "max_fit_error": max_fit_error,
        "safety_fallback": "keep-original",
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


def _resize_scalar(
    value: np.ndarray,
    width: int,
    height: int,
    *,
    nearest: bool = False,
) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(value, (width, height), interpolation=interpolation)


def _resize_rgb(value: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)


def consolidate_repeated_overlay_models(
    models: list[OverlayModel],
    *,
    max_fit_error: float = 0.045,
    min_support_ratio: float = 0.35,
    max_alpha_mad: float = 0.08,
    max_rgb_mad: float = 0.12,
) -> list[OverlayModel]:
    """
    Build one conservative shared alpha/RGB profile from repeated copies.

    Only normalized pixels supported by multiple independent watermark instances
    survive. Everything else is removed from the active mask and therefore remains
    identical to the source frame.
    """
    if len(models) < 3:
        return models

    canonical_width = max(
        1,
        int(round(np.median([model.rect.width for model in models]))),
    )
    canonical_height = max(
        1,
        int(round(np.median([model.rect.height for model in models]))),
    )

    mask_stack = []
    alpha_stack = []
    rgb_stack = []
    error_stack = []

    for model in models:
        mask_stack.append(
            _resize_scalar(
                model.mask.astype(np.float32),
                canonical_width,
                canonical_height,
                nearest=True,
            )
            > 0.5
        )
        alpha_stack.append(
            _resize_scalar(
                model.alpha,
                canonical_width,
                canonical_height,
            )
        )
        rgb_stack.append(
            _resize_rgb(
                model.rgb,
                canonical_width,
                canonical_height,
            )
        )
        error_stack.append(
            _resize_scalar(
                model.fit_error,
                canonical_width,
                canonical_height,
            )
        )

    masks = np.stack(mask_stack, axis=0)
    alphas = np.stack(alpha_stack, axis=0)
    rgbs = np.stack(rgb_stack, axis=0)
    errors = np.stack(error_stack, axis=0)

    valid = (
        masks
        & (alphas >= 0.02)
        & (alphas <= 0.88)
        & (errors <= max_fit_error)
    )

    required_support = max(
        2,
        int(math.ceil(len(models) * min_support_ratio)),
    )
    support = np.sum(valid, axis=0)

    alpha_masked = np.ma.array(alphas, mask=~valid)
    alpha_median = np.ma.median(alpha_masked, axis=0).filled(0.0).astype(
        np.float32
    )
    alpha_mad = np.ma.median(
        np.ma.abs(alpha_masked - alpha_median[None, ...]),
        axis=0,
    ).filled(1.0).astype(np.float32)

    valid_rgb = np.repeat(valid[..., None], 3, axis=3)
    rgb_masked = np.ma.array(rgbs, mask=~valid_rgb)
    rgb_median = np.ma.median(rgb_masked, axis=0).filled(0.0).astype(
        np.float32
    )
    rgb_mad = np.ma.median(
        np.ma.abs(rgb_masked - rgb_median[None, ...]),
        axis=0,
    ).filled(1.0).astype(np.float32)
    rgb_mad_mean = np.mean(rgb_mad, axis=2)

    error_masked = np.ma.array(errors, mask=~valid)
    error_median = np.ma.median(
        error_masked,
        axis=0,
    ).filled(1.0).astype(np.float32)

    consensus = (
        (support >= required_support)
        & (alpha_mad <= max_alpha_mad)
        & (rgb_mad_mean <= max_rgb_mad)
        & (alpha_median >= 0.02)
        & (alpha_median <= 0.88)
    )

    consensus_pixels = int(consensus.sum())
    canonical_candidate = np.any(masks, axis=0)
    candidate_pixels = int(canonical_candidate.sum())
    consensus_coverage = consensus_pixels / max(1, candidate_pixels)

    if consensus_pixels:
        median_alpha_mad = float(np.median(alpha_mad[consensus]))
        median_rgb_mad = float(np.median(rgb_mad_mean[consensus]))
        mean_support = float(
            np.mean(support[consensus]) / max(1, len(models))
        )
    else:
        median_alpha_mad = 1.0
        median_rgb_mad = 1.0
        mean_support = 0.0

    alpha_agreement = float(
        np.clip(1.0 - median_alpha_mad / max_alpha_mad, 0.0, 1.0)
    )
    rgb_agreement = float(
        np.clip(1.0 - median_rgb_mad / max_rgb_mad, 0.0, 1.0)
    )
    consensus_quality = 0.45 * alpha_agreement + 0.35 * rgb_agreement + 0.20 * mean_support

    output: list[OverlayModel] = []
    for model in models:
        width = model.rect.width
        height = model.rect.height

        local_consensus = (
            _resize_scalar(
                consensus.astype(np.uint8),
                width,
                height,
                nearest=True,
            )
            > 0
        )
        local_original_mask = model.mask.astype(bool)
        final_mask = local_consensus & local_original_mask

        local_alpha = _resize_scalar(
            alpha_median,
            width,
            height,
        ).astype(np.float32)
        local_rgb = _resize_rgb(
            rgb_median,
            width,
            height,
        ).astype(np.float32)
        local_error = _resize_scalar(
            error_median,
            width,
            height,
        ).astype(np.float32)

        alpha = np.zeros((height, width), dtype=np.float32)
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        fit_error = np.ones((height, width), dtype=np.float32)

        alpha[final_mask] = np.clip(
            local_alpha[final_mask],
            0.02,
            0.88,
        )
        rgb[final_mask] = np.clip(
            local_rgb[final_mask],
            0.0,
            1.0,
        )
        fit_error[final_mask] = np.clip(
            local_error[final_mask],
            0.0,
            max_fit_error,
        )

        local_coverage = int(final_mask.sum()) / max(
            1,
            int(local_original_mask.sum()),
        )
        detector_score = float(
            model.diagnostics.get("detector_score", 0.0)
        )
        confidence = float(
            np.clip(
                0.55 * consensus_quality
                + 0.25 * detector_score
                + 0.20 * min(local_coverage * 2.0, 1.0),
                0.0,
                1.0,
            )
        )

        diagnostics = {
            **model.diagnostics,
            "shared_profile": True,
            "consensus_instances": len(models),
            "consensus_required_support": required_support,
            "consensus_coverage": consensus_coverage,
            "local_consensus_coverage": local_coverage,
            "consensus_mean_support": mean_support,
            "consensus_alpha_mad": median_alpha_mad,
            "consensus_rgb_mad": median_rgb_mad,
            "safety_fallback": "keep-original",
        }

        output.append(
            OverlayModel(
                rect=model.rect,
                mask=final_mask.astype(np.uint8),
                alpha=alpha,
                rgb=rgb,
                fit_error=fit_error,
                confidence=confidence,
                method=(
                    "reverse-alpha-consensus"
                    if final_mask.any()
                    else "keep-original"
                ),
                diagnostics=diagnostics,
            )
        )

    return output


def estimate_repeated_overlay_models(
    frames: list[np.ndarray],
    candidates: list[OverlayCandidate],
    *,
    max_fit_error: float = 0.045,
) -> list[OverlayModel]:
    individual = [
        estimate_overlay_model(
            frames,
            candidate,
            max_fit_error=max_fit_error,
        )
        for candidate in candidates
    ]
    return consolidate_repeated_overlay_models(
        individual,
        max_fit_error=max_fit_error,
    )
