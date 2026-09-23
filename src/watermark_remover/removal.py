from __future__ import annotations

import cv2
import numpy as np

from .models import OverlayModel


def remove_overlay_from_frame(
    frame: np.ndarray,
    model: OverlayModel,
    *,
    allow_deblend: bool = True,
    max_deblend_alpha: float = 0.88,
    max_fit_error: float = 0.05,
    inpaint_radius: float = 3.0,
) -> np.ndarray:
    rect = model.rect
    output = frame.copy()
    roi = output[rect.y : rect.y2, rect.x : rect.x2].copy()
    roi_f = roi.astype(np.float32) / 255.0

    active = model.mask.astype(bool)
    deblend = (
        active
        & allow_deblend
        & (model.alpha >= 0.02)
        & (model.alpha <= max_deblend_alpha)
        & (model.fit_error <= max_fit_error)
    )

    if np.any(deblend):
        alpha = model.alpha[..., None]
        denominator = np.maximum(1.0 - alpha, 0.08)
        clean = (roi_f - alpha * model.rgb) / denominator
        clean = np.clip(clean, 0.0, 1.0)

        deblend3 = deblend[..., None]
        roi_f = np.where(deblend3, clean, roi_f)
        roi = np.round(roi_f * 255.0).astype(np.uint8)

    fallback = active & ~deblend
    if np.any(fallback):
        mask8 = fallback.astype(np.uint8) * 255
        roi = cv2.inpaint(roi, mask8, inpaint_radius, cv2.INPAINT_TELEA)

    output[rect.y : rect.y2, rect.x : rect.x2] = roi
    return output
