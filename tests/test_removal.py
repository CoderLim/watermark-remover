import numpy as np

from watermark_remover.models import OverlayModel, Rect
from watermark_remover.removal import remove_overlay_from_frame


def test_reverse_alpha_recovers_synthetic_background():
    height, width = 40, 60
    original = np.zeros((height, width, 3), dtype=np.uint8)
    original[..., 0] = 40
    original[..., 1] = 100
    original[..., 2] = 180

    rect = Rect(10, 8, 24, 14)
    mask = np.ones((rect.height, rect.width), dtype=np.uint8)
    alpha = np.full((rect.height, rect.width), 0.30, dtype=np.float32)
    rgb = np.ones((rect.height, rect.width, 3), dtype=np.float32)
    fit_error = np.zeros((rect.height, rect.width), dtype=np.float32)

    watermarked = original.copy()
    crop = watermarked[rect.y:rect.y2, rect.x:rect.x2].astype(np.float32) / 255.0
    composited = alpha[..., None] * rgb + (1.0 - alpha[..., None]) * crop
    watermarked[rect.y:rect.y2, rect.x:rect.x2] = np.round(composited * 255.0).astype(np.uint8)

    model = OverlayModel(
        rect=rect,
        mask=mask,
        alpha=alpha,
        rgb=rgb,
        fit_error=fit_error,
        confidence=0.99,
        method="reverse-alpha",
    )

    recovered = remove_overlay_from_frame(watermarked, model)

    delta = np.abs(recovered.astype(np.int16) - original.astype(np.int16))
    assert int(delta.max()) <= 1


def test_low_confidence_can_disable_deblend():
    frame = np.full((24, 24, 3), 128, dtype=np.uint8)
    rect = Rect(8, 8, 8, 8)
    mask = np.ones((8, 8), dtype=np.uint8)

    model = OverlayModel(
        rect=rect,
        mask=mask,
        alpha=np.full((8, 8), 0.4, dtype=np.float32),
        rgb=np.ones((8, 8, 3), dtype=np.float32),
        fit_error=np.zeros((8, 8), dtype=np.float32),
        confidence=0.1,
        method="inpaint",
    )

    recovered = remove_overlay_from_frame(frame, model, allow_deblend=False)
    assert recovered.shape == frame.shape
