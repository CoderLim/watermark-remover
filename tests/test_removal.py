import cv2
import numpy as np

from watermark_remover.detection import detect_repeated_overlay_candidates
from watermark_remover.estimation import consolidate_repeated_overlay_models
from watermark_remover.known_profiles import (
    detect_known_tiled_profile,
    get_known_profile_alpha,
)
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


def test_low_confidence_preserves_source_pixels_by_default():
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
        method="keep-original",
    )

    recovered = remove_overlay_from_frame(
        frame,
        model,
        allow_deblend=False,
    )
    assert np.array_equal(recovered, frame)


def test_unresolved_pixels_are_not_inpainted_by_default():
    frame = np.zeros((28, 28, 3), dtype=np.uint8)
    frame[8:20, 8:20] = (25, 140, 230)
    rect = Rect(8, 8, 12, 12)

    mask = np.ones((12, 12), dtype=np.uint8)
    alpha = np.zeros((12, 12), dtype=np.float32)
    alpha[4:8, 4:8] = 0.25

    rgb = np.ones((12, 12, 3), dtype=np.float32)
    fit_error = np.ones((12, 12), dtype=np.float32)
    fit_error[4:8, 4:8] = 0.0

    model = OverlayModel(
        rect=rect,
        mask=mask,
        alpha=alpha,
        rgb=rgb,
        fit_error=fit_error,
        confidence=0.9,
        method="reverse-alpha-partial",
    )

    recovered = remove_overlay_from_frame(frame, model)

    # The unsupported border of the candidate must remain byte-for-byte identical.
    local_before = frame[8:20, 8:20]
    local_after = recovered[8:20, 8:20]
    unsupported = np.ones((12, 12), dtype=bool)
    unsupported[4:8, 4:8] = False
    assert np.array_equal(local_after[unsupported], local_before[unsupported])


def _make_model(alpha_value, rgb_value, valid=True):
    rect = Rect(0, 0, 8, 8)
    mask = np.ones((8, 8), dtype=np.uint8)
    alpha = np.full((8, 8), alpha_value, dtype=np.float32)
    rgb = np.full((8, 8, 3), rgb_value, dtype=np.float32)
    fit_error = np.full(
        (8, 8),
        0.01 if valid else 0.20,
        dtype=np.float32,
    )
    return OverlayModel(
        rect=rect,
        mask=mask,
        alpha=alpha,
        rgb=rgb,
        fit_error=fit_error,
        confidence=0.8,
        method="reverse-alpha",
        diagnostics={"detector_score": 0.9},
    )


def test_repeated_consensus_rejects_disagreement_and_keeps_supported_profile():
    models = [
        _make_model(0.30, 0.92),
        _make_model(0.31, 0.91),
        _make_model(0.29, 0.93),
        _make_model(0.70, 0.20),
    ]

    # Make the bad copy disagree only in the right half. The three agreeing copies
    # should still establish a stable shared profile there.
    models[3].alpha[:, :4] = 0.30
    models[3].rgb[:, :4] = 0.92

    consolidated = consolidate_repeated_overlay_models(models)

    assert len(consolidated) == 4
    for model in consolidated:
        assert model.diagnostics["shared_profile"] is True
        assert model.method == "reverse-alpha-consensus"
        assert int(model.mask.sum()) > 0
        active_alpha = model.alpha[model.mask.astype(bool)]
        assert abs(float(np.median(active_alpha)) - 0.30) < 0.03


def _synthetic_tiled_overlay() -> np.ndarray:
    height, width = 360, 640
    yy, xx = np.mgrid[0:height, 0:width]

    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[..., 0] = ((xx * 0.30 + yy * 0.20) % 180 + 30).astype(np.uint8)
    background[..., 1] = ((xx * 0.10 + yy * 0.40) % 180 + 30).astype(np.uint8)
    background[..., 2] = ((xx * 0.20 + yy * 0.10) % 180 + 30).astype(np.uint8)

    cv2.circle(background, (80, 80), 50, (40, 120, 200), -1)
    cv2.rectangle(background, (450, 200), (600, 330), (160, 70, 40), -1)

    alpha = 0.45
    for center_y in (60, 180, 300):
        for center_x in (80, 240, 400, 560):
            overlay = background.copy()
            cv2.putText(
                overlay,
                "WM",
                (center_x - 30, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            points = np.array(
                [
                    [center_x + 25, center_y - 22],
                    [center_x + 38, center_y - 10],
                    [center_x + 25, center_y + 2],
                    [center_x + 12, center_y - 10],
                ]
            )
            cv2.fillConvexPoly(overlay, points, (240, 240, 240))

            changed = np.any(overlay != background, axis=2)
            background[changed] = np.round(
                alpha * overlay[changed]
                + (1.0 - alpha) * background[changed]
            ).astype(np.uint8)

    return background


def test_detects_repeated_overlay_as_one_generic_group():
    frame = _synthetic_tiled_overlay()
    candidates = detect_repeated_overlay_candidates([frame] * 4)

    assert len(candidates) >= 8
    assert {candidate.source for candidate in candidates} == {"spatial-repeat"}
    assert {candidate.group_id for candidate in candidates} == {"repeat-0"}



def test_known_tiled_profile_matches_and_restores_calibrated_overlay():
    height, width = 1080, 1920
    yy, xx = np.mgrid[0:height, 0:width]

    # Smooth, non-repeating scene so the only strong 420px lattice is the overlay.
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[..., 0] = np.clip(35 + xx * 0.018 + yy * 0.006, 0, 150)
    background[..., 1] = np.clip(45 + xx * 0.012 + yy * 0.010, 0, 150)
    background[..., 2] = np.clip(55 + xx * 0.008 + yy * 0.014, 0, 150)

    alpha = get_known_profile_alpha("heygen-tiled-v1")
    watermarked = background.astype(np.float32) / 255.0

    expected_positions = []
    for y in (241, 661):
        for x in (247, 667, 1087, 1507):
            expected_positions.append((x, y))
            roi = watermarked[
                y : y + alpha.shape[0],
                x : x + alpha.shape[1],
            ]
            a = alpha[..., None]
            roi[:] = a + (1.0 - a) * roi

    watermarked_u8 = np.round(watermarked * 255.0).astype(np.uint8)
    match = detect_known_tiled_profile(watermarked_u8)

    assert match is not None
    assert match.profile_id == "heygen-tiled-v1"
    assert len(match.positions) == 8
    assert match.aggregate_score >= 0.35

    actual_positions = sorted((rect.x, rect.y) for rect in match.positions)
    expected_positions = sorted(expected_positions)
    for (actual_x, actual_y), (expected_x, expected_y) in zip(
        actual_positions,
        expected_positions,
        strict=True,
    ):
        assert abs(actual_x - expected_x) <= 2
        assert abs(actual_y - expected_y) <= 2

    restored = watermarked_u8.copy()
    for model in match.to_models():
        restored = remove_overlay_from_frame(restored, model)

    active = alpha > 0
    deltas = []
    for y in (241, 661):
        for x in (247, 667, 1087, 1507):
            clean_roi = background[
                y : y + alpha.shape[0],
                x : x + alpha.shape[1],
            ]
            restored_roi = restored[
                y : y + alpha.shape[0],
                x : x + alpha.shape[1],
            ]
            deltas.append(
                np.abs(
                    restored_roi[active].astype(np.int16)
                    - clean_roi[active].astype(np.int16)
                )
            )

    delta = np.concatenate(deltas, axis=0)
    assert float(np.mean(delta)) < 1.5
    assert int(np.percentile(delta, 99)) <= 3
