from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def expand(self, padding: int, frame_width: int, frame_height: int) -> "Rect":
        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(frame_width, self.x2 + padding)
        y2 = min(frame_height, self.y2 + padding)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class OverlayCandidate:
    rect: Rect
    mask: np.ndarray
    detector_score: float
    persistence: float
    source: str = "temporal"
    group_id: str | None = None
    repeat_score: float = 0.0

    def __post_init__(self) -> None:
        if self.mask.shape != (self.rect.height, self.rect.width):
            raise ValueError("candidate mask shape does not match rect")


@dataclass
class OverlayModel:
    rect: Rect
    mask: np.ndarray
    alpha: np.ndarray
    rgb: np.ndarray
    fit_error: np.ndarray
    confidence: float
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (self.rect.height, self.rect.width)
        if self.mask.shape != expected:
            raise ValueError("mask shape does not match rect")
        if self.alpha.shape != expected:
            raise ValueError("alpha shape does not match rect")
        if self.fit_error.shape != expected:
            raise ValueError("fit_error shape does not match rect")
        if self.rgb.shape != (*expected, 3):
            raise ValueError("rgb shape does not match rect")

    def report(self) -> dict[str, Any]:
        active = self.mask.astype(bool)
        alphas = self.alpha[active]
        errors = self.fit_error[active]
        return {
            "rect": self.rect.to_dict(),
            "confidence": float(self.confidence),
            "method": self.method,
            "mask_pixels": int(active.sum()),
            "alpha_median": float(np.median(alphas)) if alphas.size else 0.0,
            "alpha_p90": float(np.percentile(alphas, 90)) if alphas.size else 0.0,
            "fit_error_median": float(np.median(errors)) if errors.size else 1.0,
            "diagnostics": self.diagnostics,
        }
