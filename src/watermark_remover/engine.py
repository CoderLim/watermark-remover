from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detection import candidate_from_full_frame_mask, detect_static_overlay_candidates
from .estimation import estimate_overlay_model
from .models import OverlayModel
from .removal import remove_overlay_from_frame
from .video import encode_processed_video, sample_video_frames


@dataclass
class AnalysisResult:
    model: OverlayModel
    report: dict[str, Any]


class WatermarkRemover:
    def __init__(
        self,
        *,
        sample_count: int = 18,
        min_confidence: float = 0.55,
        max_candidates: int = 12,
    ) -> None:
        self.sample_count = sample_count
        self.min_confidence = min_confidence
        self.max_candidates = max_candidates

    def analyze(
        self,
        input_path: str | Path,
        *,
        mask_path: str | Path | None = None,
        debug_dir: str | Path | None = None,
    ) -> AnalysisResult:
        frames = sample_video_frames(input_path, self.sample_count)
        frame_height, frame_width = frames[0].shape[:2]

        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"cannot read mask: {mask_path}")
            candidates = [candidate_from_full_frame_mask(mask, (frame_height, frame_width))]
        else:
            candidates = detect_static_overlay_candidates(
                frames,
                max_candidates=self.max_candidates,
            )

        if not candidates:
            raise RuntimeError(
                "no persistent overlay candidate found; provide --mask for a manual region"
            )

        models = [estimate_overlay_model(frames, candidate) for candidate in candidates]
        models.sort(key=lambda item: item.confidence, reverse=True)
        model = models[0]

        report: dict[str, Any] = {
            "input": str(input_path),
            "frame": {"width": frame_width, "height": frame_height},
            "sample_count": len(frames),
            "candidate_count": len(candidates),
            "selected": model.report(),
            "candidates": [candidate_model.report() for candidate_model in models],
        }

        if debug_dir is not None:
            self._write_debug_files(model, Path(debug_dir))
            report["debug_dir"] = str(debug_dir)

        return AnalysisResult(model=model, report=report)

    def remove(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        analysis: AnalysisResult | None = None,
        mask_path: str | Path | None = None,
        debug_dir: str | Path | None = None,
        crf: int = 18,
        preset: str = "medium",
    ) -> AnalysisResult:
        if analysis is None:
            analysis = self.analyze(
                input_path,
                mask_path=mask_path,
                debug_dir=debug_dir,
            )

        allow_deblend = analysis.model.confidence >= self.min_confidence

        def process(frame: np.ndarray) -> np.ndarray:
            return remove_overlay_from_frame(
                frame,
                analysis.model,
                allow_deblend=allow_deblend,
            )

        encode_processed_video(
            input_path,
            output_path,
            process,
            crf=crf,
            preset=preset,
        )

        analysis.report["output"] = str(output_path)
        analysis.report["deblend_enabled"] = allow_deblend
        if not allow_deblend:
            analysis.report["fallback_reason"] = (
                f"model confidence {analysis.model.confidence:.3f} "
                f"is below threshold {self.min_confidence:.3f}"
            )
        return analysis

    @staticmethod
    def save_report(report: dict[str, Any], path: str | Path) -> None:
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    @staticmethod
    def _write_debug_files(model: OverlayModel, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(directory / "mask.png"), model.mask.astype(np.uint8) * 255)
        cv2.imwrite(
            str(directory / "alpha.png"),
            np.clip(model.alpha * 255.0, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(directory / "overlay-rgb.png"),
            np.clip(model.rgb * 255.0, 0, 255).astype(np.uint8),
        )
        error_preview = np.clip(model.fit_error / 0.08 * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(directory / "fit-error.png"), error_preview)
