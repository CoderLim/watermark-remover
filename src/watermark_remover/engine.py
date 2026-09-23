from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detection import (
    candidate_from_full_frame_mask,
    detect_repeated_overlay_candidates,
    detect_static_overlay_candidates,
)
from .estimation import estimate_overlay_model
from .models import OverlayModel
from .removal import remove_overlay_from_frame
from .video import encode_processed_video, sample_video_frames


@dataclass
class AnalysisResult:
    models: list[OverlayModel]
    report: dict[str, Any]

    @property
    def model(self) -> OverlayModel:
        """Backward-compatible access to the first selected model."""
        if not self.models:
            raise RuntimeError("analysis contains no overlay model")
        return self.models[0]


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
            candidates = [
                candidate_from_full_frame_mask(
                    mask,
                    (frame_height, frame_width),
                )
            ]
            selection_mode = "manual"
        else:
            repeated = detect_repeated_overlay_candidates(
                frames,
                max_candidates=max(self.max_candidates, 16),
            )
            if len(repeated) >= 3:
                # A repeated lattice is much more selective than generic persistence:
                # keep the complete group so a tiled watermark is removed everywhere,
                # while unique HUD elements such as REC/timers are left alone.
                candidates = repeated
                selection_mode = "spatial-repeat"
            else:
                candidates = detect_static_overlay_candidates(
                    frames,
                    max_candidates=self.max_candidates,
                )
                selection_mode = "temporal-best"

        if not candidates:
            raise RuntimeError(
                "no overlay candidate found; provide --mask for a manual region"
            )

        candidate_models = [
            estimate_overlay_model(frames, candidate)
            for candidate in candidates
        ]

        if selection_mode == "spatial-repeat":
            # Every component belongs to the same repeated group.
            models = candidate_models
        else:
            candidate_models.sort(
                key=lambda item: item.confidence,
                reverse=True,
            )
            models = [candidate_models[0]]

        report: dict[str, Any] = {
            "input": str(input_path),
            "frame": {
                "width": frame_width,
                "height": frame_height,
            },
            "sample_count": len(frames),
            "selection_mode": selection_mode,
            "candidate_count": len(candidates),
            "selected_count": len(models),
            "selected": [model.report() for model in models],
            "candidates": [model.report() for model in candidate_models],
        }

        if debug_dir is not None:
            self._write_debug_files(models, Path(debug_dir))
            report["debug_dir"] = str(debug_dir)

        return AnalysisResult(models=models, report=report)

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

        deblend_flags = [
            model.confidence >= self.min_confidence
            for model in analysis.models
        ]

        def process(frame: np.ndarray) -> np.ndarray:
            output = frame
            for model, allow_deblend in zip(
                analysis.models,
                deblend_flags,
                strict=True,
            ):
                output = remove_overlay_from_frame(
                    output,
                    model,
                    allow_deblend=allow_deblend,
                )
            return output

        encode_processed_video(
            input_path,
            output_path,
            process,
            crf=crf,
            preset=preset,
        )

        analysis.report["output"] = str(output_path)
        analysis.report["deblend_enabled"] = deblend_flags
        analysis.report["deblend_model_count"] = int(sum(deblend_flags))
        analysis.report["inpaint_only_model_count"] = int(
            len(deblend_flags) - sum(deblend_flags)
        )
        return analysis

    @staticmethod
    def save_report(
        report: dict[str, Any],
        path: str | Path,
    ) -> None:
        Path(path).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_debug_files(
        models: list[OverlayModel],
        directory: Path,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        for index, model in enumerate(models):
            model_dir = directory / f"model-{index:02d}"
            model_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                str(model_dir / "mask.png"),
                model.mask.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                str(model_dir / "alpha.png"),
                np.clip(model.alpha * 255.0, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(model_dir / "overlay-rgb.png"),
                np.clip(model.rgb * 255.0, 0, 255).astype(np.uint8),
            )
            error_preview = np.clip(
                model.fit_error / 0.08 * 255.0,
                0,
                255,
            ).astype(np.uint8)
            cv2.imwrite(
                str(model_dir / "fit-error.png"),
                error_preview,
            )
