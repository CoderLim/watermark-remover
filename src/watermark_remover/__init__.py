"""Provider-agnostic transparent video watermark removal."""

from .engine import AnalysisResult, WatermarkRemover
from .models import OverlayModel, Rect

__all__ = ["AnalysisResult", "OverlayModel", "Rect", "WatermarkRemover"]
__version__ = "0.1.0"
