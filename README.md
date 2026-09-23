# watermark-remover

Provider-agnostic transparent video watermark remover.

The core engine does **not** special-case HeyGen, Gemini, Veo, CapCut, or any other provider. It analyzes the visual overlay itself and chooses between:

- reverse alpha compositing for recoverable translucent pixels;
- hybrid reverse-alpha + inpainting for mixed overlays;
- local inpainting when the alpha model is not reliable.

## How it works

For a static screen-space overlay, the engine models each pixel as:

```
C = alpha * W + (1 - alpha) * B
```

where `C` is the observed watermarked pixel, `W` is the overlay color, and `B` is the clean background.

The automatic pipeline is:

```
sample frames
  -> persistent overlay detection
  -> candidate mask
  -> local background reconstruction
  -> alpha/template estimation
  -> fit validation
  -> reverse alpha / hybrid / inpaint
  -> FFmpeg encode + original audio remux
```

Known-provider templates can be added later as a cache/optimization, but they are not part of the removal algorithm.

## Current MVP scope

Works best when the watermark:

- stays at a fixed screen position;
- is visible across most of the video;
- is partially transparent;
- sits over content that changes enough across sampled frames.

Automatic estimation is fundamentally ambiguous on a single frame. Very short videos, nearly static backgrounds, moving watermarks, adaptive-color watermarks, and fully opaque overlays can fall back to inpainting.

## Install

Requirements:

- Python 3.11+
- FFmpeg on `PATH`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI

Analyze a video without writing output:

```bash
watermark-remover analyze input.mp4 --report report.json --debug-dir debug
```

Remove an automatically detected fixed overlay:

```bash
watermark-remover remove input.mp4 output.mp4
```

If automatic detection misses a watermark, provide a full-frame grayscale mask. White pixels mean "watermark candidate":

```bash
watermark-remover remove input.mp4 output.mp4 --mask watermark-mask.png
```

Useful options:

```bash
watermark-remover remove input.mp4 output.mp4 \
  --sample-count 18 \
  --min-confidence 0.55 \
  --crf 18 \
  --preset medium
```

## Python API

```python
from watermark_remover.engine import WatermarkRemover

engine = WatermarkRemover(sample_count=18)
analysis = engine.analyze("input.mp4")
print(analysis.report)

engine.remove("input.mp4", "output.mp4", analysis=analysis)
```

## Design notes

The estimator intentionally returns confidence and fit diagnostics instead of assuming every persistent graphic is a translucent watermark. A persistent HUD, REC indicator, subtitle box, or opaque logo may also be detected. Low-confidence or high-alpha pixels are not aggressively deblended; they are handled by local inpainting instead.

See `docs/architecture.md` for implementation details.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
