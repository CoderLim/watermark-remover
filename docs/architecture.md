# Architecture

## Goal

Remove fixed screen-space overlays without maintaining separate HeyGen/Gemini/Veo removal code.

Provider metadata may eventually be used as a fast profile cache, but every path normalizes to the same `OverlayModel`.

## OverlayModel

`OverlayModel` contains:

- frame-space bounding rectangle;
- candidate mask;
- per-pixel alpha;
- per-pixel overlay RGB;
- per-pixel fit error;
- global confidence;
- chosen strategy metadata.

Removal code knows nothing about the provider.

An analysis may contain one model or a group of models. A tiled watermark is represented as multiple independent `OverlayModel` instances belonging to the same generic repeated-overlay group.

## Detection router

There are two detector paths.

### 1. Spatial repetition detector

This path runs first because it is more selective for tiled watermarks.

It:

1. downsamples a representative frame;
2. extracts a Canny edge map;
3. computes 2D FFT autocorrelation;
4. finds strong off-center peaks;
5. chooses primitive non-collinear lattice vectors rather than their harmonics;
6. shifts the edge map by those lattice vectors;
7. keeps pixels whose edges repeat at several translated positions;
8. groups similarly sized connected components into one repeated-overlay group.

This detects the *visual repetition pattern*, not a provider name or logo template.

A unique REC indicator, timer, or battery icon does not form a spatial lattice and therefore does not enter this group.

The repeated path only activates when at least three similarly sized components survive filtering. If that condition is not met, the engine falls back to temporal detection.

### 2. Temporal persistence detector

For a single fixed overlay, the engine samples frames across the video, computes Sobel gradient maps, and measures how often an edge remains in the same screen-space pixel.

Background content generally moves or changes. Static overlays produce persistent edges.

The detector:

1. computes normalized gradient maps;
2. votes on persistent edges across sampled frames;
3. performs small horizontal closing/dilation so letters in a logo can form one candidate;
4. filters connected components by geometry;
5. ranks candidates primarily by temporal persistence and edge strength.

Corner position is only a weak prior. This is intentional because overlays can appear anywhere.

## Alpha/template estimation

The compositing model is:

```
C = alpha * W + (1 - alpha) * B
```

Define:

```
beta = 1 - alpha
gamma = alpha * W
```

Then:

```
C = beta * B + gamma
```

For the MVP, the engine first creates a conservative local clean-background estimate `B_hat` with Telea inpainting. Across sampled frames it performs a per-pixel linear fit with one `beta` shared by RGB channels and one `gamma` per channel.

From the fitted values:

```
alpha = 1 - beta
W = gamma / alpha
```

Pixels are accepted for deblending only when:

- background variance is sufficient;
- fitted alpha is plausible;
- fitted watermark RGB is plausible;
- reconstruction error is low.

Unresolved pixels are routed to inpainting rather than forced through an unstable inverse.

## Removal router

For every active mask pixel:

- reliable alpha < 0.88: reverse-alpha deblend;
- high alpha, invalid fit, or low-confidence model: local inpaint.

For a repeated group, the same operation is applied sequentially to every selected model. There is still no provider-specific branch.

That makes the same engine work as:

- reverse-alpha remover;
- hybrid remover;
- inpaint fallback.

## Why confidence matters

Unknown overlay recovery is underdetermined from a single frame. A video provides more constraints, but static backgrounds and short clips may still not contain enough information.

The engine therefore does not claim exact recovery when it cannot validate the fit.

## Next quality step: temporal background reconstruction

The largest quality improvement after this MVP is replacing the bootstrap Telea background estimate with multi-frame temporal reconstruction:

1. estimate optical flow between nearby frames;
2. warp neighboring content into the target frame;
3. prefer observations that originated outside the watermark mask;
4. robust-median valid samples;
5. fit the same alpha/template model against the improved background.

This can be implemented first with OpenCV DIS flow and later RAFT. It does not require changing `OverlayModel` or the removal API.

For repeated overlays, aligned copies also provide extra observations of the same motif over different backgrounds. A future estimator can jointly fit one canonical alpha/template across the repeated group.

## Optional profile cache

A future profile registry can store an already learned template:

```json
{
  "id": "profile-hash",
  "anchor": "bottom-right",
  "relativeWidth": 0.08,
  "alphaMap": "...",
  "rgb": "..."
}
```

Matching a profile only skips detection/estimation. It must still produce the same `OverlayModel`; there should be no provider-specific remover branch.
