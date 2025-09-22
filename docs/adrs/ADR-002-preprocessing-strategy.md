# ADR-002: Heuristic Brain Crop & Corner Artifact Mitigation

## Status
Accepted

## Context
Early Grad-CAM visualizations showed attention clustering in image corners and around burned-in text/annotation overlays. Training images contain variable margins and non-brain background. A robust learned segmentation model was not yet integrated.

## Decision
Adopt a lightweight preprocessing sequence:
1. `CornerTextRemover` (heuristic overlay detection & zeroing)
2. Optional `BrainCrop` (grayscale threshold bounding box with padding)
3. Resize -> Augment (RandomCornerMask + Flip) -> Normalize

## Rationale
- Zero external dependency footprint beyond NumPy & PIL.
- Immediate reduction of shortcut features (text, borders) without labeling masks.
- BrainCrop narrows receptive field to relevant anatomy, improving signal-to-noise.
- RandomCornerMask regularizes against residual corner artifacts.

## Consequences
- Possible over-cropping on low-contrast images (risk mitigated by fallback to original when heuristic fails).
- Potential removal of clinically relevant peripheral features if pathology near margins.
- Adds slight preprocessing overhead per sample.

## Alternatives Considered
| Approach | Pros | Cons |
|----------|------|------|
| Learned UNet brain mask | Accurate, adaptable | Requires mask labels, higher latency |
| Classical morphology (Otsu + closing) | Better segmentation shape | More tuning, added complexity |
| No crop / only augmentation | Simpler | Leaves corner shortcut risk |

## Future Evolution
- Plug-in segmentation interface for model-driven masks.
- Adaptive thresholding or intensity normalization before crop.
- Cache cropping boxes per image for faster multi-epoch training.
