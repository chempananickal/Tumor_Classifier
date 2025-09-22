# Limitations & Responsible Use

This repository provides an experimental brain MRI tumor slice classifier with Grad-CAM visualization. This is a college project. It is NOT a medical device and MUST NOT be used for autonomous clinical decision making.

## 1. Scope Constraints

- Single 2D slice classification (not whole-volume reasoning).
- No temporal / longitudinal context.
- Assumes input resembles training distribution (orientation, contrast, intensity range).

## 2. Dataset Limitations

- Underlying public dataset may not represent all scanner vendors, field strengths, protocols (T1 vs T2 vs FLAIR not explicitly separated here).
- Ground truth labels do not include tumor segmentation masks or volumetric annotations; only slice-level class labels are provided.
- Patient-level leakage because multiple slices from the same subject can be distributed across splits.
- Pituitary has 825 items, negative 395 items, glioma 826 items, meningioma 822 items. Negative is underrepresented and may lead to biased decision thresholds.

## 3. Preprocessing Risks

- BrainCrop is a simple NumPy heuristic that may under‑ or over‑crop in low contrast or artifact-heavy slices.
- CornerTextRemover and RandomCornerMask remove or alter corner regions; if pathology extends into extreme periphery, attention during training may be reduced.
- Normalization uses ImageNet mean/std rather than MRI‑specific intensity standardization—contrast shifts could degrade performance.

## 4. Model Limitations

- DenseNet121 2D backbone does not leverage 3D anatomical continuity.
- No uncertainty quantification beyond softmax probabilities (which may be miscalibrated).
- Grad-CAM heatmaps highlight correlated regions, not causal evidence of pathology.
- No out-of-distribution (OOD) detection; unfamiliar modalities may yield overconfident predictions.

## 5. Ethical & Safety Considerations

- Misclassification risk: False negatives (tumor predicted as negative) could delay hypothetical downstream review if misused.
- Interpretability caveat: Heatmaps can create false confidence—highlighted area DOES NOT correspond to tumor boundaries. It just indicates regions influencing the model's decision.
- Dataset bias: If demographic, acquisition, or pathology spectrum skew exists, model may generalize poorly elsewhere.

## 6. Privacy & PHI Concerns

- Heuristic corner cleanup is not a guaranteed anonymization step; embedded identifiers elsewhere (center overlays, embedded DICOM burn‑ins) are not detected. Models may still train themselves on these hidden features.
- Users are responsible for ensuring data is properly de‑identified before use.

## 7. Operational Constraints

- CPU-only environment by default; inference performance isn't significantly impacted, but retraining or updating the model may be slow without GPU acceleration.
- Checkpoint resume logic not implemented: training restarts from first epoch.
- No automated evaluation script for test split included yet.

## 8. Known Failure Modes (Illustrative)

| Scenario | Potential Outcome | Mitigation |
|----------|------------------|------------|
| Very dark / low-intensity slice | BrainCrop fails -> full unprocessed frame used | Add adaptive threshold / contrast stretch |
| Text overlay occupies large region | Misclassification | Increase corner mask probability; refine detector |
| Unusual orientation (rotated) | Misclassification | Add rotation normalization / orientation check |
| Tumor at extreme periphery | Reduced emphasis in CAM | Expand crop padding; consider learned segmentation |
| Different MRI sequence | Distribution shift | Sequence-specific fine-tuning / intensity harmonization |

## 9. Out-of-Scope Uses

- Clinical diagnosis, triage, or treatment planning. THIS IS JUST A PROOF-OF-CONCEPT.
- Detection / segmentation of tumor extent.
- Subclassification beyond 4 categories.

---
For questions or clarifications, open an issue with concrete reproduction details.
