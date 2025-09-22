# Architecture Overview

## 1. High-Level Components

- Dataset Preparation (`scripts/prepare_dataset.py`): Normalizes raw class folder names and creates deterministic train/val/test splits.
- Training Pipeline (`scripts/train.py`): Loads ImageFolder datasets, applies preprocessing/augmentation, trains DenseNet121 classifier, saves checkpoints.
- Model Definition (`models/unet_densenet.py`): Provides `get_model` and wrapper `ModelWithHooks` to expose an internal layer for Grad-CAM.
- Preprocessing (`app/preprocessing.py`): Composable transforms (corner text removal, brain crop, resize, augmentation, normalization).
- Inference Engine (`app/inference.py`): Loads checkpoint, applies transforms, runs forward pass, optionally generates Grad-CAM heatmaps.
- Grad-CAM (`app/grad_cam.py`): Hook-based activation + gradient capture and heatmap generation.
- Streamlit UI (`app/main.py`): User-facing interface for single-image upload and visualization.
- Tests (`tests/test_inference_smoke.py`): Minimal forward-shape sanity check.

## 2. Data & Control Flow

Note: AI-generated diagram below.
```mermaid
flowchart TD
    subgraph DataPrep[Dataset Preparation]
        A[Raw class folders<br/>pituitary_tumor/... etc] --> B[prepare_dataset.py<br/>rename + split]
        B --> C[data_prepared/train]
        B --> D[data_prepared/val]
        B --> E[data_prepared/test]
    end

    subgraph Training[Training Pipeline]
        C --> TDS[ImageFolder(train)]
        D --> VDS[ImageFolder(val)]
        TDS --> TTF[Train Transforms<br/>CornerTextRemover -> BrainCrop? -> Resize -> CornerMask? -> Flip -> ToTensor+Norm]
        VDS --> VTF[Val Transforms<br/>CornerTextRemover -> BrainCrop? -> Resize -> ToTensor+Norm]
        TTF --> TL[Train DataLoader]
        VTF --> VL[Val DataLoader]
        TL --> TLK[DenseNet121 Model]
        TLK --> OPT[(Adam Optimizer)]
        OPT -->|update| TLK
        TLK --> CRIT[CrossEntropy Loss]
        TLK --> LOG[Softmax (for metrics)]
        VL --> EV[evaluate()]
        EV --> METRICS[val_loss / val_acc]
        METRICS --> CKPT{Better val_acc?}
        CKPT -->|yes| BEST[(best.pt + classes)]
        CKPT -->|always| LAST[(last.pt + classes)]
    end

    subgraph Inference[Inference & Visualization]
        UIMG[User Image] --> ITF[Inference Transforms<br/>CornerTextRemover -> BrainCrop? -> Resize -> ToTensor+Norm]
        BEST --> IELOAD[InferenceEngine<br/>_load_weights()]
        LAST --> IELOAD
        IELOAD --> HOOK{enable_cam?}
        HOOK -->|yes| MW[ModelWithHooks]
        HOOK -->|no| MPlain[Model]
        ITF --> RUN[(Forward Pass)]
        MW --> RUN
        MPlain --> RUN
        RUN --> PRED[Softmax + Argmax]
        MW -->|backward grad| CAMGEN[Grad-CAM<br/>denseblock4 hooks]
        CAMGEN --> HEAT[Heatmap Overlay]
        PRED --> OUT[Result Dict<br/>class, confidence, probs]
        HEAT --> OUT
    end

    OUT --> UI[Streamlit UI]

    classDef minor fill=#f5f5f5,stroke=#bbb,color=#333,font-size=11px;
    classDef ckpt fill=#e3f2fd,stroke=#2196f3;
    class BEST,LAST ckpt;
    classDef infer fill=#fff3e0,stroke=#ff9800;
    class ITF,IELOAD,RUN,PRED,CAMGEN,HEAT,OUT infer;
```

## 3. Preprocessing Pipeline (Ordered)
1. CornerTextRemover – heuristic cleanup of burned‑in labels or scanner overlays.
2. BrainCrop (optional) – threshold-based bounding box around high-intensity brain tissue.
3. Resize (224×224) – fixed spatial resolution for DenseNet.
4. (Train only) RandomCornerMask – stochastically darkens 1–2 corners to reduce spurious reliance.
5. (Train only) RandomHorizontalFlip – mild augmentation.
6. ToTensor + Normalize (ImageNet mean/std) – standardization for pretrained CNN priors.

## 4. Checkpoint Format
- `model_state`: `state_dict()` of classifier.
- `optimizer_state`: Adam optimizer state (not yet used for resume logic).
- `classes`: Class ordering captured from training dataset.
- `epoch`, `val_acc`, `val_loss`.

## 5. Grad-CAM Integration
- Wrapper `ModelWithHooks` registers forward & backward hooks on `features.denseblock4` of DenseNet121.
- Inference triggers backward pass per target class to accumulate gradients, then weights activations by global-average gradient.
- Resulting heatmap normalized (0–1), resized, and color-mapped before overlay.

## 6. Extension Points
| Area | How to Extend | Notes |
|------|---------------|-------|
| Backbone Model | Add flag in `get_model` to select architecture (e.g., ConvNeXt) | Update Grad-CAM target layer accordingly |
| Preprocessing | Add transforms in `preprocessing.py` | Maintain order invariants (cleanup -> crop -> geometric -> tensor) |
| Multi-Slice Input | Stack adjacent slices before `ToTensor` | Adjust model first conv layer (in_channels) |
| Patient-Level Aggregation | Implement MIL/attention pooling over slice embeddings | Requires dataset restructure |
| Segmentation + Classification | Replace `BrainCrop` with learned segmenter or joint UNet | Add new script + masks |
| Deployment | Add ONNX / TorchScript export utility | Ensure deterministic preprocessing |

## 7. Failure & Edge Handling
- Missing checkpoint: raises `FileNotFoundError` in `InferenceEngine`.
- Class mismatch: classes loaded from checkpoint override defaults.
- Grad-CAM failure (hook issues): raises `RuntimeError` (could be wrapped for UI resilience).
- Empty crop / low contrast: BrainCrop returns original image.

## 8. Rationale (Selected Design Choices)
- DenseNet121: Strong baseline CNN with good feature reuse and CAM compatibility.
- Heuristic BrainCrop First: Avoid added latency/complexity of a learned segmenter while reducing background noise.
- CornerTextRemover + RandomCornerMask: Mitigate dataset‑specific artifacts that can create shortcut learning.
- Checkpoint Class Persistence: Prevents semantic logit misalignment across varying dataset orderings.

## 9. Future Diagram Enhancements
Potential to add: multi-task branch, patient-level pooling stage, or segmentation mask pathway once implemented.

## 10. Security / Privacy Considerations
- Assumes de-identified images; no PHI scanning implemented.
- Corner text removal is heuristic—should not be relied on for PHI redaction.

---
This document was generated with AI (GitHub Copilot).
