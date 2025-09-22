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
flowchart LR
    %% Lanes: Dataset -> Training -> Best Model -> Inference (User)
    %% 1. Dataset Preparation
    subgraph DS[Dataset Preparation]
        RAW[Raw MRI class folders\n(pituitary_tumor / ... / no_tumor)] --> PREP[prepare_dataset.py\nrename + deterministic split]
        PREP --> TR[train/]
        PREP --> VA[val/]
        PREP --> TE[test/]
    end

    %% 2. Shared Preprocessing Definition
    subgraph PP[Shared Preprocessing (same logic)]
        P1[CornerTextRemover]
        P2[BrainCrop?]
        P3[Resize 224x224]
        P4[Augment (train only):\nCornerMask? + Flip]
        P5[ToTensor + Normalize]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    %% 3. Training Path
    subgraph TRN[Training & Validation Loop]
        TR --> LTR[ImageFolder(train)]
        VA --> LVA[ImageFolder(val)]
        LTR --> PTR[Apply Preprocessing\n(train path)]
        LVA --> PVA[Apply Preprocessing\n(val path)]
        PTR --> DLT[Train DataLoader]
        PVA --> DLV[Val DataLoader]
        DLT --> MOD[DenseNet121 model]
        DLV --> MOD
        MOD --> OPT[Adam updates]
        OPT --> MOD
        MOD --> METRICS[Loss + Acc (train/val)]
        METRICS --> CKPT{val_acc improved?}
        CKPT -->|yes| BEST[(best.pt\n+ classes)]
        CKPT -->|every epoch| LAST[(last.pt\n+ classes)]
    end

    %% 4. User Inference Path
    subgraph INF[User Inference (Streamlit)]
        UP[User Uploaded Image\n(often from test/)] --> PINF[Apply Preprocessing\n(inference path)]
        BEST --> LOAD[Load checkpoint\n(classes restored)]
        PINF --> RUN[Forward pass]
        LOAD --> RUN
        RUN --> PROB[Softmax + Top-1]
        RUN --> CAM[Grad-CAM\n(denseblock4)]
        CAM --> OVER[Overlay heatmap]
        PROB --> OUT[Prediction dict]
        OVER --> OUT
        OUT --> UI[Streamlit UI\n(display results)]
    end

    %% 5. Test Split (Optional Evaluation Outside App)
    TE --> LTE[ImageFolder(test)] --> PTE[Apply Preprocessing\n(val/infer mode)] --> EVAL[Test Metrics\n(script future)]

    %% Cross-links / Emphasis
    classDef shared fill=#e8f5e9,stroke=#2e7d32,color=#1b5e20;
    class P1,P2,P3,P4,P5,PTR,PVA,PINF,PTE shared;

    %% Notes:
    %% - Augment stage (CornerMask + Flip) only active for training batches (PTR).
    %% - BEST checkpoint is the one the Streamlit app should use by default.
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
