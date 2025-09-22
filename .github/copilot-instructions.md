## HVU_Tumor_Visualizer – Copilot Working Guide

Purpose: End-to-end brain MRI tumor classification (pituitary, glioma, meningioma, negative) with training (DenseNet121 + CrossEntropy) and a Streamlit inference UI providing Grad-CAM heatmaps.

Current Repo State (facts):
- Implemented: model (`models/unet_densenet.py`), Grad-CAM (`app/grad_cam.py`), preprocessing (`app/preprocessing.py`), training script (`scripts/train.py`), inference engine (`app/inference.py`), Streamlit UI (`app/main.py`), smoke test (`tests/test_inference_smoke.py`).
- Environment: `environment.yml` (CPU PyTorch 2.8.0, torchvision 0.23.0). Added matplotlib for optional visualization.
- Weights expected under `models/weights/` (ignored via `.gitignore`). Best checkpoint named `best.pt`.

Model / Classes:
- Architecture: DenseNet121 (ImageNet weights by default) with classifier head replaced: in_features -> 4 logits.
- Class tuple constant (default fallback): `CLASSES = ("pituitary", "glioma", "meningioma", "negative")`.
- Actual training class ordering is read from the dataset (`ImageFolder.classes`) and saved in checkpoints as `classes`; inference dynamically uses that if present to avoid mismatches.
- Loss: `nn.CrossEntropyLoss()`; Optimizer: `Adam(lr=1e-4)` (see `scripts/train.py`).

Training Workflow:
```
python scripts/train.py --data-root /path/to/data --batch-size 16 --epochs 15 --lr 1e-4 --out-dir models/weights
```
Run from repository root so that the implicit path insertion in `scripts/train.py` works. Alternatively:
```
PYTHONPATH=. python scripts/train.py --data-root /path/to/data
```
Data layout required (ImageFolder):
```
data_root/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
```
Outputs: `last.pt` (every epoch) + `best.pt` (highest val_acc). Stored keys: model_state, optimizer_state, val_acc, val_loss, epoch.

Inference (programmatic):
```
from app.inference import InferenceEngine
engine = InferenceEngine(Path('models/weights/best.pt'))
result = engine.predict(PIL.Image.open('example.jpg'))
```
Streamlit UI:
```
streamlit run app/main.py
```
Provide checkpoint path in UI text field if different from default.

Streamlit Import Notes:
- `app/` and `models/` contain `__init__.py`; absolute imports use `from app.inference import InferenceEngine`.
- `app/main.py` & `app/inference.py` insert project root into `sys.path` defensively for environments launching Streamlit from subdirs.

Grad-CAM Details:
- Hooked layer: DenseNet last denseblock (`features.denseblock4`).
- Implementation: forward/backward hooks capture activation & gradient; weights = GAP(grad); CAM = ReLU(weighted sum). Normalized 0–1, resized, blended via OpenCV JET colormap (`overlay_heatmap`).

Preprocessing:
- Uses ImageNet mean/std; resize to 224x224; minimal augmentation: horizontal flip in training. Adjust in `app/preprocessing.py` if dataset differences arise.

Edge Handling:
- Missing weights: UI error; CLI training enforces directory presence.
- Class mismatch: warning logged in training if dataset classes differ from `CLASSES`.
- Grad-CAM failure raises RuntimeError (can be wrapped later for graceful degradation).

Testing:
- `tests/test_inference_smoke.py`: forward shape sanity (2x3x224x224 -> (2,4)). Add more tests for transforms & Grad-CAM normalization when stable.

Extending / Next Steps (update when done):
- Add evaluation script for test split using `best.pt` with metrics (accuracy, per-class recall).
- Optional: persist Grad-CAM images for batch evaluation.
- Consider mixed precision (not enabled; CPU only now).
 - Automate dataset download/sanity checks.

Dataset Preparation:
- Raw folders expected (example): `pituitary_tumor/`, `glioma_tumor/`, `meningioma_tumor/`, `no_tumor/`.
- Convert to training layout with reproducible split (seeded) using:
```
python scripts/prepare_dataset.py --source data --dest data_prepared --seed 42 \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
- Mapping applied: pituitary_tumor->pituitary, glioma_tumor->glioma, meningioma_tumor->meningioma, no_tumor->negative.
- Result: `data_prepared/train/<class>/`, `.../val/<class>/`, `.../test/<class>/`.

Agent Guidelines:
1. Preserve `CLASSES` ordering across training/inference/UI.
2. When changing model internals, update target Grad-CAM layer reference in `ModelWithHooks`.
3. Always load `best.pt` for user-facing inference unless user overrides.
4. Add dependencies only via `environment.yml` and re-export after changes.

Open Clarifications (pending maintainer input):
- Confirm final input resolution (currently 224x224) & whether center crop is desired.
- Any additional augmentations (rotation, normalization from dataset stats) planned?
- Dataset test split path to script for evaluation.

Revision Log:
- v0.3 Added Grad-CAM implementation details and updated structure.
- v0.2 Added preliminary app scaffold.
- v0.1 Initial app placeholder with model training logic already present.

End of instructions.
