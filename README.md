# Streamlit Tumor Classifier and Visualizer

End-to-end brain MRI tumor classifier (pituitary / glioma / meningioma / negative) with a training pipeline (PyTorch + DenseNet121) and an interactive Streamlit app producing Grad-CAM heatmaps for transparency.

Authored by Rubin James (BIN23 Group 3) as my submission for the "Projektpraktikum" course at Provadis School of International Management and Technology, BIN23.

---

## 1. Features

- 4‑class tumor classification (pituitary, glioma, meningioma, negative)
- DenseNet121 backbone with custom classifier head
- Grad-CAM visualization (overlay heatmaps)
- Heuristic brain-focused preprocessing (brain crop + corner text cleanup)
- Optional augmentation to reduce corner artifact reliance (random corner masking)
- Reproducible dataset splitting script (train / val / test)
- Checkpointing (`last.pt`, `best.pt`) with saved class ordering. Only `best.pt` used for inference by the app.
- Streamlit UI for drag‑and‑drop inference

---

## 2. Quick Start (TL;DR)

```bash
git clone https://github.com/chempananickal/Tumor_Classifier.git
cd Tumor_Classifier
# Assuming you have conda installed:
conda env create -f environment.yml
conda activate tumor

# Download and extract the dataset from https://figshare.com/ndownloader/files/49403884 (curl didn't work. Use browser)
# mkdir data_raw
# (Extract the RAR archive contents into data_raw/)

python scripts/prepare_dataset.py --source data_raw --dest data_prepared --seed 42 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

# If you want to train the model yourself, uncomment and run the following line. Otherwise, skip to the next step to use the provided checkpoint (models/weights/best.pt):
# python scripts/train.py --data-root data_prepared --epochs 15 --batch-size 16 --brain-crop --corner-mask-prob 0.25 --out-dir models/weights

streamlit run app/main.py
```
This should will give you a link in the terminal (likely `http://localhost:8501`) to open in your browser. You can use the app there. Upload an MRI slice and see the prediction + Grad-CAM heatmap. Ideally, use an image from data_prepared/test because the model has never seen those.

---

## 3. Environment Setup

### 3.1 Prerequisites

- 4+ CPU cores, ~16 GB RAM (CPU training and inference; I don't have a CUDA or ROCm GPU setup)
- x86-64 architecture (AMD64/Intel64). ARM (Apple Silicon) might work but I haven't tested it.
- OS: Linux or WSL2 [(Windows Subsystem for Linux)](https://docs.microsoft.com/en-us/windows/wsl/install). MacOS might work but I haven't tested it. 
Tested and validated on [OpenSUSE Tumbleweed (WSL2)](https://apps.microsoft.com/detail/9mssk2zxxn11).
- [Git](https://git-scm.com/)
- Conda ([Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows-subsystem-for-linux-wsl) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2) recommended)
- Python 3.12+ (environment file targets recent PyTorch CPU build). Should be handled by conda.

### 3.2 Create Environment

```bash
conda env create -f environment.yml
conda activate tumor
```
P.S: If you add packages, update the `environment.yml` with this. Trust me, it'll save you headaches later.

```bash
conda env export | grep -v "^prefix: " > environment.yml
```

### 3.3 Verify Installation

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

---

## 4. Dataset Preparation

Source dataset: 2024 Figshare Brain Tumor MRI (link: https://figshare.com/ndownloader/files/49403884). 

Citation:

```
@article{Afzal2024,
	author = "Shiraz Afzal",
	title = "{Brain tumor dataset}",
	year = "2024",
	month = "9",
	url = "https://figshare.com/articles/figure/Brain_tumor_dataset/27102082",
	doi = "10.6084/m9.figshare.27102082.v1"
}
```

### 4.1 Raw Layout Expectation

The aforementioned dataset is a RAR archive. After extracting it to a folder called `data_raw`, it should look something like this:
```
data_raw/
	pituitary_tumor/*.jpg
	glioma_tumor/*.jpg
	meningioma_tumor/*.jpg
	no_tumor/*.jpg
```

### 4.2 Prepare Train/Val/Test Split

Use the provided script (it normalizes class names and splits deterministically by seed):
```bash
python scripts/prepare_dataset.py --source data_raw --dest data_prepared --seed 42 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
Resulting structure (`torchvision.datasets.ImageFolder` compliant):
```
data_prepared/
	train/<class_name>/*
	val/<class_name>/*
	test/<class_name>/*  (not used in training; for final evaluation)
```

Class name mapping applied:
```
pituitary_tumor -> pituitary
glioma_tumor    -> glioma
meningioma_tumor-> meningioma
no_tumor        -> negative
```

---

## 5. Training

### 5.1 Basic Command
```bash
python scripts/train.py --data-root data_prepared --batch-size 16 --epochs 15 --lr 1e-4 --out-dir models/weights
```

### 5.2 Recommended Flags

| Flag | Purpose |
|------|---------|
| `--brain-crop` | Enable heuristic brain-focused crop (improves focus on anatomy) |
| `--corner-mask-prob 0.25` | Randomly zero out corners to avoid text / border bias |
| `--log-interval 20` | Print training batch metrics periodically |

Example with preprocessing enhancements:
```bash
python scripts/train.py --data-root data_prepared --epochs 20 --batch-size 32 --brain-crop --corner-mask-prob 0.25 --lr 1e-4 --log-interval 25 --out-dir models/weights
```

### 5.3 Outputs
`models/weights/last.pt` is written each epoch; `best.pt` when validation accuracy improves.

Checkpoint keys include: `model_state`, `optimizer_state`, `epoch`, `val_acc`, `val_loss`, `classes`.

---

## 6. Programmatic Inference (not required for Streamlit app)

```python
from pathlib import Path
from PIL import Image
from app.inference import InferenceEngine

engine = InferenceEngine(Path('models/weights/best.pt'), brain_crop=True)
img = Image.open('some_scan.jpg')
result = engine.predict(img)
print(result['class'], result['confidence'])
```

Result dictionary fields:
```
class: predicted class label
confidence: softmax probability of predicted class
probs: mapping of all classes -> probability
heatmap: (numpy array) Grad-CAM overlay (only if enable_cam=True)
```

---

## 7. Streamlit App

Launch:
```bash
streamlit run app/main.py
```

Once you run this command in your terminal, it should give you a link (likely `http://localhost:8501`) to open in your browser. The app should be accessible there.

UI Features:
- Upload an MRI slice (RGB or grayscale; auto-converted if not JPEG)
- Displays prediction + confidence + Grad-CAM heatmap

Troubleshooting:
- Missing checkpoint: verify `best.pt` exists in `models/weights`. If not, run the model training script first (see section 5).

---

## 8. Preprocessing & Augmentations

This is the "secret sauce" of the entire model.

Pipeline (in order):
1. Corner text removal (`CornerTextRemover`) – heuristically wipes scanner overlays/labels.
2. Optional brain crop (`BrainCrop`) – threshold-based bounding box.
3. Resize to 224×224.
4. (Train only) Random corner masking (stochastic removal of 1–2 corners).
5. Random horizontal flip (train only).
6. ToTensor + Normalize (ImageNet mean/std).

Rationale:
- Reduces spurious attention on burned-in text or black borders.
- Encourages learning of intra-cranial signal patterns.

Future extensibility: plug in model-driven segmentation, multi-slice stacks, or intensity standardization.

---

## 9. Grad-CAM

Implemented for the final dense block of DenseNet121 (`features.denseblock4`).
Steps: capture activations + gradients -> weight gradients (GAP) -> ReLU(weighted sum) -> normalize -> overlay colormap.

If you change architectures, update the hooked layer in `app/grad_cam.py` or wrapper logic.

---

## 10. Testing

Current test: `tests/test_inference_smoke.py` ensures forward pass shape (2×3×224×224 -> logits (2×4)).

Suggested additions (Copilot's suggestions, not yet implemented):
- Transform integrity tests (brain crop doesn’t change aspect ratio catastrophically).
- Grad-CAM value range (0–1 after normalization).
- Class ordering persistence (checkpoint vs default).

Run tests (if pytest installed):
```bash
pytest -q
```

---

## 11. Brainstorming / Roadmap (Potential Enhancements)

- More architecture choices for the user (not just DenseNet121, probably ResNet etc...)
- 2.5D or 3D context (since multiple slices are from the same scan)
- Mixed precision + GPU support (CPU only training is torturously slow)
- Evaluation script for `test/` split (confusion matrix, ROC)
- Evaluate on much larger datasets (random MRI scans from the internet?)
- Deploy on my server?

---

## 12. Repository Structure (Key Paths)

```
app/
	main.py            # Streamlit UI
	inference.py       # Inference engine + Grad-CAM integration
	preprocessing.py   # Transforms (brain crop, corner cleanup, augmentation)
	grad_cam.py        # Grad-CAM implementation
models/
	unet_densenet.py   # DenseNet121 model wrapper & checkpoint utilities
scripts/
	train.py           # Training loop / CLI
	prepare_dataset.py # Dataset split & renaming utility
tests/
	test_inference_smoke.py
```

---

## 13. Glossary

- PyTorch: A deep learning framework originally from Facebook AI Research.
- DenseNet121: A convolutional neural network architecture known for its dense connectivity pattern. Used as a pretrained backbone for the classifier.
- Grad-CAM: Gradient-weighted Class Activation Mapping, a technique to visualize which parts of an image influenced the model's decision.
- Streamlit: A Python framework for building interactive web apps for machine learning and data science.
- MRI: Magnetic Resonance Imaging, a medical imaging technique used to visualize internal structures of the body, particularly soft tissues like the brain.
- Tumor Types:
  - Pituitary Tumor: A tumor located in the pituitary gland, which can affect hormone production.
  - Glioma: A type of tumor that arises from glial cells in the brain or spine.
  - Meningioma: A tumor that forms on the meninges, the protective membranes covering the brain and spinal cord.
  - Negative: No tumor present.

---

## 14. License

This app is licensed under Apache 2.0. See `LICENSE` in repository root.

As for the individual open source components that make up this app, please refer to their respective licenses.

The dataset is under its own license (CC-BY-4.0, see Figshare link in section 4 for details).

---

## 15. Citation:

Hi, I don't expect anyone to read this far or find this thing useful, but if you do, and you want to build on this repository, please cite the dataset and the open source tools in environment.yml. I couldn't have done this without them. And if you insist on citing this repo too, here's a BibTeX entry:
```
@misc{hvu_tumor_visualizer,
	title  = {HVU Tumor Visualizer: Brain MRI Tumor Classification + Grad-CAM as a Streamlit App},
	author = "Rubin James",
	year   = "2025",
	month  = "9",
	note   = {https://github.com/chempananickal/Tumor_Classifier}
}
```

---

## 16. AI Disclosure

GitHub Copilot (GPT 5) has been used in the creation of this project. All information available to GitHub Copilot during Agent Mode is available under .github/copilot-instructions.md.

In addition to writing everything in tests/, GitHub Copilot was prompted to help with bug fixes and feature improvements. The following prompts were used:

Prompt: "The data is currently in the format data/pituitary_tumor, data/no_tumor, etc... Write a script that turns it into a format that your scripts expect. The shuffling into train, val and test should be random but reproducible (seed)"

LLM Response: --created scripts/prepare_dataset.py following my instructions--

Prompt:"Check through the entire training regimen. It seems to classify everything exactly wrong in the same way. As in, pituitary is predicted as negative, meningioma as glioma, glioma as pituitary, and negative as meningioma. My data is in the correct folders. It's some slight oversight in the training code"

LLM Response: "You were seeing a consistent “rotated” misclassification because the class ordering used at inference didn’t match the dataset’s true class index ordering during training. ImageFolder assigns class indices alphabetically based on folder names. Your earlier CLASSES constant (and possibly a modified version) didn’t match that ordering. So each logit index was being interpreted as the wrong semantic class, producing a stable permutation error (a cycle mapping each class to another). The pattern you described (pituitary -> negative, meningioma -> glioma, glioma -> pituitary, negative -> meningioma) is exactly what happens when label interpretation is shifted by a permutation."

Prompt: "How do I teach the model to stop taking the parts outside the MRI into account? GRAD CAM shows the reason for false negatives being it looking at the corners of the image, not the bounds of the brain".

LLM Response: --added RandomCornerMask to app/preprocessing.py, fixed __call__ method in BrainCrop--

Prompt: "Make the training a bit more verbose. Right now, I don't even know if it's running until an entire epoch is completed"

LLM Response: --added batch-level verbose logging to scripts/train.py--

Additionally, automatic code completion has been used during the code writing process (GPT 4.1). This model also assisted in writing the initial version of this README.md file and helps write commit messages.

Every substantial instance of AI assistance, whether by code completion or otherwise, has been disclosed in comments in the code itself with the NOTE tags `# NOTE: AI Generated` or `# NOTE: Fixed by AI`.

I hereby certify that I have disclosed the use of AI assistance in this project to the best of my ability.