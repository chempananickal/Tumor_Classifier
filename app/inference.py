from pathlib import Path
from typing import Dict, Any
import sys
import torch
from PIL import Image
import numpy as np

# Ensure project root on path when run under Streamlit (which sets CWD to repo root typically)
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.unet_densenet import get_model, ModelWithHooks, CLASSES as DEFAULT_CLASSES
from app.preprocessing import infer_transforms
from app.grad_cam import GradCAM, overlay_heatmap


class InferenceEngine:
    def __init__(self, weights_path: Path, device: str = 'cpu', enable_cam: bool = True, brain_crop: bool = True):
        self.device = torch.device(device)
        base_model = get_model(pretrained=False)
        self.model = ModelWithHooks(base_model) if enable_cam else base_model
        self.model = self.model.to(self.device)
        self.classes = list(DEFAULT_CLASSES)
        self._load_weights(weights_path)
        # Build inference transforms (optionally applying brain-centric crop)
        self.transforms = infer_transforms(brain_crop=brain_crop)
        self.enable_cam = enable_cam

    def _load_weights(self, weights_path: Path):
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        state = torch.load(weights_path, map_location=self.device)
        # Accept several serialization formats
        if isinstance(state, dict) and 'model_state' in state:
            state_dict = state['model_state']
            if 'classes' in state:
                self.classes = state['classes']
        elif isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state

        # Try direct load; on failure attempt key surgery
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # Collect model keys for debugging
            model_keys = list(self.model.state_dict().keys())[:5]
            sd_keys = list(state_dict.keys())[:5]
            # Heuristic: if keys lack 'model.' but our wrapper expects prefixed or vice versa.
            fixed = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    fixed[k.split('model.', 1)[-1]] = v
                else:
                    fixed[f'model.{k}'] = v
            try:
                self.model.load_state_dict(fixed, strict=False)
                print('[INFO] Loaded weights after adjusting key prefixes.')
            except Exception:
                raise RuntimeError(
                    'Failed to load weights. Original error: '\
                    f'{e}\nSample model keys: {model_keys}\nSample ckpt keys: {sd_keys}'
                )
        self.model.eval()

    def predict(self, image: Image.Image, return_cam: bool = True) -> Dict[str, Any]:
        img = image.convert('RGB')
        tensor = self.transforms(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
        result: Dict[str, Any] = {
            'class': self.classes[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probs': {c: float(p) for c, p in zip(self.classes, probs)},
        }
        if self.enable_cam and return_cam:
            cam_generator = GradCAM(self.model)
            cam, _ = cam_generator.generate(tensor, class_idx=pred_idx)
            np_img = np.array(img)
            heatmap = overlay_heatmap(cam, np_img)
            result['heatmap'] = heatmap
        return result
