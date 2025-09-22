import torch
import cv2
import numpy as np
from typing import Optional

class GradCAM: # NOTE: AI Generated
    """Grad-CAM implementation for visual explanations from CNN-based models.

    This class provides methods to generate Grad-CAM heatmaps for a given input image
    and overlay them on the original image. Ideally the heatmap should show where the model
    is "looking" to make its classification decision (where it thinks the tumor is).
    """
    def __init__(self, model_with_hooks, target_class: Optional[int] = None):
        self.model = model_with_hooks
        self.target_class = target_class

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        activation = self.model.activation  # [B, C, H, W]
        gradient = self.model.gradient      # [B, C, H, W]
        if activation is None or gradient is None:
            raise RuntimeError("Hooks did not capture activation/gradient")

        # Global average pool gradients
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy()
        # Normalize 0-1
        cam -= cam.min() if cam.min() != cam.max() else 0.0
        if cam.max() > 0:
            cam /= cam.max()
        return cam, logits.softmax(dim=1).detach().cpu().numpy()[0]


def overlay_heatmap(cam: np.ndarray, image_rgb: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a CAM (H,W) onto an RGB image (H,W,3). Returns blended RGB uint8."""
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)
    return blended
