from typing import Tuple, Optional
from torchvision import transforms
from PIL import Image
import numpy as np
import random

IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BrainCrop:
    """Naive brain region crop.

    Heuristic: convert to grayscale, threshold relative to max, find bounding box of
    non-background pixels (value > max*threshold_ratio) and crop with optional padding.
    Falls back to no-op if heuristic fails (e.g. completely dark image).
    """
    def __init__(self, threshold_ratio: float = 0.1, padding: int = 4):
        self.threshold_ratio = threshold_ratio
        self.padding = padding

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert('L'))
        max_val = arr.max()
        if max_val == 0:
            return img
        mask = arr > (max_val * self.threshold_ratio)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        y0 = max(y0 - self.padding, 0)
        x0 = max(x0 - self.padding, 0)
        y1 = min(y1 + self.padding, arr.shape[0]-1)
        x1 = min(x1 + self.padding, arr.shape[1]-1)
        return img.crop((x0, y0, x1 + 1, y1 + 1))


class RandomCornerMask: # NOTE: AI Generated
    """Randomly darken image corners to discourage model reliance on corner artifacts.

    Applies up to k corners (TL, TR, BL, BR) with a square mask sized as a fraction
    of the shorter side.
    """
    def __init__(self, prob: float = 0.25, max_corners: int = 2, size_frac: float = 0.18):
        self.prob = prob
        self.max_corners = max_corners
        self.size_frac = size_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        arr = np.array(img).copy()
        h, w = arr.shape[0], arr.shape[1]
        side = int(min(h, w) * self.size_frac)
        corners = []
        if side <= 1:
            return img
        coords = [('tl', 0, 0), ('tr', 0, w - side), ('bl', h - side, 0), ('br', h - side, w - side)]
        random.shuffle(coords)
        for name, y, x in coords[:random.randint(1, self.max_corners)]:
            arr[y:y+side, x:x+side] = 0
        return Image.fromarray(arr)


class CornerTextRemover:
    """Detect and remove (zero out) likely text/annotation overlays in image corners.

    Strategy (lightweight heuristic, no OCR):
    1. Convert to grayscale and optionally contrast stretch.
    2. For each corner patch (square based on fraction of min side), measure:
       - Mean intensity vs global mean.
       - Presence of high-contrast foreground pixels (bright on dark or dark on bright).
       - Fraction of pixels that are near binary extremes (0 or 255) suggesting rendered text.
    3. If heuristics exceed thresholds, zero out (or optionally blur) the patch to remove distractions.

    This is intentionally conservative to avoid erasing clinically relevant anatomy at edges.
    """
    def __init__(self, size_frac: float = 0.18, intensity_delta: float = 25.0,
                 extreme_frac: float = 0.20, min_contrast: float = 40.0, mode: str = 'zero'):
        self.size_frac = size_frac
        self.intensity_delta = intensity_delta
        self.extreme_frac = extreme_frac
        self.min_contrast = min_contrast
        self.mode = mode  # 'zero' | 'blur' (future)

    def _analyze_patch(self, patch: np.ndarray, global_mean: float) -> bool:
        mean = patch.mean()
        contrast = patch.max() - patch.min()
        # Extreme pixel fraction (either near black or near white)
        extreme = ((patch < 15) | (patch > 240)).sum() / patch.size
        # Heuristic triggers: patch substantially brighter/darker OR high extremes w/ contrast
        if abs(mean - global_mean) > self.intensity_delta and contrast > self.min_contrast:
            return True
        if extreme > self.extreme_frac and contrast > self.min_contrast:
            return True
        return False

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        gray = np.array(img.convert('L'))
        h, w = gray.shape
        side = int(min(h, w) * self.size_frac)
        if side < 8:  # too small to be meaningful
            return img
        global_mean = gray.mean()
        # Corner coordinates: (y0:y1, x0:x1)
        patches = {
            'tl': (0, side, 0, side),
            'tr': (0, side, w - side, w),
            'bl': (h - side, h, 0, side),
            'br': (h - side, h, w - side, w)
        }
        modified = False
        for name, (y0, y1, x0, x1) in patches.items():
            patch_gray = gray[y0:y1, x0:x1]
            if self._analyze_patch(patch_gray, global_mean):
                if self.mode == 'zero':
                    arr[y0:y1, x0:x1] = 0
                # Future: implement blur mode
                modified = True
        if not modified:
            return img
        return Image.fromarray(arr)


def build_transforms(train: bool, brain_crop: bool = False, corner_mask_prob: float = 0.0):
    ops = []
    # Remove typical text overlays first so cropping/masking focuses on anatomy.
    ops.append(CornerTextRemover())
    if brain_crop:
        ops.append(BrainCrop())
    ops.append(transforms.Resize(IMAGE_SIZE))
    if train:
        if corner_mask_prob > 0:
            ops.append(RandomCornerMask(prob=corner_mask_prob))
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def train_transforms(brain_crop: bool = False, corner_mask_prob: float = 0.0):
    return build_transforms(train=True, brain_crop=brain_crop, corner_mask_prob=corner_mask_prob)


def val_transforms(brain_crop: bool = False):
    return build_transforms(train=False, brain_crop=brain_crop)


def infer_transforms(brain_crop: bool = False):
    return val_transforms(brain_crop=brain_crop)
